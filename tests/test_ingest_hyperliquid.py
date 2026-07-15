import datetime as dt
import json
import time
from argparse import Namespace
from pathlib import Path

import boto3
import lz4.frame
import pytest
from botocore.exceptions import EndpointConnectionError
from moto import mock_aws

import order_flow_imbalance_strategy.ingest_hyperliquid as ih


# A minimal two-snapshot NDJSON payload in the REAL archive schema (confirmed by
# a live probe — see scripts/probe_hyperliquid_format.py): the WebSocket l2Book
# message nested under "raw", plus the archive's capture "time" and "ver_num".
def _archive_line(event_ms: int, bid_px: str, ask_px: str) -> dict:
    return {
        "time": "2024-01-01T00:00:00.000000000",
        "ver_num": 1,
        "raw": {
            "channel": "l2Book",
            "data": {
                "coin": "BTC",
                "time": event_ms,
                "levels": [
                    [{"px": bid_px, "sz": "1.0", "n": 3}],
                    [{"px": ask_px, "sz": "0.5", "n": 2}],
                ],
            },
        },
    }


SNAPSHOT_LINES = [
    _archive_line(1704067200000, "42000", "42010"),
    _archive_line(1704067200100, "42001", "42011"),
]
RAW_NDJSON = ("\n".join(json.dumps(s) for s in SNAPSHOT_LINES) + "\n").encode("utf-8")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch, request):
    if "backoff" in request.node.name:
        return
    monkeypatch.setattr(time, "sleep", lambda x: None)


@pytest.fixture
def s3_client():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=ih.BUCKET)
        yield client


def _put_object(client, coin, date_str, hour, data=RAW_NDJSON):
    client.put_object(Bucket=ih.BUCKET, Key=ih.s3_key(coin, date_str, hour), Body=data)


# --- key / path builders ---------------------------------------------------


class TestKeyAndPath:
    def test_key_uses_compact_date_and_unpadded_hour(self):
        assert ih.s3_key("BTC", "2024-01-01", 9) == "market_data/20240101/9/l2Book/BTC.lz4"

    def test_key_hour_zero(self):
        assert ih.s3_key("BTC", "2024-01-01", 0) == "market_data/20240101/0/l2Book/BTC.lz4"

    def test_output_path_zero_pads_hour(self, tmp_path):
        p = ih.output_path(tmp_path, "BTC", "2024-01-01", 9)
        assert p.name == "BTC-l2Book-2024-01-01-09.jsonl"
        assert p.parent == Path(tmp_path) / "BTC" / "l2Book"


# --- task generation -------------------------------------------------------


class TestGenerateTasks:
    def test_single_day_is_24_hours(self):
        args = Namespace(
            symbols=["BTC"],
            start=dt.datetime.fromisoformat("2024-01-01"),
            end=dt.datetime.fromisoformat("2024-01-01"),
        )
        tasks = ih.generate_tasks(args)
        assert len(tasks) == 24
        assert ("BTC", "2024-01-01", 0) in tasks
        assert ("BTC", "2024-01-01", 23) in tasks

    def test_end_date_inclusive_and_multi_symbol(self):
        args = Namespace(
            symbols=["BTC", "ETH"],
            start=dt.datetime.fromisoformat("2024-01-01"),
            end=dt.datetime.fromisoformat("2024-01-03"),
        )
        tasks = ih.generate_tasks(args)
        assert len(tasks) == 2 * 3 * 24
        assert isinstance(tasks, list)


# --- idempotency -----------------------------------------------------------


class TestCheckTaskExists:
    def test_false_when_absent(self, tmp_path):
        assert ih.check_task_exists(tmp_path, "BTC", "2024-01-01", 0) is False

    def test_true_when_present(self, tmp_path):
        p = ih.output_path(tmp_path, "BTC", "2024-01-01", 0)
        p.parent.mkdir(parents=True)
        p.write_text('{"a": 1}\n')
        assert ih.check_task_exists(tmp_path, "BTC", "2024-01-01", 0) is True

    def test_undersized_file_not_complete(self, tmp_path):
        p = ih.output_path(tmp_path, "BTC", "2024-01-01", 0)
        p.parent.mkdir(parents=True)
        p.write_text("")  # 0 bytes, crash artifact
        assert ih.check_task_exists(tmp_path, "BTC", "2024-01-01", 0) is False


# --- validation ------------------------------------------------------------


class TestValidateExtracted:
    def test_valid_ndjson_passes(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_bytes(RAW_NDJSON)
        assert ih.validate_extracted(p) is True
        assert p.exists()

    def test_empty_file_fails_and_is_deleted(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text("")
        assert ih.validate_extracted(p) is False
        assert not p.exists()

    def test_non_json_first_line_fails_and_is_deleted(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text("not json at all\n")
        assert ih.validate_extracted(p) is False
        assert not p.exists()

    def test_json_array_first_line_rejected(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text("[1, 2, 3]\n")  # valid JSON but not a snapshot object
        assert ih.validate_extracted(p) is False


# --- download + extract (mocked S3) ----------------------------------------


class TestDownloadAndExtract:
    def test_lz4_stream_decompressed_to_jsonl(self, tmp_path, s3_client):
        _put_object(s3_client, "BTC", "2024-01-01", 0, data=lz4.frame.compress(RAW_NDJSON))
        out = ih.download_and_extract(s3_client, tmp_path, "BTC", "2024-01-01", 0)
        assert out is not None and out.exists()
        assert out.read_bytes() == RAW_NDJSON

    def test_missing_key_returns_none(self, tmp_path, s3_client):
        out = ih.download_and_extract(s3_client, tmp_path, "BTC", "2099-01-01", 0)
        assert out is None
        assert not ih.output_path(tmp_path, "BTC", "2099-01-01", 0).exists()

    def test_transient_error_retries_with_backoff(self, tmp_path, monkeypatch):
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda x: sleeps.append(x))
        calls = {"n": 0}
        good = lz4.frame.compress(RAW_NDJSON)

        class FakeBody:
            def iter_chunks(self, chunk_size):
                yield good

        class FakeClient:
            def get_object(self, **kwargs):
                calls["n"] += 1
                if calls["n"] < 3:
                    raise EndpointConnectionError(endpoint_url="https://s3")
                return {"Body": FakeBody()}

        out = ih.download_and_extract(FakeClient(), tmp_path, "BTC", "2024-01-01", 0)
        assert out is not None and out.read_bytes() == RAW_NDJSON
        assert calls["n"] == 3
        assert 2 in sleeps and 4 in sleeps

    def test_partial_tmp_cleaned_on_missing(self, tmp_path, s3_client):
        # A prior crash left a .tmp; a subsequent missing key must not leave it behind.
        out = ih.output_path(tmp_path, "BTC", "2099-01-01", 0)
        out.parent.mkdir(parents=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text("partial")
        ih.download_and_extract(s3_client, tmp_path, "BTC", "2099-01-01", 0)
        assert not tmp.exists()


# --- process_task orchestration --------------------------------------------


class TestProcessTask:
    def test_full_pipeline_ok(self, tmp_path, s3_client):
        _put_object(s3_client, "BTC", "2024-01-01", 0, data=lz4.frame.compress(RAW_NDJSON))
        result = ih.process_task(tmp_path, "BTC", "2024-01-01", 0, client=s3_client)
        assert result["status"] == "ok"
        assert ih.output_path(tmp_path, "BTC", "2024-01-01", 0).exists()

    def test_missing_object(self, tmp_path, s3_client):
        result = ih.process_task(tmp_path, "BTC", "2099-01-01", 0, client=s3_client)
        assert result["status"] == "missing"

    def test_skips_existing(self, tmp_path, s3_client):
        p = ih.output_path(tmp_path, "BTC", "2024-01-01", 0)
        p.parent.mkdir(parents=True)
        p.write_bytes(RAW_NDJSON)
        result = ih.process_task(tmp_path, "BTC", "2024-01-01", 0, client=s3_client)
        assert result["status"] == "skipped"

    def test_malformed_payload(self, tmp_path, s3_client):
        _put_object(s3_client, "BTC", "2024-01-01", 1, data=lz4.frame.compress(b"garbage\n"))
        result = ih.process_task(tmp_path, "BTC", "2024-01-01", 1, client=s3_client)
        assert result["status"] == "malformed"
        assert not ih.output_path(tmp_path, "BTC", "2024-01-01", 1).exists()

    def test_exception_returns_error_not_raise(self, tmp_path):
        class Boom:
            def get_object(self, **kwargs):
                raise RuntimeError("unexpected")

        result = ih.process_task(tmp_path, "BTC", "2024-01-01", 0, client=Boom())
        assert result["status"] == "error"


# --- start > end guard -----------------------------------------------------


def test_start_after_end_exits(monkeypatch):
    import sys

    monkeypatch.setattr(
        sys, "argv", ["ingest_hyperliquid.py", "--start", "2024-01-05", "--end", "2024-01-01"]
    )
    with pytest.raises(SystemExit):
        ih.main()
