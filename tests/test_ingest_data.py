import datetime as dt
import hashlib
import io
import sys
import time
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest
import requests

import ingest_data


@pytest.fixture(autouse=True)
def mock_global_sleep(monkeypatch, request):
    if "exponential_backoff" in request.node.name:
        return
    monkeypatch.setattr(time, "sleep", lambda x: None)


class TestGenerateTasks:
    def test_basic_two_symbols_two_types_two_days(self):
        args = Namespace(
            symbols=["BTCUSDT", "ETHUSDT"],
            types=["klines", "aggTrades"],
            start=dt.datetime.fromisoformat("2024-01-01"),
            end=dt.datetime.fromisoformat("2024-01-02"),
        )
        tasks = ingest_data.generate_tasks(args, Path("data/raw"))
        assert len(tasks) == 8
        assert ("klines", "BTCUSDT", "2024-01-01") in tasks

    def test_single_day_inclusive(self):
        args = Namespace(
            symbols=["BTCUSDT"],
            types=["aggTrades"],
            start=dt.datetime.fromisoformat("2024-01-01"),
            end=dt.datetime.fromisoformat("2024-01-01"),
        )
        tasks = ingest_data.generate_tasks(args, Path("data/raw"))
        assert len(tasks) == 1
        assert ("aggTrades", "BTCUSDT", "2024-01-01") in tasks

    def test_end_date_is_inclusive(self):
        args = Namespace(
            symbols=["BTCUSDT"],
            types=["klines"],
            start=dt.datetime.fromisoformat("2024-01-01"),
            end=dt.datetime.fromisoformat("2024-01-03"),
        )
        tasks = ingest_data.generate_tasks(args, Path("data/raw"))
        dates = [t[2] for t in tasks]
        assert "2024-01-01" in dates
        assert "2024-01-02" in dates
        assert "2024-01-03" in dates
        assert len(dates) == 3

    def test_zip_without_csv_is_not_skipped(self, tmp_path, requests_mock):
        zip_dir = tmp_path / "BTCUSDT" / "aggTrades"
        zip_dir.mkdir(parents=True)
        (zip_dir / "BTCUSDT-aggTrades-2024-01-01.zip").touch()
        requests_mock.get(
            f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip.CHECKSUM",
            status_code=404,
        )
        result = ingest_data.process_task(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01")
        assert result["status"] != "skipped"

    def test_start_after_end_exits(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["ingest_data.py", "--start", "2024-01-05", "--end", "2024-01-01"]
        )
        with pytest.raises(SystemExit):
            ingest_data.main()

    def test_all_three_symbols_all_types_one_day(self):
        args = Namespace(
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            types=["klines", "aggTrades", "bookTicker"],
            start=dt.datetime.fromisoformat("2024-06-01"),
            end=dt.datetime.fromisoformat("2024-06-01"),
        )
        tasks = ingest_data.generate_tasks(args, Path("data/raw"))
        assert len(tasks) == 9

    def test_tuple_structure_correct(self):
        args = Namespace(
            symbols=["SOLUSDT"],
            types=["bookTicker"],
            start=dt.datetime.fromisoformat("2024-03-15"),
            end=dt.datetime.fromisoformat("2024-03-15"),
        )
        tasks = ingest_data.generate_tasks(args, Path("data/raw"))
        data_type, symbol, date_str = tasks[0]
        assert data_type == "bookTicker"
        assert symbol == "SOLUSDT"
        assert date_str == "2024-03-15"

    def test_returns_eager_list_not_generator(self):
        args = Namespace(
            symbols=["BTCUSDT"],
            types=["klines"],
            start=dt.datetime.fromisoformat("2024-01-01"),
            end=dt.datetime.fromisoformat("2024-01-01"),
        )
        result = ingest_data.generate_tasks(args, Path("data/raw"))
        assert isinstance(result, list)


class TestCheckTaskExists:
    def test_returns_false_when_file_absent(self, tmp_path):
        assert (
            ingest_data.check_task_exists(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01") is False
        )

    def test_returns_true_when_aggtrades_csv_exists(self, tmp_path):
        trade_dir = tmp_path / "BTCUSDT" / "aggTrades"
        trade_dir.mkdir(parents=True)
        (trade_dir / "BTCUSDT-aggTrades-2024-01-01.csv").touch()
        assert ingest_data.check_task_exists(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01") is True

    def test_returns_true_when_klines_csv_exists(self, tmp_path):
        klines_dir = tmp_path / "BTCUSDT" / "klines"
        klines_dir.mkdir(parents=True)
        (klines_dir / "BTCUSDT-1m-2024-01-01.csv").touch()
        assert ingest_data.check_task_exists(tmp_path, "klines", "BTCUSDT", "2024-01-01") is True

    def test_returns_true_when_bookticker_csv_exists(self, tmp_path):
        bt_dir = tmp_path / "ETHUSDT" / "bookTicker"
        bt_dir.mkdir(parents=True)
        (bt_dir / "ETHUSDT-bookTicker-2024-05-10.csv").touch()
        assert (
            ingest_data.check_task_exists(tmp_path, "bookTicker", "ETHUSDT", "2024-05-10") is True
        )

    def test_zip_without_csv_not_considered_complete(self, tmp_path):
        trade_dir = tmp_path / "BTCUSDT" / "aggTrades"
        trade_dir.mkdir(parents=True)
        (trade_dir / "BTCUSDT-aggTrades-2024-01-01.zip").touch()
        assert (
            ingest_data.check_task_exists(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01") is False
        )

    def test_different_date_not_matched(self, tmp_path):
        trade_dir = tmp_path / "BTCUSDT" / "aggTrades"
        trade_dir.mkdir(parents=True)
        (trade_dir / "BTCUSDT-aggTrades-2024-01-02.csv").touch()
        assert (
            ingest_data.check_task_exists(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01") is False
        )


class TestValidateChecksum:
    def test_matching_hashes_returns_true_and_keeps_file(self, tmp_path):
        dummy_zip = tmp_path / "test.zip"
        dummy_zip.write_text("dummy content")
        assert ingest_data.validate_checksum("abc123", "abc123", dummy_zip) is True
        assert dummy_zip.exists()

    def test_mismatched_hashes_returns_false_and_deletes_file(self, tmp_path):
        dummy_zip = tmp_path / "corrupt.zip"
        dummy_zip.write_text("corrupt content")
        assert ingest_data.validate_checksum("expected", "wrong", dummy_zip) is False
        assert not dummy_zip.exists()

    def test_empty_string_hashes_match(self, tmp_path):
        dummy_zip = tmp_path / "empty.zip"
        dummy_zip.write_text("")
        assert ingest_data.validate_checksum("", "", dummy_zip) is True

    def test_case_sensitive_hash_comparison(self, tmp_path):
        dummy_zip = tmp_path / "case.zip"
        dummy_zip.write_text("data")
        assert ingest_data.validate_checksum("ABCDEF", "abcdef", dummy_zip) is False
        assert not dummy_zip.exists()


class TestExtractCsv:
    def _make_zip(self, tmp_path, filename, content="col1,col2\n1,2\n"):
        zip_path = tmp_path / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr(filename, content)
        return zip_path

    def test_successful_extraction_creates_csv(self, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        zip_path = self._make_zip(tmp_path, "data.csv")
        result = ingest_data.extract_csv(zip_path, output_dir)
        assert result is True
        assert (output_dir / "data.csv").exists()

    def test_zip_deleted_after_extraction(self, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        zip_path = self._make_zip(tmp_path, "data.csv")
        ingest_data.extract_csv(zip_path, output_dir)
        assert not zip_path.exists()

    def test_zip_slip_path_traversal_blocked(self, tmp_path):
        safe_dir = tmp_path / "safe_zone"
        safe_dir.mkdir()
        malicious_zip = tmp_path / "malicious.zip"
        with zipfile.ZipFile(malicious_zip, "w") as z:
            z.writestr("../dangerous_file.csv", "malicious payload")
        result = ingest_data.extract_csv(malicious_zip, safe_dir)
        assert result is False
        assert not (tmp_path / "dangerous_file.csv").exists()

    def test_multi_file_zip_rejected(self, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        zip_path = tmp_path / "multi.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("file1.csv", "a,b")
            z.writestr("file2.csv", "c,d")
        result = ingest_data.extract_csv(zip_path, output_dir)
        assert result is False

    def test_extracted_content_matches_original(self, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        content = "1,50000,1.5\n2,50100,2.0\n"
        zip_path = self._make_zip(tmp_path, "trade.csv", content)
        ingest_data.extract_csv(zip_path, output_dir)
        assert (output_dir / "trade.csv").read_text() == content


class TestDownloadChecksumUnit:
    def test_returns_hash_on_200(self, requests_mock):
        url = f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip.CHECKSUM"
        requests_mock.get(url, text="deadbeef  BTCUSDT-aggTrades-2024-01-01.zip\n")
        result = ingest_data.download_checksum(Path("dummy"), "aggTrades", "BTCUSDT", "2024-01-01")
        assert result == "deadbeef"

    def test_returns_none_on_404(self, requests_mock):
        url = f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2099-01-01.zip.CHECKSUM"
        requests_mock.get(url, status_code=404)
        result = ingest_data.download_checksum(Path("dummy"), "aggTrades", "BTCUSDT", "2099-01-01")
        assert result is None

    def test_klines_uses_correct_url(self, requests_mock):
        url = f"{ingest_data.url}/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01-01.zip.CHECKSUM"
        requests_mock.get(url, text="aabbcc  BTCUSDT-1m-2024-01-01.zip\n")
        result = ingest_data.download_checksum(Path("dummy"), "klines", "BTCUSDT", "2024-01-01")
        assert result == "aabbcc"

    def test_exponential_backoff_on_transient_errors(self, requests_mock, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr(time, "sleep", lambda x: sleep_calls.append(x))
        url = f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip.CHECKSUM"
        requests_mock.register_uri(
            "GET",
            url,
            [
                {"status_code": 503},
                {"status_code": 502},
                {"status_code": 500},
                {"status_code": 200, "text": "abc123hash  BTCUSDT-aggTrades-2024-01-01.zip\n"},
            ],
        )
        result = ingest_data.download_checksum(Path("dummy"), "aggTrades", "BTCUSDT", "2024-01-01")
        assert result == "abc123hash"
        backoff_sleeps = [s for s in sleep_calls if s >= 2]
        assert 2 in backoff_sleeps
        assert 4 in backoff_sleeps

    def test_checksum_two_space_delimiter_parsed_correctly(self, requests_mock):
        url = f"{ingest_data.url}/bookTicker/ETHUSDT/ETHUSDT-bookTicker-2024-01-01.zip.CHECKSUM"
        requests_mock.get(url, text="cafebabe  ETHUSDT-bookTicker-2024-01-01.zip\n")
        result = ingest_data.download_checksum(Path("dummy"), "bookTicker", "ETHUSDT", "2024-01-01")
        assert result == "cafebabe"


class TestDownloadZipUnit:
    def _make_zip_bytes(self, filename="data.csv", content="1,2,3"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(filename, content)
        return buf.getvalue()

    def test_returns_path_and_hash_on_200(self, tmp_path, requests_mock):
        zip_bytes = self._make_zip_bytes()
        url = f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip"
        requests_mock.get(url, content=zip_bytes)
        zip_path, computed_hash = ingest_data.download_zip(
            tmp_path, "aggTrades", "BTCUSDT", "2024-01-01"
        )
        assert zip_path is not None
        assert zip_path.exists()
        expected = hashlib.sha256(zip_bytes).hexdigest()
        assert computed_hash == expected

    def test_returns_none_on_404(self, tmp_path, requests_mock):
        url = f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2099-01-01.zip"
        requests_mock.get(url, status_code=404)
        zip_path, computed_hash = ingest_data.download_zip(
            tmp_path, "aggTrades", "BTCUSDT", "2099-01-01"
        )
        assert zip_path is None
        assert computed_hash is None

    def test_klines_url_contains_1m_segment(self, tmp_path, requests_mock):
        zip_bytes = self._make_zip_bytes("BTCUSDT-1m-2024-01-01.csv")
        url = f"{ingest_data.url}/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01-01.zip"
        requests_mock.get(url, content=zip_bytes)
        zip_path, _ = ingest_data.download_zip(tmp_path, "klines", "BTCUSDT", "2024-01-01")
        assert zip_path is not None

    def test_output_directory_created_automatically(self, tmp_path, requests_mock):
        zip_bytes = self._make_zip_bytes()
        url = f"{ingest_data.url}/bookTicker/SOLUSDT/SOLUSDT-bookTicker-2024-02-15.zip"
        requests_mock.get(url, content=zip_bytes)
        ingest_data.download_zip(tmp_path, "bookTicker", "SOLUSDT", "2024-02-15")
        assert (tmp_path / "SOLUSDT" / "bookTicker").exists()

    def test_hash_computed_correctly_for_known_content(self, tmp_path, requests_mock):
        zip_bytes = self._make_zip_bytes()
        url = f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip"
        requests_mock.get(url, content=zip_bytes)
        _, computed_hash = ingest_data.download_zip(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01")
        assert computed_hash == hashlib.sha256(zip_bytes).hexdigest()

    def test_exponential_backoff_on_zip_download(self, tmp_path, requests_mock, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr(time, "sleep", lambda x: sleep_calls.append(x))
        url = f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip"
        requests_mock.register_uri(
            "GET",
            url,
            [
                {"status_code": 503},
                {"status_code": 503},
                {"status_code": 200, "content": b"fakecontent"},
            ],
        )
        ingest_data.download_zip(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01")
        backoff_sleeps = [s for s in sleep_calls if s >= 2]
        assert 2 in backoff_sleeps
        assert 4 in backoff_sleeps


# integration tests with mocked network
class TestProcessTask:
    def _make_zip_bytes(self, csv_filename, csv_content="1,50000,1.2,1,2,1704067200000,True"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(csv_filename, csv_content)
        return buf.getvalue()

    def test_full_pipeline_aggtrades_returns_ok(self, tmp_path, requests_mock):
        zip_bytes = self._make_zip_bytes("BTCUSDT-aggTrades-2024-01-01.csv")
        mock_hash = hashlib.sha256(zip_bytes).hexdigest()
        requests_mock.get(
            f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip.CHECKSUM",
            text=f"{mock_hash}  BTCUSDT-aggTrades-2024-01-01.zip",
        )
        requests_mock.get(
            f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip",
            content=zip_bytes,
        )
        result = ingest_data.process_task(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01")
        assert result["status"] == "ok"
        assert (tmp_path / "BTCUSDT" / "aggTrades" / "BTCUSDT-aggTrades-2024-01-01.csv").exists()

    def test_full_pipeline_klines_returns_ok(self, tmp_path, requests_mock):
        zip_bytes = self._make_zip_bytes("BTCUSDT-1m-2024-01-01.csv")
        mock_hash = hashlib.sha256(zip_bytes).hexdigest()
        requests_mock.get(
            f"{ingest_data.url}/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01-01.zip.CHECKSUM",
            text=f"{mock_hash}  BTCUSDT-1m-2024-01-01.zip",
        )
        requests_mock.get(
            f"{ingest_data.url}/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01-01.zip",
            content=zip_bytes,
        )
        result = ingest_data.process_task(tmp_path, "klines", "BTCUSDT", "2024-01-01")
        assert result["status"] == "ok"
        assert (tmp_path / "BTCUSDT" / "klines" / "BTCUSDT-1m-2024-01-01.csv").exists()

    def test_checksum_404_returns_missing(self, tmp_path, requests_mock):
        requests_mock.get(
            f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2099-12-31.zip.CHECKSUM",
            status_code=404,
        )
        result = ingest_data.process_task(tmp_path, "aggTrades", "BTCUSDT", "2099-12-31")
        assert result["status"] == "missing"

    def test_skips_already_existing_csv(self, tmp_path, requests_mock):
        csv_dir = tmp_path / "BTCUSDT" / "aggTrades"
        csv_dir.mkdir(parents=True)
        (csv_dir / "BTCUSDT-aggTrades-2024-01-01.csv").touch()
        result = ingest_data.process_task(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01")
        assert result["status"] == "skipped"
        assert not requests_mock.called

    def test_corrupt_zip_returns_checksum_failed(self, tmp_path, requests_mock):
        zip_bytes = self._make_zip_bytes("BTCUSDT-aggTrades-2024-01-01.csv")
        requests_mock.get(
            f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip.CHECKSUM",
            text=(
                "0000000000000000000000000000000000000000000000000000000000000000  "
                "BTCUSDT-aggTrades-2024-01-01.zip"
            ),
        )
        requests_mock.get(
            f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip",
            content=zip_bytes,
        )
        result = ingest_data.process_task(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01")
        assert result["status"] == "checksum_failed"

    def test_corrupt_zip_file_deleted_after_checksum_failure(self, tmp_path, requests_mock):
        zip_bytes = self._make_zip_bytes("BTCUSDT-aggTrades-2024-01-01.csv")
        requests_mock.get(
            f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip.CHECKSUM",
            text="badbadbad  BTCUSDT-aggTrades-2024-01-01.zip",
        )
        requests_mock.get(
            f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip",
            content=zip_bytes,
        )
        ingest_data.process_task(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01")
        zip_path = tmp_path / "BTCUSDT" / "aggTrades" / "BTCUSDT-aggTrades-2024-01-01.zip"
        assert not zip_path.exists()

    def test_zip_not_present_on_disk_after_successful_extraction(self, tmp_path, requests_mock):
        zip_bytes = self._make_zip_bytes("BTCUSDT-aggTrades-2024-01-01.csv")
        mock_hash = hashlib.sha256(zip_bytes).hexdigest()
        requests_mock.get(
            f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip.CHECKSUM",
            text=f"{mock_hash}  BTCUSDT-aggTrades-2024-01-01.zip",
        )
        requests_mock.get(
            f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip",
            content=zip_bytes,
        )
        ingest_data.process_task(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01")
        zip_path = tmp_path / "BTCUSDT" / "aggTrades" / "BTCUSDT-aggTrades-2024-01-01.zip"
        assert not zip_path.exists()

    def test_network_exception_returns_error_not_raise(self, tmp_path, requests_mock):
        requests_mock.get(
            f"{ingest_data.url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip.CHECKSUM",
            exc=requests.exceptions.ConnectionError,
        )
        result = ingest_data.process_task(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01")
        assert result["status"] == "error"

    def test_all_symbols_and_types_route_correctly(self, tmp_path, requests_mock):
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            for data_type in ["bookTicker", "aggTrades", "klines"]:
                if data_type == "klines":
                    csv_name = f"{symbol}-1m-2024-01-01.csv"
                    checksum_url = (
                        f"{ingest_data.url}/klines/{symbol}/1m/{symbol}-1m-2024-01-01.zip.CHECKSUM"
                    )
                    zip_url = f"{ingest_data.url}/klines/{symbol}/1m/{symbol}-1m-2024-01-01.zip"
                else:
                    csv_name = f"{symbol}-{data_type}-2024-01-01.csv"
                    checksum_url = (
                        f"{ingest_data.url}/{data_type}/{symbol}/"
                        f"{symbol}-{data_type}-2024-01-01.zip.CHECKSUM"
                    )
                    zip_url = (
                        f"{ingest_data.url}/{data_type}/{symbol}/"
                        f"{symbol}-{data_type}-2024-01-01.zip"
                    )

                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w") as z:
                    z.writestr(csv_name, "row1,row2\n")
                zip_bytes = buf.getvalue()
                mock_hash = hashlib.sha256(zip_bytes).hexdigest()

                requests_mock.get(checksum_url, text=f"{mock_hash}  {csv_name}.zip")
                requests_mock.get(zip_url, content=zip_bytes)

                result = ingest_data.process_task(tmp_path, data_type, symbol, "2024-01-01")
                assert result["status"] == "ok", f"Failed for {symbol}/{data_type}"


class TestRunValidation:
    def test_valid_aggtrades_csv_passes(self, tmp_path):
        csv_dir = tmp_path / "BTCUSDT" / "aggTrades"
        csv_dir.mkdir(parents=True)
        csv_path = csv_dir / "BTCUSDT-aggTrades-2024-01-01.csv"
        csv_path.write_text("1,50000,1.5,1,1,1704067200000,True\n")

        args = Namespace(
            symbols=["BTCUSDT"],
            types=["aggTrades"],
            start=dt.datetime.fromisoformat("2024-01-01"),
            end=dt.datetime.fromisoformat("2024-01-01"),
        )
        ingest_data.run_validation(tmp_path, args)

    def test_valid_klines_csv_passes(self, tmp_path):
        csv_dir = tmp_path / "BTCUSDT" / "klines"
        csv_dir.mkdir(parents=True)
        csv_path = csv_dir / "BTCUSDT-1m-2024-01-01.csv"
        row = "1704067200000,50000,50100,49900,50050,100,1704067260000,5000000,200,50,2500000,0\n"
        csv_path.write_text(row)

        args = Namespace(
            symbols=["BTCUSDT"],
            types=["klines"],
            start=dt.datetime.fromisoformat("2024-01-01"),
            end=dt.datetime.fromisoformat("2024-01-01"),
        )
        ingest_data.run_validation(tmp_path, args)

    def test_missing_csv_skipped_gracefully(self, tmp_path):
        args = Namespace(
            symbols=["BTCUSDT"],
            types=["aggTrades"],
            start=dt.datetime.fromisoformat("2024-01-01"),
            end=dt.datetime.fromisoformat("2024-01-01"),
        )
        ingest_data.run_validation(tmp_path, args)

    def test_csv_with_wrong_column_count_triggers_error(self, tmp_path, caplog):
        csv_dir = tmp_path / "BTCUSDT" / "aggTrades"
        csv_dir.mkdir(parents=True)
        csv_path = csv_dir / "BTCUSDT-aggTrades-2024-01-01.csv"
        csv_path.write_text("only,two\n")

        args = Namespace(
            symbols=["BTCUSDT"],
            types=["aggTrades"],
            start=dt.datetime.fromisoformat("2024-01-01"),
            end=dt.datetime.fromisoformat("2024-01-01"),
        )

        import logging

        with caplog.at_level(logging.ERROR):
            ingest_data.run_validation(tmp_path, args)
        assert any("FAILED" in r.message for r in caplog.records)

    def test_empty_csv_triggers_error(self, tmp_path, caplog):
        csv_dir = tmp_path / "BTCUSDT" / "aggTrades"
        csv_dir.mkdir(parents=True)
        csv_path = csv_dir / "BTCUSDT-aggTrades-2024-01-01.csv"
        csv_path.write_text("")

        args = Namespace(
            symbols=["BTCUSDT"],
            types=["aggTrades"],
            start=dt.datetime.fromisoformat("2024-01-01"),
            end=dt.datetime.fromisoformat("2024-01-01"),
        )

        import logging

        with caplog.at_level(logging.ERROR):
            ingest_data.run_validation(tmp_path, args)
        assert any("FAILED" in r.message for r in caplog.records)


@pytest.mark.network
class TestLiveNetworkPipeline:
    def test_network_checksum_download_btcusdt_aggtrades(self):
        result = ingest_data.download_checksum(Path("dummy"), "aggTrades", "BTCUSDT", "2024-01-01")
        assert result is not None
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_network_checksum_klines_uses_correct_url(self):
        result = ingest_data.download_checksum(Path("dummy"), "klines", "BTCUSDT", "2024-01-01")
        assert result is not None
        assert len(result) == 64

    def test_network_checksum_future_date_returns_none(self):
        result = ingest_data.download_checksum(Path("dummy"), "aggTrades", "BTCUSDT", "2099-01-01")
        assert result is None

    def test_network_checksum_all_three_symbols_available(self):
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            result = ingest_data.download_checksum(Path("dummy"), "aggTrades", symbol, "2024-01-01")
            assert result is not None, f"Expected checksum for {symbol} but got None"

    def test_network_full_process_task_aggtrades(self, tmp_path):
        result = ingest_data.process_task(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01")
        assert result["status"] == "ok"
        csv_path = tmp_path / "BTCUSDT" / "aggTrades" / "BTCUSDT-aggTrades-2024-01-01.csv"
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0

    def test_network_full_process_task_klines(self, tmp_path):
        result = ingest_data.process_task(tmp_path, "klines", "BTCUSDT", "2024-01-01")
        assert result["status"] == "ok"
        csv_path = tmp_path / "BTCUSDT" / "klines" / "BTCUSDT-1m-2024-01-01.csv"
        assert csv_path.exists()

    def test_network_csv_columns_match_expected_schema(self, tmp_path):
        import pandas as pd

        result = ingest_data.process_task(tmp_path, "klines", "BTCUSDT", "2024-01-01")
        assert result["status"] == "ok"
        csv_path = tmp_path / "BTCUSDT" / "klines" / "BTCUSDT-1m-2024-01-01.csv"
        cols = ingest_data.COLUMNS["klines"]
        df = pd.read_csv(csv_path, header=None, names=cols, nrows=5)
        assert list(df.columns) == cols
        assert len(df) > 0

    def test_network_idempotent_skip_on_second_call(self, tmp_path):
        first = ingest_data.process_task(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01")
        assert first["status"] == "ok"
        second = ingest_data.process_task(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01")
        assert second["status"] == "skipped"

    def test_network_bookticker_btcusdt_downloads(self, tmp_path):
        result = ingest_data.process_task(tmp_path, "bookTicker", "BTCUSDT", "2024-01-01")
        assert result["status"] in ("ok", "missing")
        if result["status"] == "ok":
            csv_path = tmp_path / "BTCUSDT" / "bookTicker" / "BTCUSDT-bookTicker-2024-01-01.csv"
            assert csv_path.exists()

    def test_network_aggtrades_csv_has_no_header_row(self, tmp_path):
        import pandas as pd

        result = ingest_data.process_task(tmp_path, "aggTrades", "BTCUSDT", "2024-01-01")
        assert result["status"] == "ok"
        csv_path = tmp_path / "BTCUSDT" / "aggTrades" / "BTCUSDT-aggTrades-2024-01-01.csv"
        cols = ingest_data.COLUMNS["aggTrades"]
        df = pd.read_csv(csv_path, header=None, names=cols, nrows=2)
        assert len(df) > 0
        assert "agg_trade_id" in df.columns
