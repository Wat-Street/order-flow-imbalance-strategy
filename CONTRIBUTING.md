# Contributing

## Workflow

1. Create a branch from `main`: `feat/…`, `bugfix/…`, or `refactor/…`
2. Make your changes and commit
3. Run `./scripts/check.sh`
4. Open a PR to `main`

Do not push directly to `main`. Every PR needs CI to pass and one approval before merge.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pytest ruff
chmod +x scripts/check.sh
```

## Pull requests

- Clear title and a short description of what changed
- Link issues when relevant: `Closes #123`

## Protecting `main` (maintainers, one-time)

**Settings → Rules → Rulesets** → create a rule for `main`:

- Require pull request + **1 approval**
- Require status check **Lint & Test** *(shows up after the first PR runs CI)*
- Block force pushes
