# GitHub publish checklist

This archive is prepared for **manual upload**. It has not been pushed to any GitHub account.

## Review before publishing

1. Keep all original large/private assets outside this repository: actual `data/`, `.env`, PDF corpus, Mongo backup, indexed vectors, pretrained/fine-tuned weights, teacher labels, DPO Preference Pairs, API tokens. Only `examples/` contains synthetic public samples.
2. Review [`docs/STATUS.md`](STATUS.md) and avoid describing the synthetic smoke checks as real model or index validation.
3. Run `python -m pip install -e '.[test]' && python -m pytest -q` and `python -m controlled_rag.demo`, `python eval/run_all.py` and `python eval/safety/run_control_probes.py` in the repository root.
4. Check tracked content rather than trusting `.gitignore`: `git status --short`, `git diff --cached --stat`, `git ls-files`. If secrets/data were committed to Git previously, `.gitignore` alone will not erase the history.
5. Add your preferred project license only if you own the required rights; the initial upload's README license badge is not evidence of a valid LICENSE file, so this archive does not invent one.

## New repository

```bash
cd Controlled-Agentic-RAG-GitHub
python -m pip install -e '.[test]'
python -m pytest -q

git init
git add .
git status --short
# Look at the staged files and remove anything you don't want published.
git commit -m "Add controlled agentic RAG reference implementation"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPO.git
git push -u origin main
```

If merging into an **existing** repository, use a review branch and inspect the diff instead of blindly overwriting history. Any private source data should remain private. This package intentionally excludes full training datasets and checkpoints.
