# GitHub repository and local setup

This project is hosted at [MrlameChicken/photo-select](https://github.com/MrlameChicken/photo-select).

```bash
git clone https://github.com/MrlameChicken/photo-select.git
cd photo-select
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

**Privacy:** This repository is currently public. Never commit family photos, face reference images, model weights, cached SQLite databases, or export manifests with local paths. The `.gitignore` excludes common local directories and extensions; store personal images **outside the checkout** to avoid accidentally tracking them. To make the *repository* private, change its visibility in GitHub repository settings. The program itself processes photos locally.

For usage commands see [README](../README.md); for design diagrams see [ARCHITECTURE](ARCHITECTURE.md).
