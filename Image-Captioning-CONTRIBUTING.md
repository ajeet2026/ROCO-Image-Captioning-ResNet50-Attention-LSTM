# Contributing to Image Captioning using Deep Learning

Thanks for your interest. This started as a coursework/portfolio project, so contributions are welcome but the workflow is intentionally simple since it's notebook-based rather than a packaged library.

## Development setup

```bash
git clone https://github.com/Ritik650/Image-Captioning-using-Deep-Learning.git
cd Image-Captioning-using-Deep-Learning

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install notebook
jupyter notebook
```

You'll need the ROCO dataset downloaded locally — see the [Dataset](README.md#dataset) section in the README.

## Before opening a PR

- Run the notebook end-to-end (Preprocessing → Training → Evaluation → Caption generation) and confirm it executes without errors on a clean environment.
- If your change affects reported metrics (BLEU/ROUGE/METEOR), include the before/after numbers in your PR description and update the [Results](README.md#results) table in the same PR.
- Clear notebook output cells before committing unless the output itself is the point of the change (e.g., updating the qualitative examples in Results) — keeps diffs reviewable.

## Ideas for contributions

- Filling in real qualitative examples (input image → generated caption vs. ground truth) in the Results section — currently a placeholder.
- Additional evaluation metrics or a held-out test split breakdown by image category.
- Refactoring shared preprocessing/training code out of the notebook into importable modules, if the project grows beyond notebook-only use.

## Security issues

Please don't open a public issue for security concerns — see [SECURITY.md](SECURITY.md) instead.
