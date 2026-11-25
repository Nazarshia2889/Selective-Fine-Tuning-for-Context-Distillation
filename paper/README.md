# Paper (NeurIPS 2015 style)

Upload this whole folder to Overleaf (New Project → Upload Project, as a zip) and compile `main.tex` with pdfLaTeX.

- `nips15submit_e.sty`: the official 2015 style file. `main.tex` is based on the official `nips2015.tex` template, with `\nipsfinalcopy` on so names show.
- Red `[TODO: ...]` notes mark text still to write. `\renewcommand{\todo}[1]{}` in the preamble hides them all.
- `refs.bib`: candidate references. Check every entry before submitting.
- `figures/*.pdf` and `tables/*.tex` are generated. Don't edit them by hand; regenerate them from the repo root:

```
pip install -e ".[paper,modal]"
python scripts/export_results.py      # download results from the Modal volume -> results/paper/*.csv
python scripts/make_figures.py        # results/paper/*.csv -> paper/figures/*.pdf, paper/tables/*.tex
```

The CSVs in `results/paper/` are committed, so `make_figures.py` runs without Modal access.
