# NeuralToneTransform Literature Review

This folder contains an IEEE LaTeX Literature Review for the Final Year Project **NeuralToneTransform**, focused on deep learning-based black-box audio modelling for instrument tone transformation.

## Files

- `main.tex`: IEEEtran conference-format LaTeX source.
- `references.bib`: BibTeX database for all cited references.
- `figures/`: Generated PDF figures used by the paper, plus editable SVG versions.
- `main.pdf`: Expected compiled output after running LaTeX.

## Compile

Run the following commands from this folder:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The expected output is:

```text
main.pdf
```

If a full TeX Live installation is not available, use an equivalent LaTeX compiler that supports `IEEEtran`, BibTeX, and PDF figure inclusion.

## Scope

This task only prepares the IEEE LaTeX Literature Review. It does not include source-code implementation, model training, dataset scripting, or changes to the existing project implementation.

## Reference Verification Note

The requested ten main references were used. No replacement references were needed. Zotero Desktop's local API was not running during preparation, so the bibliography was prepared directly from verified public academic and official software sources rather than exported from the local Zotero library.
