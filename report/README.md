# Course report

`main.tex` is the project paper in the official DLAI 2025/2026 template. Its
scientific body ends at the explicit `\clearpage` before the mandatory AI-use
statement, so the single-student body can be checked independently against the
two-page limit. The AI-use statement and references follow outside that limit;
detailed supporting evidence remains in `docs/` rather than an appendix.

Before submission:

1. confirm the author name, student number and institutional email in `main.tex`;
2. read and edit the AI-use statement so it exactly describes the final
   workflow;
3. compile and confirm that the scientific body occupies no more than two
   pages without changing margins, spacing, or the course style;
4. verify that every result still matches the frozen evidence in
   [`docs/results.md`](../docs/results.md); and
5. submit the PDF, repository link, and AI statement using the addresses and
   subject specified in the current course guidelines.

Build from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Clean auxiliary files with:

```bash
latexmk -c
```

The files `dlaiml2026.sty`, `dlaiml2026.bst`, and `fancyhdr.sty` are unmodified
copies from the professor's official `template.zip`. Do not edit them to fit
more material. The Markdown documents under `docs/` remain the detailed source
of truth and reproducibility record; this directory is the concise submission
artifact. The primary 2,086-request results, strategy-repetition ranges,
correctness details and uncertainty notes are consolidated in
[`docs/results.md`](../docs/results.md). Arena/Triton tables, manifests and
figures are retained in the committed
[`docs/generated/arena_triton`](../docs/generated/arena_triton/README.md)
directory. Raw result archives remain excluded from Git.
