# Course report

`main.tex` is the project paper in the official DLAI 2025/2026 template. Its
scientific body ends at the explicit `\clearpage` before the mandatory AI-use
statement, so the single-student body can be checked independently against the
two-page limit. The AI-use statement and references follow outside that limit;
detailed supporting evidence remains in `docs/` rather than an appendix.

Before submission:

1. replace `[Student name]` and `[institutional email]` in `main.tex`;
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
artifact. The primary 2,086-request evidence is frozen in
[`docs/generated/full_dev_confirmation`](../docs/generated/full_dev_confirmation/README.md).
That directory includes the original CSV/JSON analysis, figures, all 12 run
manifests, and a `SHA256SUMS` integrity file. Its results belong to measured
revision `89acf36a603adf7dacd5c8dd0b32a2967f4bbee2` (schema v3); the current
schema-v4 source contains later fixes and was not used for those measurements.

The separate [strategy repetitions](../docs/generated/strategy_repetitions/README.md)
are complete: three 1,000-request runs per organization/workload, measured at
revision `68e7080440d38a3b47610ed398d6d00ed098d066` with schema v4. The body
summarizes their result; per-run mean/p90 values and ranges remain in the linked
evidence. Keep these shorter timing checks distinct from the full-dev accuracy
tables.

Repetition tables, correctness details, uncertainty notes and the Triton
startup calculation remain under `docs/` and `docs/generated/`. The paper
links those files directly so the compact submission can be verified without
duplicating them.
