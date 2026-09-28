# Verification report

Generated on 2026-09-29 from the local working tree. This report records the requested short reproducibility verification; it does not claim that the long experiments were rerun.

## Executed successfully

```text
python scripts/check_release.py                         PASS
python scripts/check_english_docs.py                    PASS
python scripts/build_output_manifest.py                 PASS (1,894 reference artifacts)
python -m compileall -q src experiments tools generation   PASS
python scripts/reproduce_tables.py                      PASS
python scripts/reproduce_figures.py                     PASS
python scripts/verify_release.py                       PASS
python tools/verify_outputs.py --output-root /does/not/exist  PASS (reference_not_bundled mode)
```

The summary scripts regenerated two CSV summaries and four PNG/PDF release-level summary figures from the 14 mixed-scenario and 105 pairwise derived CSV files. Repeating the commands produced identical summary-table SHA-256 values. The independent composition self-test also passed against the local canonical caches for `S1-A` and `P001` in both unique and non-unique modes. The artifact manifest now maps 1,884 generated/reference-derived files to generators and identifies 10 editorial or imported assets separately.

## Not executed in this draft

The full E1-E4 and E20-E24 simulations were intentionally not run, per the requested verification scope. They require the official raw archives, canonical caches, trained models and multi-gigabyte intermediate result directories. The staged runner contains the ordered commands for a later full run, but this release is verified here at the release-data and artifact-contract level.

The current local interpreter was Python 3.9.12 while the release declares Python 3.10 or newer. The release therefore requires a Python 3.10+ environment for an authoritative rerun.

## Interpretation

The package is verified as reproducible for the included derived-data summaries, deterministic composition logic, source-code contract, and final-artifact mapping. Full numerical equivalence of every E1-E4/E20-E24 output would require the separate long run described in the README. Manual or imported editorial files are labelled `metadata_or_manual` in the manifest rather than being falsely attributed to a generator.
