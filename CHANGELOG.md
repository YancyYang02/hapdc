# Changelog

## 0.6.8

- `prep` records the sample identifier mode and passes it to phenotype and
  covariate alignment. Legacy/API automatic alignment rejects competing IID
  and FID_IID mappings instead of silently choosing one from a partial match.
- `merge` validates all input tables and summaries before writing outputs,
  rejects empty inputs and P values outside [0, 1], and stages both outputs
  before replacing existing files. Failed validation or staging leaves prior
  outputs intact. The two final replacements are not a crash-atomic transaction.
- Statistical components with no residual degrees of freedom are marked untestable using
  the genotype/design rank. Valid single-marker components remain available
  when a saturated haplotype-additive component is excluded.
- The direct regression-test entry point runs all 33 test definitions.
  Input-validation tests are collected separately by pytest.
- Covariates are centred and scaled before the rank check, the REML fit and the
  fixed-effect projection. Adding a constant to a covariate or changing its
  units no longer changes the rank decision, the REML estimate or any p-value.
  The REML profile and the projection use an orthogonal (QR) decomposition of
  the standardised design. A column is redundant when its residual norm, on
  unit-norm columns, is at most max(1e-8, 64 eps max|x| / sd). The mean and
  standard deviation used for each covariate are recorded in `summary.json`
  (`covariate_transform`). Results without covariates are unchanged.
- Sample identifiers must be unique in the VCF header, the GRM `.grm.id`
  (FID, IID pairs), the phenotype table and the covariate table. A repeated
  identifier stops the run with exit code 1 and names it. A GRM that uses one
  IID under several FIDs is matched on `FID_IID` when the VCF names samples
  that way, and is otherwise refused. Phenotype and covariate tables accept a
  `FID IID` header. The `.grm.bin` size is checked against the `.grm.id`.
  `io.read_grm` without `return_fid=True` refuses an IID that occurs twice.
- `ms` writes P and `P_nonadditive` with 17 significant digits, so `merge`
  reproduces the per-chromosome significance decision. `merge` warns about
  tables with P at 4 significant digits and lists them under
  `low_precision_tables`.
- `merge` derives the summary path from the real suffix of the output name
  (`NAME.tsv` gives `NAME.summary.json`) and refuses to overwrite an input. It
  finds each input table's own summary with the same rule.
- A trait with no variation, or with no residual degrees of freedom after the
  fixed effects, and any non-finite anchor statistic stop `ms` and `scan` with
  exit code 2. `ms` then writes `summary.json` with `"status": "FAILED"`, a
  `reason` and `n_sig` null, and no anchor table; a completed run records
  `"status": "completed"`. `merge` refuses FAILED shards.
- `report` reads the default LD-unit scan output (`scan_ld.npz`) as well as
  fixed-width windows, and its help states that it does not read `ms` tables.
- `--chrom` treats `1` and `chr1` as the same chromosome and stops when a
  requested chromosome is absent.
- An unphased heterozygous genotype (`0/1`) in the VCF stops `prep` with exit
  code 1 and names the marker and sample. `--skip-unphased` drops such markers
  instead and records the count.
