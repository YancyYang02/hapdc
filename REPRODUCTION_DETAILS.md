# Reproduction details

This note documents implementation details that matter for an exact numerical
reproduction of hapdc results.

## Sample identity

`prep/meta.json` records `sample_id_mode`: `iid` for ordinary VCF sample names,
or `fid_iid` when repeated GRM IIDs require FID_IID composite names. Phenotype
tables and subsequent `ms --covar` alignment use that recorded mode. A table
whose first two headers are FID and IID uses the appropriate key; a one-key
table uses its sample strings directly. Legacy prep files without the field
use `auto`, which raises an identity error if IID and composite keys map the
requested samples to competing rows. API callers can supply `id_mode`.

## Interpreting an association result

An anchor association is evidence against the fitted null model. The combined
statistic includes a single-marker component, so an associated anchor alone
does not establish an effect unique to a multi-marker haplotype or diplotype.
Check the leading scale and unit, not just the anchor P value.

The non-additivity statistics test departures from their specified additive
designs. A recessive or dominant single-SNP effect can produce such a departure.
To investigate additional diplotype structure, report additive, dominant and
recessive SNP comparisons, retained haplotypes, diplotype-cell masses and any
small-cell reassignment. Opposite additive effects of one SNP on two different
traits are a separate question from non-additivity within one trait.

Choose frequency/cell thresholds before a confirmatory scan. Raising `minc`
can merge cells and change the model being tested; it is not merely a change
to the reporting threshold. Checks selected after viewing a signal describe
sensitivity and do not provide a new selection-adjusted P value. A small
non-additivity P that depends on a small cell should not be presented as a
robust diplotype-specific finding without further evidence.

Independent replication requires an aligned variant or haplotype definition,
effect direction and compatible phenotype. Combining unsigned anchor P values
does not itself provide those checks. Real-data detection differences also do
not estimate statistical power in the absence of a known truth or a suitable
validation design.

## Genotype partition and windows

LD statistics and frequencies are computed in the analysed sample set. Initial
blocks are cut at an adjacent LD threshold failure, a marker gap larger than
`ld_maxspan`, or a chromosome change. A following initial block shorter than
two markers merges into its preceding block, except across a hard gap or
chromosome boundary. A leading singleton is not specially merged forward.
This merging happens **before** the haplotype-coverage split.

Common haplotype coverage is the sum of the integer counts of haplotypes with
frequency at least `minf`, divided by the total number of copies. A block with
insufficient coverage is recursively split at the weakest internal D-prime
link, with the leftmost link chosen on a tie. Pieces are not merged again
after this split. Consecutive blocks form anchors with at least `anchor_min`
markers; a final short remainder joins the previous anchor in its chromosome.

Within each anchor, W2/W4/W8/W16 windows advance by half their width. If the
last regular window stops before the anchor's right edge, an additional window
ends exactly at that edge. LD units and the whole anchor are also included.
The single-marker scale uses additive SNP tests. This precise partition order
and the right-edge windows matter for reproduction.

## Design, rank and null model

Rare haplotypes map to a nearest retained haplotype by Hamming distance, then
frequency; exact ties share mass equally. If no haplotypes pass `minf`, the
haplotype design is empty; testable single-marker components can remain.
Fractional diplotype-cell memberships use the product of the two copy
distributions. Cell masses use an absolute 1e-9 tolerance at the `minc`
boundary and when comparing equally frequent target cells.

The fixed-effect span contains an intercept and retained, centred/scaled
covariates. The GRM is subset to analysed samples, decomposed in float64, and
its eigenvalues floored at 1e-6. REML estimates the variance ratio within
log(delta) bounds [-10, 10]. Whitened designs use a pivoted QR after scaling
nonzero columns to unit norm, with a relative 1e-8 rank criterion. Nested F
tests require positive numerator and residual degrees of freedom; an
undefined component is excluded using genotype/design validity, not its P.

An independent reimplementation with its own REML search and F distribution
code should not be expected to agree bitwise.

## Cauchy combination and numeric endpoints

The D unit statistic combines defined A_joint, CELL_F, SACAT and NADD_F with
equal component weights. Component P values entering that unit combination
are clipped to [1e-300, 1-1e-12]. SACAT has the same upper clipping for its
marker components. These are implementation choices that must be explicit in
a numerical reproduction.

At the anchor level, each present scale has total weight one, divided equally
over its testable units. Before combination, unit P >= 1-1e-12 is replaced by
1-1/max(d,2), where d is the number of testable units across the anchor. This
endpoint regularization avoids an infinite negative Cauchy contribution. It
is part of the implemented procedure; claims about calibration must refer to
the procedure including this rule and the relevant null simulations.

For the Cauchy transform, use 1/(pi*p) for p < 1e-10. Direct evaluation of
tan(pi*(0.5-p)) at very small p loses precision when subtracting p from 0.5.
The combined upper-tail P uses 1/(pi*T) when T > 1e7; otherwise the production
implementation uses 0.5-atan(T)/pi. An independent arctan(1/T) expression can
retain more digits near the positive tail while representing the same
mathematical combination.

## Merge output

Merge input validation completes before any output mutation, and both new files
are fully staged before renaming. The pair of renames is not a single
filesystem transaction across a process crash.

## Diplotype support

### The cell floor does not bound support for every joint contrast

`minc` applies to retained cell columns in the window design. The non-additivity
alternative additionally uses haplotype dosage and original SNP additive columns.
Their joint column space can still distinguish genotype patterns supported by
fewer than `minc` animals, even after small diplotype cells have been merged.
A small NADD P value therefore does not by itself establish a well-supported
full diplotype effect. Inspect original haplotype/genotype counts, rare-haplotype
assignment, nested-model ranks, and comparisons to additive, dominant and
recessive SNP models. Changing `minc` changes a model; a post-hoc sensitivity
check must not replace a prespecified primary analysis.
