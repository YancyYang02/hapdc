import shutil
import glob
"""Regression tests for hapdc. Run: python -m pytest tests  (or python tests/test_regressions.py)."""
import gzip
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from hapdc import core, io, ms, scan  # noqa: E402

N = 120
PATS = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1]], np.int8)
PAIRS = {0: (0, 1), 1: (2, 2), 2: (2, 3), 3: (3, 3)}
META = [("1", "m%d" % j, j + 1, "A", "C") for j in range(4)]


def _grm(seed=0, n=N):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n // 4))
    K = X @ X.T / X.shape[1]
    S, U = np.linalg.eigh(K)
    return K, U, np.clip(S, 1e-6, None)


def _data(seed, p_ab=0.6):
    """Haplotypes A and B are always carried together, so dos_A == dos_B: an
    early linear dependency among the haplotype-dosage columns."""
    rng = np.random.default_rng(seed)
    typ = rng.choice(4, N, p=[p_ab, (1 - p_ab) * .4, (1 - p_ab) * .4, (1 - p_ab) * .2])
    H = np.zeros((2 * N, 4), np.int8)
    for i, t in enumerate(typ):
        a, b = PAIRS[t]
        H[2 * i], H[2 * i + 1] = PATS[a], PATS[b]
    y = 0.5 * (typ == 2) + 0.4 * (typ == 1) + rng.normal(size=N)
    return H, y, rng


def _ref(H, U, S, y):
    """Nested-model F tests from an SVD column basis (reference)."""
    sc = scan.Scan(H, META, U, S, y)
    d = core.window_design(H, 0.05, 5)
    Xd = (U.T @ d["dos"]) * sc.w[:, None]
    Xc = (U.T @ d["C"]) * sc.w[:, None]
    ys = sc.yr * sc.w

    def rss(X):
        u, s, _ = np.linalg.svd(X, full_matrices=False)
        Q = u[:, s > 1e-9 * s[0]]
        r = ys - Q @ (Q.T @ ys)
        return r @ r, Q.shape[1]

    one = sc.one[:, None]
    r0, k0 = rss(one)
    r1, k1 = rss(np.column_stack([one, Xd]))
    rc, kc = rss(np.column_stack([one, Xc]))
    rf, kf = rss(np.column_stack([one, Xd, Xc]))
    f = lambda a, b, ka, kb: stats.f.sf(((a - b) / (kb - ka)) / (b / (N - kb)), kb - ka, N - kb)
    return dict(A_joint=f(r0, r1, k0, k1), CELL_F=f(r0, rc, k0, kc), NADD_F=f(r1, rf, k1, kf))


def _p(H, U, S, y):
    r = scan.Scan(H, META, U, S, y).run(blocks=[(0, 4)], verbose=False)
    return {k: float(r["P"][k][0, 0]) for k in ("A_joint", "CELL_F", "NADD_F")}


def test_f_tests_match_reference_and_sample_order():
    K, U, S = _grm()
    for seed in range(20):
        H, y, rng = _data(seed, 0.6 if seed % 2 else 0.3)
        perm = rng.permutation(N)
        hp = np.concatenate([[2 * i, 2 * i + 1] for i in perm])
        Sp, Up = np.linalg.eigh(K[np.ix_(perm, perm)])
        a = _p(H, U, S, y)
        b = _p(H[hp], Up, np.clip(Sp, 1e-6, None), y[perm])
        r = _ref(H, U, S, y)
        for k in a:
            assert abs(np.log10(a[k]) - np.log10(r[k])) < 1e-6, (seed, k, a[k], r[k])
            assert abs(np.log10(b[k]) - np.log10(r[k])) < 1e-6, (seed, k, b[k], r[k])


def test_permutation_null_is_calibrated():
    """Replicates from permuted whitened residuals: p-values ~ U(0,1) under H0."""
    K, U, S = _grm(1, 200)
    rng = np.random.default_rng(5)
    L = np.linalg.cholesky(K + 1e-6 * np.eye(200))
    H = rng.integers(0, 2, size=(400, 1)).astype(np.int8)
    y = 2.0 * (L @ rng.normal(size=200)) + rng.normal(size=200)
    sc = scan.Scan(H, [("1", "m", 1, "A", "C")], U, S, y).permute(2000)
    YS = sc._design_matrix(None, 0)[:, 1:]
    g = sc._p0(((U.T @ (H[0::2, 0] + H[1::2, 0]).astype(float)) * sc.w)[:, None])[:, 0]
    r0 = sc._p0(YS)
    b = g @ r0 / (g @ g)
    t = b / np.sqrt(((r0 * r0).sum(0) - b * b * (g @ g)) / (200 - 2) / (g @ g))
    p = 2 * stats.t.sf(np.abs(t), 198)
    assert stats.kstest(p, "uniform").pvalue > 1e-3
    assert abs((p < 0.05).mean() - 0.05) < 0.02


def test_wacat_is_standard_acat_except_p_one():
    p, _, _ = ms.wacat([np.array([0.8]), np.array([0.9])], ["SNP", "SNP"])
    t = (np.tan((0.5 - 0.8) * np.pi) + np.tan((0.5 - 0.9) * np.pi)) / 2
    assert abs(p[0] - (0.5 - np.arctan(t) / np.pi)) < 1e-12
    p1, _, _ = ms.wacat([np.array([1.0]), np.array([1e-6])], ["SNP", "SNP"])
    assert p1[0] < 1e-5


def _write_vcf(path, rows, ids):
    with gzip.open(path, "wt") as f:
        f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(ids) + "\n")
        for chrom, pos, ref, alt, gts in rows:
            f.write("%s\t%d\t.\t%s\t%s\t.\tPASS\t.\tGT\t%s\n" % (chrom, pos, ref, alt, "\t".join(gts)))


def test_vcf_reader_rejects_multiallelic_unphased_and_counts_nmax():
    d = tempfile.mkdtemp()
    ids = ["s1", "s2", "s3"]
    rows = [("1", 10, "A", "C,G", ["1|2", "0|0", "0|1"]),   # multi-allelic ALT
            ("1", 20, "A", "C", ["0/1", "0|0", "1|1"]),     # unphased heterozygote
            ("1", 30, "A", "C", ["0|2", "0|0", "1|1"]),     # allele index 2 with one ALT
            ("1", 40, "A", "C", ["1", "0|0", "1|1"]),       # haploid call
            ("1", 50, "A", "C", ["1|0", "0|0", "1|1"]),
            ("1", 60, "A", "C", ["1/1", "0|1", "0|0"])]     # unphased homozygote: fine
    p = os.path.join(d, "x.vcf.gz")
    _write_vcf(p, rows, ids)
    try:
        io.read_vcf(p)
        raise AssertionError("unphased heterozygote accepted by default")
    except io.UnphasedError as e:
        assert "1:20" in str(e) and "s1" in str(e) and "phas" in str(e)
    _, H, meta, dr = io.read_vcf(p, unphased="skip")
    assert [m[2] for m in meta] == [50, 60]
    assert H[:, 0].tolist() == [1, 0, 0, 0, 1, 1]
    assert dr["multiallelic"] == 2 and dr["unphased"] == 1 and dr["nondiploid"] == 1
    rows = [("1", 100 + j, "A", "C", ["1|0", "0|0", "0|1"]) for j in range(5000)]
    _write_vcf(p, rows, ids)
    _, H, meta, _ = io.read_vcf(p, nmax=4500)
    assert H.shape == (6, 4500)


def _toy_prep(d, n=20, m=60, missing_all=("i7",), missing_t2=("i3",)):
    rng = np.random.default_rng(3)
    ids = ["i%d" % i for i in range(n)]
    F = rng.integers(0, 2, size=(4, m))
    rows = []
    hap = F[rng.integers(0, 4, size=2 * n)]
    for j in range(m):
        if hap[:, j].min() == hap[:, j].max():
            hap[0, j] = 1 - hap[0, j]
        rows.append(("1", 1000 * (j + 1), "A", "C", ["%d|%d" % (hap[2 * i, j], hap[2 * i + 1, j]) for i in range(n)]))
    _write_vcf(os.path.join(d, "g.vcf.gz"), rows, ids)
    with open(os.path.join(d, "ph.txt"), "w") as f:
        f.write("IID\tt1\tt2\n")
        for i in ids:
            v1 = "-9" if i in missing_all else "%.3f" % rng.normal()
            v2 = "-9" if (i in missing_all or i in missing_t2) else "%.3f" % rng.normal()
            f.write("%s\t%s\t%s\n" % (i, v1, v2))
    return ids


def _cli(*args):
    env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-c", "import sys; from hapdc.cli import main; sys.exit(main(sys.argv[1:]))"]
                       + list(args), capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_prep_multitrait_and_scan_after_missing():
    d = tempfile.mkdtemp()
    _toy_prep(d)
    _cli("prep", os.path.join(d, "g.vcf.gz"), "auto", os.path.join(d, "ph.txt"), os.path.join(d, "prep"), "--grm-thin", "1")
    mj = json.load(open(os.path.join(d, "prep", "meta.json")))
    assert mj["traits"] == ["t1", "t2"] and mj["n"] == 19
    Y = np.load(os.path.join(d, "prep", "traits.npy"))
    assert Y.shape == (19, 2) and np.isnan(Y[:, 1]).sum() == 1
    for t in ("0", "1"):
        _cli("scan", os.path.join(d, "prep"), os.path.join(d, "scan" + t), t, "--units", "fixed",
             "--wsize", "4", "--nperm", "0", "--minc", "1", "--minf", "0.05", "--quiet")


def test_merge_genome_wide_threshold():
    d = tempfile.mkdtemp()
    hdr = "chrom\tstart_bp\tend_bp\tn_markers\tP\tlead_scale\tlead_unit_start\tlead_unit_end\tP_nonadditive\tsig_bonf\n"
    for c, ps in (("1", [1e-3, 0.5]), ("2", [0.02, 0.2])):  # 0.02: significant per chromosome (0.025), not genome-wide (0.0125)
        with open(os.path.join(d, "c%s.tsv" % c), "w") as f:
            f.write(hdr)
            for k, p in enumerate(ps):
                f.write("%s\t%d\t%d\t20\t%g\tSNP\t1\t2\t1\t%d\n" % (c, 10 * k, 10 * k + 5, p, int(p < 0.05 / 2)))
    _cli("merge", os.path.join(d, "all"), os.path.join(d, "c1.tsv"), os.path.join(d, "c2.tsv"))
    js = json.load(open(os.path.join(d, "all.summary.json")))
    assert js["n_anchors"] == 4 and abs(js["threshold_bonf"] - 0.0125) < 1e-12 and js["n_sig"] == 1


def test_projection_cache_is_bounded():
    rng = np.random.default_rng(0)
    n, m = 50, 2048 * 20
    H = rng.integers(0, 2, size=(2 * n, m)).astype(np.int8)
    K, U, S = _grm(2, n)
    sc = scan.Scan(H, [("1", "m%d" % j, j, "A", "C") for j in range(m)], U, S, rng.normal(size=n))
    for s in range(0, m - 100, 1500):
        sc._projG(s, s + 100)
    assert len(sc._UG) <= sc.cache_blocks
    X = sc._projG(5000, 5100)
    assert np.allclose(X, U.T @ (H[0::2, 5000:5100] + H[1::2, 5000:5100]).astype(float))


# ---------------------------------------------------------------- audit items

def _founders(seed, n=300, nblk=6, blk=6):
    rng = np.random.default_rng(seed)
    m = nblk * blk
    H = np.zeros((2 * n, m), np.uint8)
    for b in range(nblk):
        F = rng.integers(0, 2, (4, blk)).astype(np.uint8)
        H[:, b * blk:(b + 1) * blk] = F[rng.choice(4, 2 * n, p=rng.dirichlet(np.ones(4) * 2))]
    H = (H ^ (rng.random(H.shape) < 0.01)).astype(np.uint8)
    pos = np.cumsum(rng.integers(100, 3000, m))
    return H, pos, rng


def _proj(X):
    from scipy.linalg import orth
    Q = orth(np.column_stack([np.ones(X.shape[0]), X]))
    return Q @ Q.T


def test_window_design_is_invariant_to_allele_coding_and_sample_order():
    """S02: rare-haplotype and small-cell ties are shared, not broken by byte order."""
    # auditor's example: two equally frequent retained haplotypes, rare 01 equidistant
    H = np.repeat(np.array([[0, 0], [1, 1], [0, 1]], np.uint8), [98, 98, 4], axis=0)
    Hf = H.copy(); Hf[:, 0] = 1 - Hf[:, 0]
    a, b = core.window_design(H), core.window_design(Hf)
    assert np.allclose(_proj(a["dos"]), _proj(b["dos"]))
    assert np.allclose(_proj(a["C"]), _proj(b["C"]))
    rng = np.random.default_rng(1)
    for _ in range(150):
        nf = rng.integers(2, 6)
        F = rng.integers(0, 2, (nf, rng.integers(2, 8))).astype(np.uint8)
        n = int(rng.integers(30, 200))
        Hh = F[rng.choice(nf, 2 * n, p=rng.dirichlet(np.ones(nf)))]
        Hh = (Hh ^ (rng.random(Hh.shape) < 0.01)).astype(np.uint8)
        fl = rng.random(Hh.shape[1]) < .5
        perm = rng.permutation(n)
        Hp = Hh.copy(); Hp[:, fl] = 1 - Hp[:, fl]
        Hp = Hp.reshape(n, 2, -1)[perm][:, ::-1].reshape(2 * n, -1)
        a, b = core.window_design(Hh), core.window_design(Hp)
        # each individual sits in exactly one class pair and one cell in total
        assert np.allclose(a["dos"].sum(1), 2) and (a["ncell"] == 0 or np.allclose(a["C"].sum(1), 1))
        assert np.allclose(_proj(a["dos"])[np.ix_(perm, perm)], _proj(b["dos"]))
        assert np.allclose(_proj(a["C"])[np.ix_(perm, perm)], _proj(b["C"]))


def test_ld_blocks_are_invariant_to_allele_coding_and_sample_order():
    """S01/S12: |D'| and coverage from integer counts; minf reaches the splitter."""
    for seed in range(40):
        H, pos, rng = _founders(seed)
        n = H.shape[0] // 2
        fl = rng.random(H.shape[1]) < .5
        perm = rng.permutation(n)
        Hp = H.copy(); Hp[:, fl] = 1 - Hp[:, fl]
        Hp = Hp.reshape(n, 2, -1)[perm].reshape(2 * n, -1)
        for kw in (dict(), dict(hap_cover=0.9, minf=0.1)):
            a = core.ld_blocks(H, pos, dmin=0.8, max_span=200000, min_mk=2, **kw)["blocks"]
            b = core.ld_blocks(Hp, pos, dmin=0.8, max_span=200000, min_mk=2, **kw)["blocks"]
            assert [tuple(x) for x in a] == [tuple(x) for x in b]
    # minf changes the splitting (it used to be ignored by the ms worker)
    src = open(os.path.join(os.path.dirname(HERE), "hapdc", "cli.py"), encoding="utf-8").read()
    assert "hap_cover=a.ld_hapcover, minf=a.minf" in src


class _Batch(scan.Scan):
    def __init__(self, H, meta, U, S, Y):
        super().__init__(H, meta, U, S, Y[:, 0])
        self._Y = (U.T @ Y) * self.w[:, None]

    def _design_matrix(self, rows, B):
        return self._Y


def test_ms_scan_does_not_depend_on_the_batch():
    """S04: components are selected by genotype-only validity, not by batch p-values."""
    H, pos, rng = _founders(3)
    n = H.shape[0] // 2
    H[:, 5] = 0                                    # a monomorphic marker inside a block
    meta = [("1", "m%d" % j, int(pos[j]), "A", "C") for j in range(H.shape[1])]
    K, U, S = _grm(4, n)
    bl = core.ld_blocks(H, pos, dmin=0.8, max_span=200000, min_mk=2)["blocks"]
    anc = ms.anchors([(int(s), int(e)) for s, e in bl], 16)
    y = rng.normal(size=n)
    alone = ms.ms_scan(_Batch(H, meta, U, S, y[:, None]), anc, variants=("D", "O"), nadd=True, nadds=True)
    Y = np.column_stack([y, rng.normal(size=(n, 7)), np.where(np.arange(n) % 2, 1.0, 0.0)])
    batch = ms.ms_scan(_Batch(H, meta, U, S, Y), anc, variants=("D", "O"), nadd=True, nadds=True)
    for v in ("D", "O"):
        assert np.allclose(np.log(alone["P_MS"][v][:, 0]), np.log(batch["P_MS"][v][:, 0]), atol=1e-9)
    assert np.allclose(alone["P_NADD"][:, 0], batch["P_NADD"][:, 0], atol=1e-12)


def test_ms_scan_is_invariant_to_allele_coding():
    H, pos, rng = _founders(5)
    n = H.shape[0] // 2
    meta = [("1", "m%d" % j, int(pos[j]), "A", "C") for j in range(H.shape[1])]
    K, U, S = _grm(6, n)
    y = rng.normal(size=n)
    out = []
    for Hh in (H, (1 - H).astype(np.uint8)):
        bl = core.ld_blocks(Hh, pos, dmin=0.8, max_span=200000, min_mk=2)["blocks"]
        anc = ms.anchors([(int(s), int(e)) for s, e in bl], 16)
        out.append(ms.ms_scan(scan.Scan(Hh, meta, U, S, y), anc, variants=("D",))["P_MS"]["D"][:, 0])
    assert np.allclose(np.log(out[0]), np.log(out[1]), atol=1e-8)


def test_lmm_scan_additive_uses_rotated_intercept():
    """S14: the intercept is U'1 on the eigen scale, not the weights alone."""
    rng = np.random.default_rng(2)
    n = 150
    K, U, S = _grm(2, n)
    G = rng.integers(0, 3, (n, 4)).astype(float)
    y = 5.0 + rng.normal(size=n)
    t, df = core.lmm_scan_additive(U.T @ G, (U.T @ y)[:, None], S, 0.7, U.T @ np.ones(n))
    L = np.linalg.cholesky(K + 0.7 * np.eye(n) + 1e-9 * np.eye(n))
    Li = np.linalg.inv(L)
    for j in range(4):
        X = Li @ np.column_stack([np.ones(n), G[:, j]])
        b, *_ = np.linalg.lstsq(X, Li @ y, rcond=None)
        r = Li @ y - X @ b
        se = np.sqrt(r @ r / (n - 2) * np.linalg.inv(X.T @ X)[1, 1])
        assert abs(t[j, 0] - b[1] / se) < 1e-6


def test_ms_scan_batch_invariance_at_p_equal_one():
    """S04, auditor's construction: a valid A_joint whose observed p rounds to 1."""
    from scipy import linalg
    rng = np.random.default_rng(704); n = 160
    pats = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.uint8)
    H = pats[rng.integers(0, 4, 2 * n)]
    d = core.window_design(H)
    Qa = linalg.orth(np.column_stack([np.ones(n), d["dos"]]))
    y = d["C"][:, 1] + .08 * rng.normal(size=n)
    y = y - Qa @ (Qa.T @ y)
    other = d["dos"][:, 0] + rng.normal(size=n)
    meta = [("1", str(j), j, "A", "G") for j in range(2)]
    U, S = np.eye(n), np.ones(n)
    anc = [(0, 2, [(0, 2)])]
    pa = ms.ms_scan(_Batch(H, meta, U, S, y[:, None]), anc, variants=("D",), nadds=False)["P_MS"]["D"][0, 0]
    pb = ms.ms_scan(_Batch(H, meta, U, S, np.column_stack([y, other])), anc, variants=("D",), nadds=False)["P_MS"]["D"][0, 0]
    assert abs(np.log(pa) - np.log(pb)) < 1e-9, (pa, pb)


# ------------------- testability reference, covariate rank, provenance keys
def _v01_counterexample():
    """Reviewer's construction (independent_checks, seed 20261121): n = 120, one
    marker with MAF 0.129, covariate g + 5.416e-5 * z, nearly collinear with the
    marker dosage. Before the fix the testability reference norm used the whitening
    vector w instead of the whitened intercept (U'1) * w, and the marker was
    testable under one allele coding and not under the other."""
    rng = np.random.default_rng(20261121)
    n = 120
    H = rng.binomial(1, .15, (2 * n, 1)).astype(np.int8)
    g = (H[0::2, 0] + H[1::2, 0]).astype(float)
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    S = np.linspace(.3, 2, n)
    z = rng.normal(size=n)
    y = z + .2 * rng.normal(size=n)
    eps = None
    for e in np.geomspace(1e-6, .001, 200):          # same grid as the review
        if abs(e - 5.416e-5) / 5.416e-5 < 1e-3:
            eps = e
            break
    assert eps is not None
    C = (g + eps * z)[:, None]
    return H, g, U, S, y, C, z


def test_v01_testable_is_invariant_to_allele_coding_with_collinear_covariate():
    H, g, U, S, y, C, _ = _v01_counterexample()
    assert abs(g.mean() / 2 - 0.129) < 5e-4, g.mean() / 2
    meta = [("1", "m1", 100, "A", "C")]
    res = []
    for HH in (H, 1 - H):
        sc = scan.Scan(HH, meta, U, S, y, covar=C)
        p, v = ms._snp_p(sc, np.array([0]))
        pd = ms.ms_scan(sc, [(0, 1, [(0, 1)])], variants=("D",))["P_MS"]["D"][0, 0]
        res.append((bool(v[0]), p[0, 0], pd))
    assert res[0][0] == res[1][0], res
    assert abs(np.log(res[0][1]) - np.log(res[1][1])) < 1e-6, res
    assert abs(np.log(res[0][2]) - np.log(res[1][2])) < 1e-6, res


def test_v01_testable_reference_is_whitened_intercept():
    H, g, U, S, y, C, _ = _v01_counterexample()
    sc = scan.Scan(H, [("1", "m1", 100, "A", "C")], U, S, y, covar=C)
    o = (U.T @ np.ones(len(y))) * sc.w
    assert np.allclose(sc.o, o)
    # Q0 is always the basis used by _p0, also with intercept only
    sc1 = scan.Scan(H, [("1", "m1", 100, "A", "C")], U, S, y)
    X = np.random.default_rng(3).normal(size=(len(y), 3))
    o1 = sc1.o / np.linalg.norm(sc1.o)
    assert sc1.nfix == 1 and np.allclose(sc1.Q0[:, 0], o1)
    assert np.allclose(sc1._p0(X), X - np.outer(o1, o1 @ X))


def test_v01_eigenvector_sign_flip_and_allele_flip():
    """Flipping the sign of GRM eigenvectors, alone or together with recoding
    the alleles, changes neither testability nor the p-values."""
    H, g, U, S, y, C, _ = _v01_counterexample()
    meta = [("1", "m1", 100, "A", "C")]
    Uf = U.copy()
    Uf[:, ::3] *= -1
    base = None
    for UU in (U, Uf):
        for HH in (H, 1 - H):
            for CC in (None, C):
                sc = scan.Scan(HH, meta, UU, S, y, covar=CC)
                p, v = ms._snp_p(sc, np.array([0]))
                key = CC is None
                if base is None:
                    base = {}
                if key not in base:
                    base[key] = (bool(v[0]), p[0, 0], sc.delta)
                b = base[key]
                assert bool(v[0]) == b[0]
                assert abs(np.log(p[0, 0]) - np.log(b[1])) < 1e-6, (p[0, 0], b[1])
                assert abs(sc.delta - b[2]) < 1e-6 * max(1, b[2])
    # and the multi-marker window statistics
    Hm, ym, _ = _data(11)
    K, U2, S2 = _grm(5)
    U2f = U2.copy()
    U2f[:, 1::2] *= -1
    cov = np.random.default_rng(9).normal(size=(N, 1))
    ra = scan.Scan(Hm, META, U2, S2, ym, covar=cov).run(blocks=[(0, 4)], verbose=False)
    rb = scan.Scan(1 - Hm, META, U2f, S2, ym, covar=cov).run(blocks=[(0, 4)], verbose=False)
    for q in ("A_joint", "CELL_F", "NADD_F", "SACAT"):
        a_, b_ = float(ra["P"][q][0, 0]), float(rb["P"][q][0, 0])
        assert np.isfinite(a_) == np.isfinite(b_), q
        if np.isfinite(a_):
            assert abs(np.log(a_) - np.log(b_)) < 1e-6, (q, a_, b_)


def test_v02_covariate_rank_check_before_reml():
    H, g, U, S, y, C, z = _v01_counterexample()
    n = len(y)
    meta = [("1", "m1", 100, "A", "C")]
    ref = scan.Scan(H, meta, U, S, y, covar=z[:, None])
    p_ref = ms._snp_p(ref, np.array([0]))[0][0, 0]
    p_int = ms._snp_p(scan.Scan(H, meta, U, S, y), np.array([0]))[0][0, 0]
    cases = {
        "constant": (np.ones((n, 1)), ["covar1"], p_int, 1),
        "constant_5": (np.full((n, 1), 5.0), ["covar1"], p_int, 1),
        "duplicated": (np.column_stack([z, z]), ["covar2"], p_ref, 2),
        "scaled_duplicate": (np.column_stack([z, 3 * z + 1]), ["covar2"], p_ref, 2),
        "constant_then_valid": (np.column_stack([np.ones(n), z]), ["covar1"], p_ref, 2),
    }
    for lab, (CC, drop, pexp, nfix) in cases.items():
        sc = scan.Scan(H, meta, U, S, y, covar=CC)
        assert [d_["name"] for d_ in sc.covar_dropped] == drop, (lab, sc.covar_dropped)
        assert sc.nfix == nfix, lab
        p = ms._snp_p(sc, np.array([0]))[0][0, 0]
        assert abs(np.log(p) - np.log(pexp)) < 1e-8, (lab, p, pexp)
        try:
            scan.Scan(H, meta, U, S, y, covar=CC, covar_policy="error")
            raise AssertionError("no error for " + lab)
        except core.FixedEffectRankError as e:
            assert drop[0] in str(e)
    assert core.check_fixed_effects(np.ones((n, 1)), n)[2][0]["reason"] == "constant"
    assert core.check_fixed_effects(np.column_stack([z, z]), n, names=["pc1", "pc1b"])[2][0]["reason"] == "duplicate of pc1"
    # a nearly collinear but full-rank covariate (the V01 case) is kept
    assert core.check_fixed_effects(C, n)[2] == []
    # REML itself refuses a singular design instead of raising LinAlgError
    try:
        core.reml_delta(U.T @ y, S, U.T @ np.column_stack([np.ones(n), np.ones(n)]))
        raise AssertionError("reml_delta accepted a singular design")
    except core.FixedEffectRankError:
        pass
    try:
        core.check_fixed_effects(np.column_stack([z, np.r_[np.nan, z[1:]]]), n)
        raise AssertionError("NaN accepted")
    except ValueError:
        pass


def _cli_raw(*args):
    env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, "-c", "import sys; from hapdc.cli import main; sys.exit(main(sys.argv[1:]))"]
                          + list(args), capture_output=True, text=True, env=env)


def test_v02_cli_drops_and_logs_rank_deficient_covariates():
    rng = np.random.default_rng(31)
    n, m = 80, 40
    H = rng.binomial(1, .3, (2 * n, m)).astype(np.int8)
    d = tempfile.mkdtemp()
    ids = ["s%d" % i for i in range(n)]
    rows = [("1", 1000 * (j + 1), "A", "C", ["%d|%d" % (H[2 * i, j], H[2 * i + 1, j]) for i in range(n)])
            for j in range(m)]
    _write_vcf(os.path.join(d, "g.vcf.gz"), rows, ids)
    y = H[0::2, 3] + H[1::2, 3] + rng.normal(size=n)
    with open(os.path.join(d, "p.txt"), "w") as f:
        f.write("IID\tt\n" + "".join("%s\t%.6f\n" % (i, v) for i, v in zip(ids, y)))
    z = rng.normal(size=n)
    with open(os.path.join(d, "c.txt"), "w") as f:
        f.write("IID\tsex\tpc1\tpc1copy\n" + "".join("%s\t1\t%.6f\t%.6f\n" % (i, v, v) for i, v in zip(ids, z)))
    with open(os.path.join(d, "c1.txt"), "w") as f:
        f.write("IID\tpc1\n" + "".join("%s\t%.6f\n" % (i, v) for i, v in zip(ids, z)))
    pr = os.path.join(d, "prep")
    _cli("prep", os.path.join(d, "g.vcf.gz"), "auto", os.path.join(d, "p.txt"), pr, "--grm-thin", "1")
    o1 = _cli("ms", pr, os.path.join(d, "o1"), "0", "--anchor-min", "8", "--covar", os.path.join(d, "c.txt"))
    assert "covariate sex dropped before REML: constant" in o1, o1[-1500:]
    assert "covariate pc1copy dropped before REML: duplicate of pc1" in o1
    _cli("ms", pr, os.path.join(d, "o2"), "0", "--anchor-min", "8", "--covar", os.path.join(d, "c1.txt"))
    j1 = json.load(open(os.path.join(d, "o1", "p.summary.json")))
    assert j1["covariates_used"] == ["pc1"]
    assert [x["name"] for x in j1["covariates_dropped"]] == ["sex", "pc1copy"]
    t1 = open(os.path.join(d, "o1", "p.anchors.tsv")).read()
    t2 = open(os.path.join(d, "o2", "p.anchors.tsv")).read()
    assert t1 == t2
    r3 = _cli_raw("ms", pr, os.path.join(d, "o3"), "0", "--anchor-min", "8", "--covar", os.path.join(d, "c.txt"),
                  "--covar-rank", "error")
    assert r3.returncode != 0 and "sex (constant)" in (r3.stderr + r3.stdout), r3.stderr[-800:]
    # V15: no hardcoded lambda reference range; an optional user range is only reported
    assert "0.61" not in o1 and "WARNING: lambda_gc" not in o1
    assert j1["lambda_ref"] is None
    o4 = _cli("ms", pr, os.path.join(d, "o4"), "0", "--anchor-min", "8", "--lambda-ref", "0.5,0.9")
    assert "user-supplied reference range 0.500-0.900" in o4
    assert json.load(open(os.path.join(d, "o4", "p.summary.json")))["lambda_ref"] == [0.5, 0.9]


def test_v03_provenance_keys_are_unique_per_path_and_content():
    from types import SimpleNamespace
    from hapdc.cli import _provenance
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for chrom in ("chr1", "chr2", "chr3"):
            os.mkdir(os.path.join(d, chrom))
            p = os.path.join(d, chrom, "height.anchors.tsv")
            open(p, "w").write(chrom if chrom != "chr3" else "chr1")   # chr3 has the same content as chr1
            paths.append(p)
        pr = _provenance(SimpleNamespace(cmd="merge"), paths + [paths[0]])
        assert pr["n_inputs"] == 3 and len(pr["inputs"]) == 3, pr["inputs"]
        recs = list(pr["inputs"].values())
        assert sorted(r["path"] for r in recs) == sorted(os.path.realpath(p) for p in paths)
        for k, r in pr["inputs"].items():
            assert k.endswith("#md5=" + r["md5"]) and len(r["md5"]) == 32 and len(r["sha256"]) == 64
        assert recs[0]["md5"] == recs[2]["md5"] or len({r["md5"] for r in recs}) == 2


def test_v03_merge_records_every_shard():
    with tempfile.TemporaryDirectory() as d:
        tabs = []
        hdr = "chrom\tstart_bp\tend_bp\tn_markers\tP\tlead_scale\tlead_unit_start\tlead_unit_end\tP_nonadditive\tsig_bonf\n"
        for k, chrom in enumerate(("1", "2")):
            sub_ = os.path.join(d, "chr" + chrom)
            os.mkdir(sub_)
            p = os.path.join(sub_, "height.anchors.tsv")
            open(p, "w").write(hdr + "%s\t1\t100\t16\t%.3g\tSNP\t1\t1\t0.5\t0\n" % (chrom, 0.01 * (k + 1)))
            json.dump(dict(trait="height", n=100 + k, transform="rank_inverse_normal", covar=None,
                           provenance=dict(hapdc_version="test")), open(os.path.join(sub_, "height.summary.json"), "w"))
            tabs.append(p)
        out = os.path.join(d, "merged")
        _cli("merge", out, *tabs)
        m = json.load(open(out + ".summary.json"))
        assert len(m["shards"]) == 2 and [s_["n"] for s_ in m["shards"]] == [100, 101]
        assert m["provenance"]["n_inputs"] == 4



# ---------------------------------------------------------------- covariates, identifiers, precision
def _w01_setup(seed=720, n=80, m=8):
    rng = np.random.default_rng(seed)
    H = rng.binomial(1, .3, (2 * n, m)).astype(np.int8)
    meta = [("1", "s%d" % j, 100 + j, "A", "C") for j in range(m)]
    Kx = rng.normal(size=(n, 20)); K = Kx @ Kx.T / 20
    S, U = np.linalg.eigh(K); S = np.maximum(S, 1e-6)
    y = rng.normal(size=n); z = rng.normal(size=n)
    return rng, H, meta, U, S, y, z


def _w01_p(H, meta, U, S, y, C=None):
    sc = scan.Scan(H, meta, U, S, y, covar=C)
    r = ms.ms_scan(sc, [(0, H.shape[1], [(0, H.shape[1])])], variants=("D",))
    return float(r["P_MS"]["D"][0, 0]), sc


def test_w01_covariate_shift_scale_and_order_invariance():
    rng, H, meta, U, S, y, z = _w01_setup()
    yy = y + .9 * z
    p0, s0 = _w01_p(H, meta, U, S, yy, z[:, None])
    assert s0.nfix == 2
    for shift in (1e4, 1e7, 1e8, 1e9):
        p, sc = _w01_p(H, meta, U, S, yy, (z + shift)[:, None])
        assert sc.nfix == 2 and sc.covar_dropped == [], shift
        assert abs(np.log(sc.delta) - np.log(s0.delta)) < 1e-4, (shift, sc.delta, s0.delta)
        assert abs(p - p0) < 1e-5 * max(p0, 1e-300) + 1e-12, (shift, p, p0)
    for scale in (1e-6, 1e6):
        p, sc = _w01_p(H, meta, U, S, yy, (scale * z)[:, None])
        assert sc.nfix == 2 and abs(p - p0) < 1e-9, (scale, p, p0)
    x2 = rng.normal(size=len(y))
    pa, _ = _w01_p(H, meta, U, S, yy, np.column_stack([z, x2]))
    pb, _ = _w01_p(H, meta, U, S, yy, np.column_stack([x2 + 3e7, z - 5e8]))
    assert abs(pa - pb) < 1e-6 * pa, (pa, pb)


def test_w01_same_linear_model_control_and_near_collinear_reml():
    rng, H, meta, U, S, y, z = _w01_setup()
    x = rng.normal(size=len(y))
    pa, sa = _w01_p(H, meta, U, S, y, x[:, None])
    pb, sb = _w01_p(H, meta, U, S, y, (x + 1e8)[:, None])
    assert sa.nfix == sb.nfix == 2 and abs(pa - pb) < 1e-5 * pa, (pa, pb)
    for eps in (1e-4, 1e-6, 1e-7, 3e-8):          # near-collinear covariates: REML must stay finite (no LinAlgError)
        p, sc = _w01_p(H, meta, U, S, y, np.column_stack([z, z + eps * rng.normal(size=len(y))]))
        assert np.isfinite(p) and np.isfinite(sc.delta) and sc.nfix == 3, eps
    # an exact affine copy, and a copy with a large offset, are dropped as duplicates
    for C in (np.column_stack([z, 2 * z + 7]), np.column_stack([z, z + 1e8])):
        _, kn, dr = core.check_fixed_effects(C, len(y), names=["a", "b"])
        assert kn == ["a"] and dr[0]["reason"] == "duplicate of a", dr
    # a linear combination of two columns (with offsets) is named as such
    C = np.column_stack([z, x, z - 2 * x + 1e6])
    _, kn, dr = core.check_fixed_effects(C, len(y), names=["a", "b", "c"])
    assert kn == ["a", "b"] and dr == [dict(name="c", index=2, reason="linear combination of earlier columns")]
    # REML through QR agrees with the normal-equation profile on a well-conditioned design
    X = U.T @ np.column_stack([np.ones(len(y)), z])
    yr = U.T @ y
    for d in (0.1, 1.0, 10.0):
        w = 1 / (S + d); Xw = X * w[:, None]; XtX = X.T @ Xw
        b = np.linalg.solve(XtX, Xw.T @ yr); r_ = yr - X @ b
        rss, ld = core._gls_profile(yr, S, X, d)
        assert abs(rss - (w * r_ * r_).sum()) < 1e-9 * rss and abs(ld - np.linalg.slogdet(XtX)[1]) < 1e-9


def test_w01_constant_trait_is_refused():
    _, H, meta, U, S, y, z = _w01_setup()
    for v in (0.0, 1.0, 10.0):
        try:
            scan.Scan(H, meta, U, S, np.full(len(y), v))
            raise AssertionError("constant trait accepted")
        except core.AnalysisFailed as e:
            assert "no variation" in str(e)


def _write_grm(pref, pairs, K):
    with open(pref + ".grm.id", "w") as f:
        f.write("".join("%s %s\n" % p for p in pairs))
    np.asarray(K, "<f4")[np.tril_indices(len(pairs))].tofile(pref + ".grm.bin")


def test_w02_repeated_identifiers_are_rejected():
    with tempfile.TemporaryDirectory() as d:
        K = np.array([[1, .2, .3], [.2, 1, .4], [.3, .4, 1]])
        _write_grm(os.path.join(d, "dup"), [("F1", "a"), ("F2", "a"), ("F3", "b")], K)
        ids, K2, fids = io.read_grm(os.path.join(d, "dup"), return_fid=True)
        for vcf in (["a", "b"], ["b", "a"]):
            try:
                io.match_ids(vcf, ids, fids)
                raise AssertionError("ambiguous IID accepted")
            except io.IdentityError as e:
                assert "a" in str(e) and "FID" in str(e)
        # the FID_IID composite key maps the same GRM one to one, in VCF order
        keep, vr, gr = io.match_ids(["F3_b", "F1_a", "F2_a"], ids, fids)
        assert keep == ["F1_a", "F2_a", "F3_b"] and vr == [1, 2, 0] and gr == [0, 1, 2]
        assert np.allclose(K2[np.ix_(gr, gr)], K, atol=1e-7)
        # repeated (FID, IID) pair and a truncated .grm.bin
        _write_grm(os.path.join(d, "dup2"), [("F1", "a"), ("F1", "a"), ("F3", "b")], K)
        for pref in ("dup2",):
            try:
                io.read_grm(os.path.join(d, pref)); raise AssertionError("repeated pair accepted")
            except io.IdentityError:
                pass
        with open(os.path.join(d, "dup.grm.bin"), "ab") as f:
            f.write(b"\0\0\0\0")
        try:
            io.read_grm(os.path.join(d, "dup")); raise AssertionError("bad .grm.bin length accepted")
        except io.IdentityError:
            pass
        # phenotype and covariate tables: a repeated IID is an error, not "last row wins"
        for fn, rd_ in (("p.txt", lambda p: io.read_pheno_table(p, ["a", "b"])),
                        ("c.txt", lambda p: _cli_mod()._read_covar(p, ["a", "b"]))):
            with open(os.path.join(d, fn), "w") as f:
                f.write("IID\ty\na\t1\na\t999\nb\t2\n")
            try:
                rd_(os.path.join(d, fn)); raise AssertionError("%s: repeated IID accepted" % fn)
            except io.IdentityError as e:
                assert "repeated" in str(e)
        # FID IID header with composite sample names
        with open(os.path.join(d, "p2.txt"), "w") as f:
            f.write("FID\tIID\ty\nF1\ta\t1\nF2\ta\t5\nF3\tb\t2\n")
        Y, hn = io.read_pheno_table(os.path.join(d, "p2.txt"), ["F2_a", "F3_b", "F1_a"])
        assert hn == ["y"] and Y[:, 0].tolist() == [5.0, 2.0, 1.0]
        # VCF with a repeated sample name
        _write_vcf(os.path.join(d, "g.vcf.gz"), [("1", 10, "A", "C", ["0|1", "1|0", "0|0"])], ["a", "b", "a"])
        try:
            io.read_vcf(os.path.join(d, "g.vcf.gz")); raise AssertionError("repeated VCF sample accepted")
        except io.IdentityError:
            pass


def _cli_mod():
    from hapdc import cli
    return cli


def test_w02_prep_with_ambiguous_grm_exits_nonzero():
    with tempfile.TemporaryDirectory() as d:
        ids = _toy_prep(d)
        n = len(ids)
        K = np.eye(n) * .9 + .1
        pairs = [("F%d" % i, i_) for i, i_ in enumerate(ids)]
        pairs[1] = ("Fx", ids[0])                       # IID i0 twice, different FIDs
        _write_grm(os.path.join(d, "k"), pairs, K)
        r = _cli_raw("prep", os.path.join(d, "g.vcf.gz"), os.path.join(d, "k"), os.path.join(d, "ph.txt"),
                     os.path.join(d, "out"))
        assert r.returncode == 1 and "FID" in r.stderr, (r.returncode, r.stderr[-500:])
        assert not os.path.exists(os.path.join(d, "out", "meta.json"))
        # with unique IIDs the external GRM is reordered to the VCF sample order
        pairs = [("F%d" % i, i_) for i, i_ in enumerate(ids)][::-1]
        Kr = np.random.default_rng(1).normal(size=(n, n)); Kr = Kr @ Kr.T / n
        _write_grm(os.path.join(d, "k2"), pairs, Kr)
        _cli("prep", os.path.join(d, "g.vcf.gz"), os.path.join(d, "k2"), os.path.join(d, "ph.txt"),
             os.path.join(d, "out2"))
        m = json.load(open(os.path.join(d, "out2", "meta.json")))
        Kk = np.load(os.path.join(d, "out2", "K_keep.npy")).astype(float)
        keep = np.load(os.path.join(d, "out2", "keep.npy"))
        gix = [n - 1 - ids.index(i) for i in m["ids"]]   # position in the reversed GRM
        assert np.allclose(Kk[np.ix_(keep, keep)], Kr.astype("<f4")[np.ix_(gix, gix)], atol=1e-6)


def _ms_fixture(d, n=50, m=8, seed=723):
    """prep with an identity GRM, as in the reviewer's cli_checks.py."""
    rng = np.random.default_rng(seed)
    H = rng.binomial(1, .3, (2 * n, m)).astype(np.int8)
    ids = ["a%d" % i for i in range(n)]
    _write_vcf(os.path.join(d, "g.vcf.gz"), [("1", j + 1, "A", "C", ["%d|%d" % (H[2 * i, j], H[2 * i + 1, j])
                                                                    for i in range(n)]) for j in range(m)], ids)
    with open(os.path.join(d, "p.txt"), "w") as f:
        f.write("IID\ty\n" + "".join("%s\t%.17g\n" % (i, v) for i, v in zip(ids, rng.normal(size=n))))
    prep = os.path.join(d, "prep")
    _cli("prep", os.path.join(d, "g.vcf.gz"), "auto", os.path.join(d, "p.txt"), prep, "--grm-thin", "1")
    np.save(os.path.join(prep, "K_keep.npy"), np.eye(n, dtype=np.float32))
    np.savez(os.path.join(prep, "eig.npz"), S=np.ones(n), U=np.eye(n))
    return prep, H, n, m, rng


def test_w03_full_precision_table_and_merge_agree_at_threshold():
    from hapdc import cli as C
    with tempfile.TemporaryDirectory() as d:
        prep, H, n, m, rng = _ms_fixture(d)
        markers = [tuple(x) for x in json.load(open(os.path.join(prep, "markers.json")))[0]]
        anc = ms.anchors(core.ld_blocks(H, np.arange(1, m + 1), dmin=.8, max_span=200000, min_mk=2,
                                        hap_cover=.8, minf=.05)["blocks"], 16)
        causal = (H[0::2, 0] + H[1::2, 0]).astype(float)
        y0 = np.load(os.path.join(prep, "traits.npy"))[:, 0]
        U, S = np.eye(n), np.ones(n)
        ev = lambda a_: float(ms.ms_scan(scan.Scan(H, markers, U, S, y0 + a_ * causal), anc, variants=("D",),
                                         nadds=False)["P_MS"]["D"][0, 0])
        lo, hi = 0., 10.
        assert ev(lo) > .05 > ev(hi)
        for target, want in ((.049999, 1), (.050001, 0)):      # either side of alpha / 1 anchor
            l_, h_ = lo, hi
            for _ in range(45):
                a_ = (l_ + h_) / 2
                l_, h_ = (a_, h_) if ev(a_) > target else (l_, a_)
            a_ = (l_ + h_) / 2
            np.save(os.path.join(prep, "traits.npy"), (y0 + a_ * causal)[:, None])
            run = os.path.join(d, "run%d" % want)
            _cli("ms", prep, run, "0", "--transform", "none")
            tab = os.path.join(run, "p.anchors.tsv")
            row = open(tab).read().splitlines()[1].split("\t")
            assert float(row[4]) == ev(a_), (row[4], ev(a_))           # exact round trip
            s0 = json.load(open(os.path.join(run, "p.summary.json")))
            _cli("merge", os.path.join(d, "mg%d" % want), tab)
            s1 = json.load(open(os.path.join(d, "mg%d.summary.json" % want)))
            assert s0["n_sig"] == s1["n_sig"] == want == int(row[-1]), (s0["n_sig"], s1["n_sig"], want)
            assert s1["low_precision_tables"] == []
        # a 4-significant-digit table (written by an earlier version) is flagged
        old = os.path.join(d, "old.anchors.tsv")
        L = open(tab).read().splitlines(); r_ = L[1].split("\t"); r_[4] = "%.4g" % float(r_[4]); r_[8] = "%.4g" % float(r_[8])
        open(old, "w").write(L[0] + "\n" + "\t".join(r_) + "\n")
        out = _cli("merge", os.path.join(d, "mold"), old)
        assert "WARNING" in out and json.load(open(os.path.join(d, "mold.summary.json")))["low_precision_tables"]
        assert C._p_digits(tab, 4) > 4 and C._p_digits(old, 4) <= 4


def test_w04_merge_summary_path_follows_the_real_suffix():
    from hapdc import cli as C
    with tempfile.TemporaryDirectory() as d:
        for o, t, s in (("x", "x.anchors.tsv", "x.summary.json"), ("x.anchors.tsv", "x.anchors.tsv", "x.summary.json"),
                        ("custom_output_name.tsv", "custom_output_name.tsv", "custom_output_name.summary.json"),
                        ("a.b.tsv", "a.b.tsv", "a.b.summary.json"), ("r.tsv", "r.tsv", "r.summary.json")):
            tt, ss = C._merge_paths(os.path.join(d, o))
            assert (os.path.basename(tt), os.path.basename(ss)) == (t, s), (o, tt, ss)
            assert os.path.dirname(ss) == d
        prep, H, n, m, rng = _ms_fixture(d)
        _cli("ms", prep, os.path.join(d, "run"), "0")
        tab = os.path.join(d, "run", "p.anchors.tsv")
        _cli("merge", os.path.join(d, "custom_output_name.tsv"), tab)
        assert os.path.exists(os.path.join(d, "custom_output_name.summary.json"))
        assert not [f for f in os.listdir(d) if f.startswith("custom_out.")]
        r = _cli_raw("merge", tab, tab)                     # would overwrite its own input
        assert r.returncode != 0 and "overwrite" in r.stderr


def test_w05_constant_trait_fails_with_status_and_exit_code():
    with tempfile.TemporaryDirectory() as d:
        prep, H, n, m, rng = _ms_fixture(d)
        for v in (0.0, 1.0, 10.0):
            np.save(os.path.join(prep, "traits.npy"), np.full((n, 1), v))
            out = os.path.join(d, "c%g" % v)
            r = _cli_raw("ms", prep, out, "0")
            assert r.returncode == 2 and "no variation" in r.stderr, (r.returncode, r.stderr[-400:])
            sj = json.load(open(os.path.join(out, "p.summary.json")))
            assert sj["status"] == "FAILED" and sj["n_sig"] is None and "no variation" in sj["reason"]
            assert not os.path.exists(os.path.join(out, "p.anchors.tsv"))
            r = _cli_raw("scan", prep, os.path.join(d, "s%g" % v), "0", "--nperm", "0", "--quiet")
            assert r.returncode == 2 and "no variation" in r.stderr
        # merge refuses a FAILED shard sitting next to a stale table
        np.save(os.path.join(prep, "traits.npy"), rng.normal(size=(n, 1)))
        out = os.path.join(d, "c1")
        _cli("ms", prep, out, "0")
        np.save(os.path.join(prep, "traits.npy"), np.ones((n, 1)))
        assert _cli_raw("ms", prep, out, "0").returncode == 2
        assert not os.path.exists(os.path.join(out, "p.anchors.tsv")) and os.path.exists(os.path.join(out, "p.anchors.tsv.stale"))
        # one trait missing for everybody except two samples: no residual df
        y = np.full((n, 1), np.nan); y[:2, 0] = (1.0, 2.0)
        np.save(os.path.join(prep, "traits.npy"), y)
        r = _cli_raw("ms", prep, os.path.join(d, "few"), "0")
        assert r.returncode == 2, r.stderr[-400:]


def test_w06_report_reads_the_default_ld_scan():
    with tempfile.TemporaryDirectory() as d:
        prep, H, n, m, rng = _ms_fixture(d)
        sd = os.path.join(d, "scan")
        _cli("scan", prep, sd, "0", "--nperm", "0", "--quiet")
        assert os.path.exists(os.path.join(sd, "scan_ld.npz"))
        out = _cli("report", sd, "--out", os.path.join(d, "rep"))
        svg = sorted(os.listdir(os.path.join(d, "rep")))
        assert svg and all(f.endswith("_ld.svg") for f in svg), svg
        r = _cli_raw("report", os.path.join(d, "empty_dir_that_is_not_a_scan"))
        assert r.returncode == 1 and "ms" in r.stdout


def _edge_vcf(d, ids, H, chrom="1", extra=()):
    rows = [(chrom, 100 * (j + 1), "A", "C", ["%d|%d" % (H[2 * i, j], H[2 * i + 1, j]) for i in range(len(ids))])
            for j in range(H.shape[1])]
    rows += list(extra)
    rows.sort(key=lambda r: r[1])
    _write_vcf(os.path.join(d, "g.vcf.gz"), rows, ids)


def test_edge_inputs_missing_multiallelic_unphased_chr_prefix_and_sample_order():
    rng = np.random.default_rng(11)
    n, m = 40, 10
    ids = ["s%d" % i for i in range(n)]
    H = rng.binomial(1, .35, (2 * n, m)).astype(np.int8)
    H[0, :] = 1 - H[1, :]                      # no monomorphic marker
    y = rng.normal(size=n)
    het = ["0|1"] * n
    extra = [("chr1", 105, "A", "C", ["./."] + het[1:]),              # missing call
             ("chr1", 115, "A", "C,G", het),                          # multi-allelic ALT
             ("chr1", 125, "A", "C", ["0|2"] + het[1:]),              # allele index 2
             ("chr1", 135, "A", "C", ["0/1"] + het[1:]),              # unphased heterozygote
             ("chr1", 145, "A", "C", ["1/1"] + het[1:]),              # unphased homozygote: allowed
             ("chr1", 155, "A", "C", ["1"] + het[1:])]                # haploid
    with tempfile.TemporaryDirectory() as d:
        _edge_vcf(d, ids, H, chrom="chr1", extra=extra)
        def pheno(fn, order):
            with open(os.path.join(d, fn), "w") as f:
                f.write("IID\ty\n" + "".join("%s\t%.17g\n" % (ids[i], y[i]) for i in order))
        pheno("p.txt", range(n))
        pheno("p_rev.txt", list(range(n))[::-1])
        outs = []
        for fn in ("p.txt", "p_rev.txt"):
            pr = os.path.join(d, "prep_" + fn)
            _cli("prep", os.path.join(d, "g.vcf.gz"), "auto", os.path.join(d, fn), pr, "--grm-thin", "1",
                 "--skip-unphased")
            mj = json.load(open(os.path.join(pr, "meta.json")))
            assert mj["unphased_policy"] == "skip"
            assert mj["dropped"] == dict(missing=1, monomorphic=0, multiallelic=2, unphased=1, nondiploid=1), mj["dropped"]
            assert mj["m"] == m + 1 and mj["ids"] == ids
            assert np.array_equal(np.load(os.path.join(pr, "traits.npy"))[:, 0], y)
            for ch in ("1", "chr1", "CHR1"):
                o = os.path.join(d, "ms_%s_%s" % (fn, ch))
                _cli("ms", pr, o, "0", "--chrom", ch, "--transform", "none")
                tab = [f for f in os.listdir(o) if f.endswith(".anchors.tsv")]
                assert len(tab) == 1, os.listdir(o)
                outs.append(open(os.path.join(o, tab[0])).read())
            r = _cli_raw("ms", pr, os.path.join(d, "bad"), "0", "--chrom", "2")
            assert r.returncode != 0 and "not in the data" in r.stderr
        assert all(o == outs[0] for o in outs), "results depend on phenotype row order or chr prefix"
        # a phenotype file with samples not in the VCF and VCF samples without phenotype
        with open(os.path.join(d, "p_part.txt"), "w") as f:
            f.write("IID\ty\nzz\t1.0\n" + "".join("%s\t%.17g\n" % (ids[i], y[i]) for i in range(0, n, 2)))
        pr = os.path.join(d, "prep_part")
        _cli("prep", os.path.join(d, "g.vcf.gz"), "auto", os.path.join(d, "p_part.txt"), pr, "--grm-thin", "1",
             "--skip-unphased")
        assert json.load(open(os.path.join(pr, "meta.json")))["ids"] == ids[0::2]



def test_unphased_heterozygote_is_an_error_unless_skip_is_requested():
    """One 0/1 call stops prep (exit 1, message names the marker and sample);
    --skip-unphased drops that marker and records it; 1/1 alone is accepted."""
    rng = np.random.default_rng(5)
    n, m = 30, 8
    ids = ["s%d" % i for i in range(n)]
    H = rng.binomial(1, .4, (2 * n, m)).astype(np.int8)
    H[0, :] = 1 - H[1, :]
    het = ["0|1"] * n
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "p.txt"), "w") as f:
            f.write("IID\ty\n" + "".join("%s\t%.6f\n" % (s, v) for s, v in zip(ids, rng.normal(size=n))))
        _edge_vcf(d, ids, H, extra=[("1", 135, "A", "C", het[:3] + ["1/0"] + het[4:])])
        r = _cli_raw("prep", os.path.join(d, "g.vcf.gz"), "auto", os.path.join(d, "p.txt"),
                     os.path.join(d, "pr"), "--grm-thin", "1")
        rc, out = r.returncode, r.stdout + r.stderr
        assert rc == 1 and "Traceback" not in out and "unphased" in out and "s3" in out and ":135" in out, (rc, out)
        assert not os.path.exists(os.path.join(d, "pr", "meta.json"))
        _cli("prep", os.path.join(d, "g.vcf.gz"), "auto", os.path.join(d, "p.txt"), os.path.join(d, "pr2"),
             "--grm-thin", "1", "--skip-unphased")
        mj = json.load(open(os.path.join(d, "pr2", "meta.json")))
        assert mj["dropped"]["unphased"] == 1 and mj["m"] == m and mj["unphased_policy"] == "skip"
        _edge_vcf(d, ids, H, extra=[("1", 135, "A", "C", het[:3] + ["1/1"] + het[4:])])
        _cli("prep", os.path.join(d, "g.vcf.gz"), "auto", os.path.join(d, "p.txt"), os.path.join(d, "pr3"),
             "--grm-thin", "1")
        mj = json.load(open(os.path.join(d, "pr3", "meta.json")))
        assert mj["dropped"]["unphased"] == 0 and mj["m"] == m + 1 and mj["unphased_policy"] == "error"


def test_read_grm_without_fid_rejects_repeated_iid():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "dup.grm.id"), "w") as f:
        f.write("F1 a\nF2 a\nF3 b\n")
    np.array([1, .2, 1, .3, .4, 1], dtype="<f4").tofile(os.path.join(d, "dup.grm.bin"))
    try:
        io.read_grm(os.path.join(d, "dup"))
        raise AssertionError("read_grm returned a repeated IID")
    except io.IdentityError as e:
        assert "'a'" in str(e) or " a" in str(e), e
    ids, K, fids = io.read_grm(os.path.join(d, "dup"), return_fid=True)
    assert list(zip(fids, ids)) == [("F1", "a"), ("F2", "a"), ("F3", "b")] and K.shape == (3, 3)


def test_merge_reads_shard_summary_for_renamed_tables():
    """Shard tables named a.tsv, a.b.tsv, a (no suffix) or a.anchors.tsv still get
    their own summary (same stem + .summary.json) recorded in the merged summary."""
    with tempfile.TemporaryDirectory() as d:
        _toy_prep(d)
        pr = os.path.join(d, "prep")
        _cli("prep", os.path.join(d, "g.vcf.gz"), "auto", os.path.join(d, "ph.txt"), pr, "--grm-thin", "1")
        _cli("ms", pr, os.path.join(d, "ms"), "0", "--transform", "none")
        tab = glob.glob(os.path.join(d, "ms", "*.anchors.tsv"))[0]
        sj = tab[:-len(".anchors.tsv")] + ".summary.json"
        trait = json.load(open(sj))["trait"]
        for nm, stem in [("c1.tsv", "c1"), ("c2.b.tsv", "c2.b"), ("c3", "c3"), ("c4.anchors.tsv", "c4")]:
            shutil.copy(tab, os.path.join(d, nm)); shutil.copy(sj, os.path.join(d, stem + ".summary.json"))
            _cli("merge", os.path.join(d, "m_" + stem), os.path.join(d, nm))
            rec = json.load(open(os.path.join(d, "m_" + stem + ".summary.json")))["shards"][0]
            assert rec["trait"] == trait and rec["summary"].endswith(stem + ".summary.json"), rec


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
