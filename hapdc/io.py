
"""hapdc.io -- readers and writers for phased genotypes, GRM and phenotypes."""
import gzip
import numpy as np

__all__ = ["iter_vcf", "read_vcf", "read_grm", "read_pheno", "read_pheno_table", "write_matrix",
           "InputError", "IdentityError", "UnphasedError", "check_unique", "match_ids", "read_keyed_table"]


class InputError(ValueError):
    """An input file that hapdc refuses to interpret (`hapdc` exits 1)."""


class IdentityError(InputError):
    """Sample identifiers that cannot be mapped one to one (`hapdc` exits 1)."""


class UnphasedError(InputError):
    """An unphased heterozygous genotype in the VCF (`hapdc` exits 1)."""


def _dups(keys):
    seen, d = set(), []
    for k in keys:
        if k in seen and k not in d:
            d.append(k)
        seen.add(k)
    return d


def check_unique(keys, what):
    """Stop with the list of repeated identifiers when `keys` is not unique."""
    d = _dups(list(keys))
    if d:
        raise IdentityError("%s: %d repeated sample identifier(s), e.g. %s; identifiers must be unique"
                            % (what, len(d), ", ".join(map(str, d[:10]))))


def iter_vcf(path, chunk=1, unphased="error"):
    """Yield (sample_ids, hap_matrix, meta, dropped) per chunk of markers.

    hap_matrix is (n_marker, n_hap) int8 inside a chunk; read_vcf returns the
    public (n_hap, n_marker) layout with rows 2i, 2i+1 the two haplotypes of
    sample i. Only biallelic, fully phased, diploid, non-missing markers are
    used. Markers are dropped (and counted in `dropped`) when they are
    multi-allelic (ALT lists more than one allele, or a genotype carries an
    allele index other than 0/1), carry a missing call or a haploid call, or
    show no variation.

    An unphased heterozygote (`0/1`, `1/0`) has no haplotype assignment. With
    unphased="error" (default) it raises UnphasedError naming the marker and
    sample; with unphased="skip" the marker is dropped and counted under
    "unphased". Unphased homozygotes (`0/0`, `1/1`) are unambiguous and used.
    """
    if unphased not in ("error", "skip"):
        raise ValueError("unphased must be 'error' or 'skip'")
    ids = None
    keys = ("missing", "monomorphic", "multiallelic", "unphased", "nondiploid")
    G, mk = [], []
    dr = dict.fromkeys(keys, 0)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for ln in f:
            if ln.startswith("##"):
                continue
            p = ln.rstrip("\n").split("\t")
            if ln.startswith("#CHROM"):
                ids = p[9:]
                check_unique(ids, "VCF %s header" % path)
                continue
            if "," in p[4]:
                dr["multiallelic"] += 1
                continue
            gts = [g.split(":", 1)[0] for g in p[9:]]
            row = np.empty(2 * len(ids), np.int8)
            bad = None
            for j, gt in enumerate(gts):
                if "|" in gt:
                    a = gt.split("|")
                elif "/" in gt:
                    a = gt.split("/")
                    if len(a) == 2 and a[0] != a[1] and "." not in a:
                        if unphased == "error":
                            raise UnphasedError(
                                "VCF %s: unphased heterozygous genotype '%s' at %s:%s (sample %s). "
                                "hapdc needs phased haplotypes: phase the VCF first, or pass "
                                "--skip-unphased to drop every marker with an unphased heterozygote"
                                % (path, gt, p[0], p[1], ids[j]))
                        bad = "unphased"
                        break
                else:
                    a = [gt]
                if "." in a:
                    bad = "missing"
                    break
                if len(a) != 2:
                    bad = "nondiploid"
                    break
                if a[0] not in ("0", "1") or a[1] not in ("0", "1"):
                    bad = "multiallelic"
                    break
                row[2 * j] = a[0] == "1"
                row[2 * j + 1] = a[1] == "1"
            if bad is not None:
                dr[bad] += 1
                continue
            if row.min() == row.max():
                dr["monomorphic"] += 1
                continue
            G.append(row)
            mk.append((p[0], p[2] if p[2] not in (".", "") else "%s:%s" % (p[0], p[1]),
                       int(p[1]), p[3], p[4]))
            if len(G) >= chunk:
                yield ids, np.array(G), mk, dr
                G, mk = [], []
                dr = dict.fromkeys(keys, 0)
    if G or any(dr.values()):
        yield ids, (np.array(G) if G else np.zeros((0, 2 * len(ids or [])), np.int8)), mk, dr


def read_vcf(path, nmax=None, unphased="error"):
    """Read a whole phased VCF. Returns (ids, H (n_hap, m) int8, meta, dropped).

    `dropped` counts markers skipped per reason (see iter_vcf). nmax, if given,
    caps the number of markers read. unphased: "error" (default) or "skip",
    see iter_vcf.
    """
    ids = None
    outs = []
    dropped = {}
    nread = 0
    for cids, G, mk, dr in iter_vcf(path, chunk=2000, unphased=unphased):
        if ids is None:
            ids = cids
        if len(mk):
            outs.append((G, mk))
            nread += G.shape[0]          # chunk buffers are (marker, haplotype)
        for k_, v_ in dr.items():
            dropped[k_] = dropped.get(k_, 0) + v_
        if nmax and nread >= nmax:
            break
    if not outs:
        raise SystemExit("no usable markers in %s (dropped: %s)" % (path, dropped))
    H = np.concatenate([o[0] for o in outs], axis=0)
    meta = [m for o in outs for m in o[1]]
    if nmax:
        H, meta = H[:nmax, :], meta[:nmax]
    # internal buffers are (marker, haplotype); the public API is (haplotype, marker)
    return ids, np.ascontiguousarray(H.T), meta, dropped


def read_grm(pref, return_fid=False):
    """Read a GCTA .grm.{id,bin,n} triple into (ids, K) (and FIDs on request).

    The FID/IID pairs must be unique, and the .grm.bin length must equal
    n(n+1)/2. An IID repeated under different FIDs is only returned with
    return_fid=True (the caller then keys samples by (FID, IID), as `prep`
    does through `match_ids`); without FIDs the IIDs alone must be unique and a
    repeat raises IdentityError.
    """
    rec = [l.split() for l in open(pref + ".grm.id") if l.strip()]
    bad = [i + 1 for i, r in enumerate(rec) if len(r) < 2]
    if bad:
        raise IdentityError("%s.grm.id: lines %s do not have FID and IID" % (pref, bad[:10]))
    fids, ids = [r[0] for r in rec], [r[1] for r in rec]
    check_unique(list(zip(fids, ids)), "%s.grm.id (FID, IID)" % pref)
    if not return_fid:
        check_unique(ids, "%s.grm.id IID (call read_grm(..., return_fid=True) to key by FID and IID)" % pref)
    n = len(ids)
    v = np.fromfile(pref + ".grm.bin", dtype="<f4")
    if v.size != n * (n + 1) // 2:
        raise IdentityError("%s.grm.bin holds %d values; %d samples in .grm.id need %d"
                            % (pref, v.size, n, n * (n + 1) // 2))
    K = np.zeros((n, n))
    K[np.tril_indices(n)] = v
    K = K + K.T - np.diag(np.diag(K))
    return (ids, K, fids) if return_fid else (ids, K)


def match_ids(vcf_ids, grm_iids, grm_fids=None, return_mode=False):
    """Positions of the shared samples, one to one: (keep, vcf_rows, grm_rows).

    GRM samples are matched to VCF sample names by IID. When IIDs repeat in
    the GRM (different FIDs), the composite key FID_IID is used instead, and
    the VCF must name its samples that way; otherwise the mapping is ambiguous
    and an error is raised. With return_mode=True a fourth value records the
    selected mode (iid or fid_iid) for phenotype/covariate alignment.
    """
    check_unique(vcf_ids, "VCF samples")
    mode = "iid"
    keys = list(grm_iids)
    d = _dups(keys)
    if d:
        if grm_fids is None:
            raise IdentityError("GRM: repeated IID(s) %s" % ", ".join(d[:10]))
        comp = ["%s_%s" % (f, i) for f, i in zip(grm_fids, grm_iids)]
        vset = set(vcf_ids)
        if _dups(comp) or not any(c in vset for c in comp):
            raise IdentityError("GRM: IID(s) %s occur with more than one FID and the VCF does not name "
                                "samples as FID_IID; make the IIDs unique (e.g. FID_IID in both files)"
                                % ", ".join(d[:10]))
        keys = comp
        mode = "fid_iid"
    vpos = {v: j for j, v in enumerate(vcf_ids)}
    keep = [k for k in keys if k in vpos]
    gpos = {k: j for j, k in enumerate(keys)}
    result = (keep, [vpos[k] for k in keep], [gpos[k] for k in keep])
    return result + (mode,) if return_mode else result


def read_keyed_table(path, ids, what, id_mode="auto"):
    """Header line, then one row per sample keyed by IID (first column) or by
    FID and IID when the header starts with FID, IID. id_mode selects iid or
    fid_iid explicitly; auto refuses competing mappings to different rows.
    A single-key table always uses its sample column as written, including
    composite names if those are the prep sample names.
    Returns (header after the key columns,
    rows as lists of strings aligned to `ids`, None where absent). Repeated keys
    are an error, never resolved by keeping one row."""
    if id_mode not in ("auto", "iid", "fid_iid"):
        raise ValueError("id_mode must be auto, iid or fid_iid")
    with open(path) as f:
        head = _split(f.readline())
        rows = [_split(ln) for ln in f]
    rows = [r for r in rows if r and r[0]]
    two = len(head) >= 2 and head[0].upper() == "FID" and head[1].upper() == "IID"
    nk = 2 if two else 1
    idset = set(ids)
    if two:
        comp = ["%s_%s" % (r[0], r[1]) if len(r) > 1 else r[0] for r in rows]
        iid = [r[1] if len(r) > 1 else "" for r in rows]
        if id_mode == "auto":
            # Compare row identities, not just counts. A partial composite-key
            # hit must never silently steal another sample's IID row.
            by_iid = {k: j for j, k in enumerate(iid)}
            by_comp = {k: j for j, k in enumerate(comp)}
            ihit = any(k in idset for k in iid)
            chit = any(k in idset for k in comp)
            if ihit and chit and any(by_iid.get(k) != by_comp.get(k) for k in ids):
                raise IdentityError("%s %s: ambiguous IID versus FID_IID mapping; "
                                    "pass id_mode='iid' or 'fid_iid' explicitly, or rebuild prep "
                                    "so its sample_id_mode is recorded" % (what, path))
            keys = comp if chit else iid
        else:
            keys = comp if id_mode == "fid_iid" else iid
    else:
        keys = [r[0] for r in rows]
    check_unique(keys, "%s %s" % (what, path))
    tab = dict(zip(keys, (r[nk:] for r in rows)))
    return head[nk:], [tab.get(i) for i in ids]


def _split(ln):
    ln = ln.rstrip("\r\n")
    return ln.split("\t") if "\t" in ln else ln.split()


def read_pheno_table(path, ids, miss=(-9.0,), id_mode="auto"):
    """Read a phenotype table: header line, first column IID, one column per trait.

    Returns (Y (len(ids), k) float with NaN for missing, trait names). Values in
    `miss` and non-numeric entries become NaN, so each trait drops only its own
    missing individuals. A two-column file gives one trait.
    """
    bad = set(float(v) for v in miss)
    head, rows = read_keyed_table(path, ids, "phenotype file", id_mode=id_mode)
    k = len(head)
    if k < 1:
        raise SystemExit("phenotype file %s needs an IID column and at least one trait" % path)
    Y = np.full((len(ids), k), np.nan)
    for i, p in enumerate(rows):
        if p is None:
            continue
        for t in range(min(k, len(p))):
            try:
                x = float(p[t])
            except ValueError:
                continue
            if x not in bad and np.isfinite(x):
                Y[i, t] = x
    return Y, head


def read_pheno(path, ids, col=None, miss=(-9.0,), id_mode="auto"):
    """One trait from a phenotype table (column `col`, 1-based after IID; default 1)."""
    Y, _ = read_pheno_table(path, ids, miss=miss, id_mode=id_mode)
    return Y[:, (1 if col is None else col) - 1]


def write_matrix(path, M, ids=None, renames=None):
    np.save(path, M)
    return path
