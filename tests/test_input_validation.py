"""Input-validation and identifier-alignment regressions, using observable outputs."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hapdc import cli, core, io, ms, scan


def test_identity_collision_requires_explicit_mode(tmp_path):
    p = tmp_path / 'table.tsv'
    p.write_text('FID\tIID\ty\nF\tA\t10\nH\tF_A\t20\nG\tB\t30\n')
    for reader in (io.read_pheno_table, cli._read_covar):
        with pytest.raises(io.IdentityError, match='ambiguous IID'):
            reader(str(p), ['F_A', 'B'])
        y, _ = reader(str(p), ['F_A', 'B'], id_mode='iid')
        assert y[:, 0].tolist() == [20., 30.]
        y, _ = reader(str(p), ['F_A', 'G_B'], id_mode='fid_iid')
        assert y[:, 0].tolist() == [10., 30.]


def test_prep_persists_iid_mode_and_ms_reuses_it(tmp_path):
    # All target IDs exist as IIDs. The first target also collides with another
    # row's composite key; auto-GRM prep must still use the actual VCF IDs.
    rng = np.random.default_rng(17)
    ids = ['F_A', 'B'] + ['s%d' % i for i in range(28)]
    h = rng.binomial(1, .4, (60, 24))
    v = tmp_path / 'g.vcf'
    with v.open('w') as f:
        f.write('##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(ids) + '\n')
        for j in range(h.shape[1]):
            f.write('1\t%d\tm%d\tA\tC\t.\tPASS\t.\tGT\t' % (j+1, j) + '\t'.join('%d|%d' % (h[2*i,j],h[2*i+1,j]) for i in range(30)) + '\n')
    ph = tmp_path / 'ph.tsv'
    ph.write_text('FID\tIID\ty\nF\tA\t999\nH\tF_A\t0\nG\tB\t1\n' + ''.join('G\ts%d\t%d\n' % (i,i+2) for i in range(28)))
    cv = tmp_path / 'cv.tsv'
    cv.write_text('FID\tIID\tc\nF\tA\t999\nH\tF_A\t0\nG\tB\t1\n' + ''.join('G\ts%d\t%d\n' % (i,i%3) for i in range(28)))
    prep = tmp_path / 'prep'
    assert cli.main(['prep', str(v), 'auto', str(ph), str(prep), '--grm-thin', '1']) == 0
    assert json.loads((prep/'meta.json').read_text())['sample_id_mode'] == 'iid'
    assert np.array_equal(np.load(prep/'traits.npy')[:,0], np.arange(30))
    with patch.object(cli, '_read_covar', wraps=cli._read_covar) as reader:
        assert cli.main(['ms', str(prep), str(tmp_path/'out'), '0', '--covar', str(cv), '--transform', 'none']) == 0
        assert reader.call_args.kwargs == {'id_mode': 'iid'}
    cli._MS_CTX.clear()


def test_composite_identity_mode_survives_grm_mapping(tmp_path):
    ids, vr, gr, mode = io.match_ids(['F2_a','F1_a','F3_b'], ['a','a','b'], ['F1','F2','F3'], return_mode=True)
    assert mode == 'fid_iid' and ids == ['F1_a','F2_a','F3_b'] and vr == [1,0,2] and gr == [0,1,2]
    p = tmp_path/'p.tsv'
    p.write_text('FID\tIID\ty\nF1\ta\t1\nF2\ta\t2\nF3\tb\t3\n')
    y, _ = io.read_pheno_table(p, ids, id_mode=mode)
    assert y[:,0].tolist() == [1.,2.,3.]


HEADER = 'chrom\tstart_bp\tend_bp\tn_markers\tP\tlead_scale\tlead_unit_start\tlead_unit_end\tP_nonadditive\tsig_bonf\n'


def table(tmp_path, name, p='0.000123456789', status='completed'):
    t = tmp_path / (name+'.anchors.tsv')
    t.write_text(HEADER+'1\t1\t10\t16\t%s\tSNP\t1\t1\t0.5\t0\n' % p)
    s = tmp_path / (name+'.summary.json')
    s.write_text(json.dumps(dict(status=status,trait='y',reason='known failure')))
    return t, s


def test_failed_sidecar_preserves_both_existing_outputs(tmp_path):
    t, _ = table(tmp_path, 'shard', status='FAILED')
    out, summ = tmp_path/'merged.anchors.tsv', tmp_path/'merged.summary.json'
    out.write_bytes(b'OLD RESULT\n'); summ.write_bytes(b'{"status":"completed","n_anchors":99}\n')
    before = [p.read_bytes() for p in (out, summ)]
    with pytest.raises(SystemExit, match='FAILED'):
        cli.main(['merge', str(out), str(t)])
    assert [p.read_bytes() for p in (out,summ)] == before


@pytest.mark.parametrize('p', ['-0.1','1.01','nan','inf'])
def test_invalid_merge_p_never_mutates_output(tmp_path, p):
    t, _ = table(tmp_path,'shard',p=p)
    out, summ = tmp_path/'merged.anchors.tsv', tmp_path/'merged.summary.json'
    out.write_bytes(b'OLD'); summ.write_bytes(b'OLD SUMMARY')
    with pytest.raises(SystemExit):cli.main(['merge', str(out), str(t)])
    assert out.read_bytes() == b'OLD' and summ.read_bytes() == b'OLD SUMMARY'


def test_merge_refuses_overwriting_input_sidecar(tmp_path):
    t,s = table(tmp_path, 'same')
    before = s.read_bytes()
    # Different table suffix, same sidecar stem.
    renamed=tmp_path/'same.tsv';renamed.write_bytes(t.read_bytes())
    with pytest.raises(SystemExit, match='input table or summary'):
        cli.main(['merge', str(tmp_path/'same'), str(renamed)])
    assert s.read_bytes() == before and t.read_text().startswith(HEADER)


def test_merge_stages_both_files_before_replacing_either(tmp_path):
    import tempfile
    t,_=table(tmp_path,'shard')
    out,summ=tmp_path/'merged.anchors.tsv',tmp_path/'merged.summary.json'
    out.write_bytes(b'OLD');summ.write_bytes(b'OLD SUMMARY')
    original=tempfile.NamedTemporaryFile
    calls=[]
    def fail_second(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:raise OSError('simulated full disk')
        return original(*args,**kwargs)
    with patch.object(tempfile,'NamedTemporaryFile',side_effect=fail_second):
        with pytest.raises(OSError,match='simulated full disk'):
            cli.main(['merge',str(out),str(t)])
    assert out.read_bytes()==b'OLD' and summ.read_bytes()==b'OLD SUMMARY'


def test_saturated_haplotype_model_keeps_valid_snp_fallback(tmp_path):
    n,m=10,5
    h=((np.arange(2*n)[:,None] >> np.arange(m)) & 1).astype(np.int8)
    y=np.random.default_rng(20261122).normal(size=n)
    meta=[('1','m%d'%j,j+1,'A','C') for j in range(m)]
    sc=scan.Scan(h,meta,np.eye(n),np.ones(n),y)
    d=core.window_design(h,.05,5)
    assert scan._colbasis(np.column_stack([sc.one,d['dos']*sc.w[:,None]])).shape[1] == n
    # Underflow of the deliberately tiny genotype-only norm floor is benign;
    # division by zero or invalid F arithmetic is the regression of interest.
    with np.errstate(divide='raise', invalid='raise', over='raise', under='ignore'):
        r=sc.run(blocks=[(0,m)],verbose=False)
        a=ms.ms_scan(sc,[(0,m,[(0,m)])],variants=('D',))
    assert not r['V']['A_joint'][0] and r['P']['A_joint'][0,0] == 1
    assert r['V']['SACAT'][0] and np.isfinite(a['P_MS']['D']).all()
    p=tmp_path/'prep';p.mkdir()
    for name,value in [('hap',h),('traits',y[:,None]),('K_keep',np.eye(n)),('keep',np.arange(n))]:np.save(p/(name+'.npy'),value)
    (p/'meta.json').write_text(json.dumps(dict(ids=['s%d'%i for i in range(n)],traits=['y'],sample_id_mode='iid')))
    (p/'markers.json').write_text(json.dumps([meta]))
    assert cli.main(['ms',str(p),str(tmp_path/'out'),'0','--transform','none']) == 0
    assert (tmp_path/'out/y.anchors.tsv').exists()
    assert json.loads((tmp_path/'out/y.summary.json').read_text())['status'] == 'completed'
    cli._MS_CTX.clear()
