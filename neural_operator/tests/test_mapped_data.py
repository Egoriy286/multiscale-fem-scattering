"""Memory-mapped compressed datasets preserve exact samples and fingerprints."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from neural_operator.data import load_dataset, update_array_digest, write_dataset_info


@pytest.mark.parametrize('values',[
    np.arange(15,dtype=np.float64),
    np.arange(60,dtype=np.complex128).reshape(12,5),
    np.arange(60,dtype=np.int32).reshape(12,5)[::2,::-1],
    np.arange(60,dtype=np.float64).reshape(12,5).T,
    np.broadcast_to(np.array([[1.,2.,3.]]),(17,3)),
    np.empty((0,5),dtype=np.complex128),
    np.empty((4,0),dtype=np.float64),
    np.array(2.5),
])
def test_array_digest_exact_c_order(values):
    expected = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
    actual = update_array_digest(hashlib.sha256(),values,batch_rows=3).hexdigest()
    assert actual == expected


def save_tiny(directory, shift=0.):
    directory.mkdir(parents=True,exist_ok=True)
    archive = dict(A=np.array([[1.,.8,1.]]),n=np.array([[1.,2.,1.]]),A_cells=np.array([[.8]]),n_cells=np.array([[2.]]),u_scat=np.arange(18).reshape(6,3).astype(complex)*(1+2j)+shift,theta=np.arange(6,dtype=float),k0=np.repeat([1.,1.5,2.],2),medium_id=np.zeros(6,dtype=np.int32),frequency_id=np.repeat(np.arange(3,dtype=np.int32),2),split=np.repeat(np.arange(3,dtype=np.int8),2),split_strategy=np.asarray('frequency'))
    np.savez_compressed(directory/'fem_dataset.npz',**archive)
    np.savez_compressed(directory/'geometry.npz',dof_coords=np.array([[0.,0.],[1.,0.],[0.,1.]]),triangles=np.array([[0,1,2]]),cell_tags=np.array([2]))
    (directory/'etal.msh').write_text('tiny metadata fixture')
    return archive


def cache_paths(dataset):
    cache = Path(dataset.u_scat.filename)
    return cache,cache.with_name('u_scat.cache.json')


def test_compact_mmap_roundtrip_hash_and_cache_hit(tmp_path,monkeypatch):
    import neural_operator.data as module
    original = save_tiny(tmp_path)
    mapped = load_dataset(tmp_path,mmap=True)
    ordinary = load_dataset(tmp_path)
    assert isinstance(mapped.u_scat,np.memmap)
    assert not mapped.u_scat.flags.writeable
    for name in ('A','n','A_cells','n_cells','u_scat','theta','k0','medium_id','frequency_id','split'):
        assert np.array_equal(getattr(mapped,name),getattr(ordinary,name))
    assert mapped.A.shape == (6,3) and mapped.A.strides[0] == 0
    assert original['A'].shape == (1,3)
    info = write_dataset_info(mapped)
    for name,actual in info['field_sha256'].items():
        expected = hashlib.sha256(np.ascontiguousarray(getattr(mapped,name)).tobytes()).hexdigest()
        assert actual == expected
    cache,metadata = cache_paths(mapped)
    before = (cache.stat().st_mtime_ns,metadata.stat().st_mtime_ns)
    signature = json.loads(metadata.read_text())
    assert signature['source_size'] == (tmp_path/'fem_dataset.npz').stat().st_size
    def no_extract(*args,**kwargs):
        raise AssertionError('Valid cache must not extract the compressed field again')
    monkeypatch.setattr(module.shutil,'copyfileobj',no_extract)
    cached = load_dataset(tmp_path/'fem_dataset.npz',mmap=True)
    assert np.array_equal(cached.u_scat,original['u_scat'])
    assert before == (cache.stat().st_mtime_ns,metadata.stat().st_mtime_ns)
    assert not list(cache.parent.glob('*.tmp'))


def test_archive_change_invalidates_mmap_cache(tmp_path):
    save_tiny(tmp_path)
    first = load_dataset(tmp_path,mmap=True)
    _,metadata = cache_paths(first)
    before = json.loads(metadata.read_text())
    changed = save_tiny(tmp_path,shift=19.)
    updated = load_dataset(tmp_path,mmap=True)
    after = json.loads(metadata.read_text())
    assert before != after
    assert before['member_crc'] != after['member_crc']
    assert np.array_equal(updated.u_scat,changed['u_scat'])
    assert not list(Path(updated.u_scat.filename).parent.glob('*.tmp'))


def test_failed_cache_extraction_leaves_no_temporary_files(tmp_path,monkeypatch):
    import neural_operator.data as module
    save_tiny(tmp_path)
    def fail_extract(source,target,**kwargs):
        target.write(source.read(20))
        raise OSError('interrupted extraction')
    monkeypatch.setattr(module.shutil,'copyfileobj',fail_extract)
    with pytest.raises(OSError,match='interrupted'):
        load_dataset(tmp_path,mmap=True)
    cache_dir = tmp_path/'.array_cache'/'fem_dataset'
    assert not list(cache_dir.glob('*.tmp'))
    assert not (cache_dir/'u_scat.npy').exists()


def test_mmap_never_decompresses_field_through_numpy(tmp_path,monkeypatch):
    original = save_tiny(tmp_path)
    with np.load(tmp_path/'fem_dataset.npz',allow_pickle=False) as archive:
        archive_type = type(archive)
    getitem = archive_type.__getitem__
    def guarded_getitem(self,key):
        if key in ('u_scat','u_scat.npy'):
            raise AssertionError('mmap load must not materialize archive u_scat')
        return getitem(self,key)
    monkeypatch.setattr(archive_type,'__getitem__',guarded_getitem)
    mapped = load_dataset(tmp_path,mmap=True)
    assert isinstance(mapped.u_scat,np.memmap)
    assert np.array_equal(mapped.u_scat,original['u_scat'])
