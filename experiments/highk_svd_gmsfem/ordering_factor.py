"""One bounded sparse ordering trial using the identical cached matrix."""
from pathlib import Path
from time import perf_counter
import argparse,json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import reverse_cuthill_mckee
from .reduced import nested_dissection
from experiments.reference_sno_gmsfem.dataset import _write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('ordering');args=parser.parse_args()
    root=Path(__file__).resolve().parent/'results';directory=root/'ordering_pilot'
    A=sparse.load_npz(directory/'matrix.npz');rhs=np.load(directory/'rhs.npy')
    permutation=nested_dissection(20,32,2 if args.ordering=='ND2' else 1) if args.ordering in ('ND','ND2','ND_NOPIVOT') else np.arange(len(rhs))
    if args.ordering=='RCM':permutation=reverse_cuthill_mckee(A,symmetric_mode=True)
    matrix=A[permutation][:,permutation].tocsc();start=perf_counter()
    factor=splu(matrix,permc_spec='NATURAL' if args.ordering in ('ND','ND2','RCM','ND_NOPIVOT') else args.ordering,
        **(dict(diag_pivot_thresh=0.,options={'SymmetricMode':True}) if args.ordering=='ND_NOPIVOT' else {}))
    seconds=perf_counter()-start;solution=np.empty_like(rhs);solution[permutation]=factor.solve(rhs[permutation])
    np.save(directory/f'solution_{args.ordering}.npy',solution)
    row=dict(ordering=args.ordering,factor_seconds=seconds,factor_nnz=int(factor.L.nnz+factor.U.nnz),
        residual=float(np.linalg.norm(A@solution-rhs)/np.linalg.norm(rhs)))
    _write_json(root/f'gms_ordering_{args.ordering}.json',row);print(json.dumps(row),flush=True)
if __name__=='__main__':main()
