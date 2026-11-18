"""Exact sparse reduced solves with explicit, reproducible fill ordering."""
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import splu


def nested_dissection(n_coarse,modes,separator_width=1):
    def visit(xs,ys):
        if not xs or not ys:return []
        if len(xs)==1 and len(ys)==1:return [ys[0]*(n_coarse+1)+xs[0]]
        if len(xs)>=len(ys):
            mid=len(xs)//2
            cut=min(separator_width,len(xs));lo=max(0,mid-cut//2);hi=lo+cut
            if cut==len(xs):return [y*(n_coarse+1)+x for y in ys for x in xs]
            return visit(xs[:lo],ys)+visit(xs[hi:],ys)+visit(xs[lo:hi],ys)
        mid=len(ys)//2
        cut=min(separator_width,len(ys));lo=max(0,mid-cut//2);hi=lo+cut
        if cut==len(ys):return [y*(n_coarse+1)+x for y in ys for x in xs]
        return visit(xs,ys[:lo])+visit(xs,ys[hi:])+visit(xs,ys[lo:hi])
    vertices=visit(list(range(n_coarse+1)),list(range(n_coarse+1)))
    return (np.asarray(vertices)[:,None]*modes+np.arange(modes)).ravel()


def project_matrix(operator,P):
    """Exploit real local bases without complex copies of the large P matrix."""
    PH=P.T
    real=PH@(operator.real@P)
    imaginary=operator.imag.copy();imaginary.eliminate_zeros()
    return (real+1j*(PH@(imaginary@P))).tocsc()


def apply_real(matrix,vector):
    return matrix@vector.real+1j*(matrix@vector.imag)


class ReducedSystem:
    def __init__(self,problem,space,modes,ordering='COLAMD'):
        self.P=space.prolongation(modes)
        if np.iscomplexobj(self.P.data):raise ValueError('Split projection requires a real offline basis')
        self.PH=self.P.T
        t=perf_counter();self.matrix=project_matrix(problem.operator,self.P)
        self.matrix_projection_seconds=perf_counter()-t
        self.requested_ordering=ordering;self.ordering=ordering;self.fallback_used=False
        self.permutation=(nested_dissection(space.n_coarse,modes,2 if ordering=='ND2' else 1)
                          if ordering in ('ND','ND2','ND_NOPIVOT') else None)
        self.factorization_seconds=0.
        try:self._factor(ordering)
        except RuntimeError:
            if ordering!='ND_NOPIVOT':raise
            self._fallback()

    def _factor(self,ordering):
        t=perf_counter();matrix=self.matrix
        if self.permutation is not None:
            kwargs=dict(diag_pivot_thresh=0.,options={'SymmetricMode':True}) if ordering=='ND_NOPIVOT' else {}
            self.factor=splu(matrix[self.permutation][:,self.permutation].tocsc(),permc_spec='NATURAL',**kwargs)
        else:self.factor=splu(matrix,permc_spec=ordering)
        self.factorization_seconds+=perf_counter()-t;self.ordering=ordering
        self.factor_nnz=int(self.factor.L.nnz+self.factor.U.nnz)

    def _fallback(self):
        self.factor=None;self.permutation=None;self.fallback_used=True;self._factor('COLAMD')

    def _coefficients(self,rhs):
        if self.permutation is None:return self.factor.solve(rhs)
        coefficients=np.empty_like(rhs);coefficients[self.permutation]=self.factor.solve(rhs[self.permutation])
        return coefficients

    def solve(self,problem):
        t=perf_counter();rhs=apply_real(self.PH,problem.rhs);rhs_seconds=perf_counter()-t
        t=perf_counter();coefficients=self._coefficients(rhs)
        matrix_residual=float(np.linalg.norm(self.matrix@coefficients-rhs)/max(np.linalg.norm(rhs),1e-30))
        if self.requested_ordering=='ND_NOPIVOT' and (not np.isfinite(coefficients).all() or matrix_residual>1e-10):
            self._fallback();coefficients=self._coefficients(rhs)
            matrix_residual=float(np.linalg.norm(self.matrix@coefficients-rhs)/max(np.linalg.norm(rhs),1e-30))
        solve_seconds=perf_counter()-t
        t=perf_counter();solution=np.asarray(apply_real(self.P,coefficients)).ravel();reconstruction=perf_counter()-t
        residual=float(np.linalg.norm(apply_real(self.PH,problem.operator@solution-problem.rhs))/max(np.linalg.norm(rhs),1e-30))
        if self.requested_ordering=='ND_NOPIVOT' and (not np.isfinite(residual) or residual>1e-10) and not self.fallback_used:
            self._fallback();return self.solve(problem)
        tolerance=1e-10 if self.requested_ordering=='ND_NOPIVOT' else 1e-7
        if not np.isfinite(solution).all() or not np.isfinite(residual) or residual>tolerance:
            raise RuntimeError(f'Reduced residual {residual}')
        return solution,dict(dofs_coarse=self.P.shape[1],galerkin_residual=residual,reduced_matrix_residual=matrix_residual,
            matrix_projection_seconds=self.matrix_projection_seconds,rhs_projection_seconds=rhs_seconds,
            projection_seconds=self.matrix_projection_seconds+rhs_seconds,factorization_seconds=self.factorization_seconds,
            solve_seconds=solve_seconds,reconstruction_seconds=reconstruction,sparse_ordering=self.ordering,
            requested_sparse_ordering=self.requested_ordering,ordering_fallback=self.fallback_used,factor_nnz=self.factor_nnz)
