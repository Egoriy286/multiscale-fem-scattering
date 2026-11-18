import numpy as np
import pytest
from experiments.highk_svd_gmsfem.dataset import make_queries


def test_disjoint_stratified_queries_and_historical_exclusion():
    old=np.linspace(.1,4,1200)
    q=make_queries(historical_frequencies=old)
    assert len(q['k0'])==2400
    assert len(np.unique(q['k0']))==2400
    assert not set(q['k0']) & set(old)
    assert [sum(q['split']==i) for i in (0,1,2)]==[2000,200,200]
    test=q['k0'][q['split']==2]
    assert test[0]==.05 and test[-1]==20
    train=q['k0'][q['split']==0]
    for count in (400,600,800,1000,2000):
        ordered=np.sort(train[:count])
        assert np.max(np.diff(ordered)) < 19.95/300
    assert np.array_equal(np.floor((train[:400]-.05)/19.95*400),np.arange(400))
    other=make_queries(historical_frequencies=old)
    assert all(np.array_equal(q[k],other[k]) for k in q)


@pytest.mark.parametrize('kw',[dict(k_min=0),dict(n_test=2),dict(n_train=True),dict(k_max=np.inf)])
def test_reject_degenerate_query_plans(kw):
    with pytest.raises(ValueError):make_queries(**kw)
