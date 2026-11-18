"""Vectorized execution of independent FNO branches; identical learned model."""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap

from experiments.mscale_fno_100.model import MscaleFNO2d


class VectorizedMscaleFNO2d(nn.Module):
    """Stack independent weights and vmap branch calls, without weight sharing.

    The stateless meta template is deliberately excluded from module state;
    every actual branch weight lives in the registered stacked ParameterDict.
    """

    def __init__(self, *args, reference=None, **kwargs):
        super().__init__()
        original = MscaleFNO2d(*args, **kwargs) if reference is None else reference
        params, buffers = stack_module_state(original.branches)
        if buffers:
            raise ValueError('FNO branches unexpectedly contain buffers')
        self._parameter_names = tuple(params)
        self.stacked = nn.ParameterDict({key.replace('.', '_DOT_'): value for key, value in params.items()})
        self.scales = nn.Parameter(original.scales.detach().clone())
        self.gamma = nn.Parameter(original.gamma.detach().clone())
        object.__setattr__(self, '_template', deepcopy(original.branches[0]).to('meta'))

    def branch_outputs(self, x):
        parameters = {name: self.stacked[name.replace('.', '_DOT_')] for name in self._parameter_names}
        scaled = self.scales[:, None, None, None, None] * x[None]
        return vmap(lambda p, a: functional_call(self._template, p, (a,)), in_dims=(0, 0))(parameters, scaled)

    def forward(self, x):
        return (self.gamma[:, None, None, None, None] * self.branch_outputs(x)).sum(0)
