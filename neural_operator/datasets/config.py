"""Dataset recipes and inexpensive planning, independent of the FEM backend."""
from dataclasses import asdict, dataclass
import json
from numbers import Real
from pathlib import Path

import numpy as np

from ..settings import N_DIRECTIONS, N_MEDIA


@dataclass(frozen=True)
class DatasetConfig:
    """Legacy variable-material experiment: one random frequency per medium."""
    n_media: int = N_MEDIA
    n_directions: int = N_DIRECTIONS
    h: float = .10
    n_coarse: int = 4
    k_range: tuple = (1., 2.)
    seed: int = 2026
    active_range: tuple = (3, 10)
    a_range: tuple = (.6, .9)
    n_range: tuple = (1.5, 3.)
    split_fractions: tuple = (.67, .17, .16)


@dataclass(frozen=True)
class StaticDatasetConfig:
    """One fixed medium and evenly spaced frequencies, split by frequency."""
    n_frequencies: int = 1000
    n_directions: int = 8
    h: float = .1
    n_coarse: int = 4
    k_range: tuple = (1., 2.)
    seed: int = 2026
    fixed_medium_id: int = 0
    active_range: tuple = (3, 10)
    a_range: tuple = (.6, .9)
    n_range: tuple = (1.5, 3.)
    split_fractions: tuple = (.67, .17, .16)


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _vector(name, values, length):
    try:
        values = np.asarray(values)
        if values.shape == (length,) and values.dtype.kind in 'iuf' and np.all(np.isfinite(values)):
            return values
    except (TypeError, ValueError):
        pass
    raise ValueError(f'{name} requires {length} finite numeric values')


def _interval(name, values, *, strict=False):
    lo, hi = _vector(name, values, 2)
    if not 0 < lo <= hi or (strict and lo == hi):
        order = 'lo < hi' if strict else 'lo <= hi'
        raise ValueError(f'{name} requires two positive finite values with {order}')


def config_dict(config):
    """The exact on-disk recipe; legacy variable recipes have no mode field."""
    values = asdict(config)
    if isinstance(config, StaticDatasetConfig):
        values['medium_mode'] = 'static'
    # Normalize tuples and NumPy scalars so plans and cache comparisons use JSON.
    return json.loads(json.dumps(values, default=lambda value: value.item()))


def plan_dataset(config=StaticDatasetConfig()):
    """Validate a recipe and report sizes/splits without imports or disk writes.

    Counts follow group_split exactly: train/validation are rounded down and
    test receives the remainder. A plan does not estimate mesh-dependent bytes.
    """
    if not isinstance(config, (StaticDatasetConfig, DatasetConfig)):
        raise TypeError('Expected StaticDatasetConfig or DatasetConfig')
    static = isinstance(config, StaticDatasetConfig)
    count_name = 'n_frequencies' if static else 'n_media'
    groups = getattr(config, count_name)
    _integer(count_name, groups, 3 if static else 1)
    for name in ('n_directions', 'n_coarse'):
        _integer(name, getattr(config, name), 1)
    _integer('seed', config.seed)
    if static:
        _integer('fixed_medium_id', config.fixed_medium_id)
    if not (isinstance(config.h, Real) and not isinstance(config.h, bool)
            and np.isfinite(config.h) and config.h > 0):
        raise ValueError('h must be positive and finite')
    _interval('k_range', config.k_range, strict=static)
    _interval('a_range', config.a_range)
    _interval('n_range', config.n_range)
    active = _vector('active_range', config.active_range, 2)
    for value in config.active_range:
        _integer('active_range endpoint', value)
    if not active[0] <= active[1] <= 24:
        raise ValueError('active_range requires two ordered integers in [0,24]')
    fractions = _vector('split_fractions', config.split_fractions, 3)
    if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError('Three finite nonnegative split fractions must sum to one')
    if static and len(np.unique(np.linspace(*config.k_range, groups))) != groups:
        raise ValueError('Frequency range must produce unique floating-point frequencies')
    train, validation = np.floor(fractions[:2] * groups).astype(int)
    sizes = (int(train), int(validation), int(groups-train-validation))
    directions = int(config.n_directions)
    return dict(
        medium_mode='static' if static else 'variable',
        samples=int(groups)*directions,
        media=1 if static else int(groups),
        frequencies=int(groups),
        directions=directions,
        split_strategy='frequency' if static else 'medium',
        split_groups=int(groups),
        splits=[dict(name=name, groups=count, samples=count*directions)
                for name, count in zip(('train', 'validation', 'test'), sizes)],
        k_range=[float(value) for value in config.k_range],
        mesh=dict(h=float(config.h), n_coarse=int(config.n_coarse)),
        config=config_dict(config),
    )


def load_config(path):
    """Load a config.json recipe (or its directory), accepting legacy recipes."""
    path = Path(path).expanduser()
    if path.is_dir():
        path = path/'config.json'
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError('Dataset configuration must be a JSON object')
    mode = values.pop('medium_mode', 'static' if 'n_frequencies' in values else 'variable')
    if mode not in ('static', 'variable', 'dynamic'):
        raise ValueError(f'Unknown medium_mode: {mode!r}')
    config_type = StaticDatasetConfig if mode == 'static' else DatasetConfig
    for name in ('k_range', 'active_range', 'a_range', 'n_range', 'split_fractions'):
        if name in values and isinstance(values[name], list):
            values[name] = tuple(values[name])
    try:
        config = config_type(**values)
    except TypeError as error:
        raise ValueError(f'Invalid dataset configuration: {error}') from error
    plan_dataset(config)
    return config
