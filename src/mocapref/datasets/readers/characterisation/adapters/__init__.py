"""The dataset readers ("adapters"), one module per dataset, imported by name and on demand. `adapter(name)` returns
the reader registered under `name` (a key of MODULES).
"""
from __future__ import annotations

import importlib

from ..schema import Adapter

MODULES = {
    'bonfiglio-arm': 'bonfiglio',
    'bonfiglio-arm-deposited': 'bonfiglio',  # the same reader with the deposit's own pairing
    'camargo-treadmill': 'camargo',
    'compliant-terrain': 'compliant_terrain',
    'fullbody-17imu': 'fullbody17',
    'grimmer-stairs': 'grimmer_stairs',
    'grouvel-geneva': 'grouvel',
    'hip-rom-y': 'hip_rom_y',
    'is2b': 'is2b',
    'openaxes': 'openaxes',
    'opensense-s2': 'opensense',
    'reemc': 'reemc',
    'repoimu-pendulum': 'repoimu',
    'ultramocap-upper-limb': 'ultramocap',
    'wmcg-ucf': 'wmcg',
}


def adapter(name: str) -> Adapter:
    """The reader registered under `name` (a key of MODULES)."""
    module = importlib.import_module(f'{__name__}.{MODULES[name]}')
    found = [module.ADAPTER] + list(getattr(module, 'EXTRA_ADAPTERS', []))
    return next(a for a in found if a.name == name)


__all__ = ['MODULES', 'adapter']
