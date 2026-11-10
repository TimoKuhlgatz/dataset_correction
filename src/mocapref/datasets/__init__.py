"""Dataset loaders: one loader per dataset, each reading one deposit format and returning `Recording` objects.

    from mocapref.datasets import load_dataset
    loader = load_dataset('dfjimu')
    rec = loader.load(loader.trials()[0])

Short name          Dataset                                                Loader class
tuk6min             TUK six-minute walk                                    TukLoader (tuk.py)
opensense           OpenSense (Al Borno 2022)                              OpensenseLoader
fullbody17          Hankyong full body, 17 Xsens IMUs                      Fullbody17Loader
camargo             Camargo lower limb (Georgia Tech), treadmill part      CamargoLoader
wmcg                Central Florida walking (WMCG)                         WmcgLoader
compliant_terrain   Delaware compliant terrain                             CompliantTerrainLoader
hip_rom_y           Yanshan hip range of motion (Hip-ROM-Y)                HipRomYLoader
is2b                IS2B cadaver knee                                      Is2BLoader
grouvel             Geneva gait (Grouvel 2022)                             GrouvelLoader
bonfiglio           Trento upper limb (Bonfiglio 2024)                     BonfiglioLoader
grimmer_stairs      Darmstadt stairs (Grimmer)                             GrimmerStairsLoader
ultramocap          Central Florida upper limb (ULTRA-MoCap)               UltramocapLoader
diodem              DIODEM mechanical chains (Bachhuber 2025)              DiodemLoader (diodem.py)
dfjimu              dfjimu ABB IRB120 robot arm                            DfjimuLoader (dfjimu.py)
mfimt_d01           mfimt 3D-printed chain, recording d01                  MfimtLoader (mfimt.py)
reemc               REEM-C humanoid robot                                  ReemcLoader
openaxes            OpenAXES UR5 robot arm                                 OpenaxesLoader
repoimu             RepoIMU triple pendulum                                RepoimuLoader

Helper readers (not datasets of their own):
grouvel_raw         Geneva gait, IMU rebuilt from the raw session logs     GrouvelRawLoader (grouvel_raw.py)
bonfiglio_deposited Trento upper limb with the deposit's own pairing       BonfiglioDepositedLoader

Loaders without a file named are built in _bridge.py on the dataset readers in `readers/`. PAPER_DATASETS lists the
18 datasets in the order above.
"""
from .base import DatasetLoader
from .dfjimu import DfjimuLoader
from .grouvel_raw import GrouvelRawLoader
from .diodem import DiodemLoader
from ._bridge import READER_LOADERS, ReaderLoader
from .mfimt import MfimtLoader
from .tuk import TukLoader

LOADERS: dict[str, type[DatasetLoader]] = {
    loader.name: loader for loader in (DfjimuLoader, MfimtLoader, TukLoader, DiodemLoader,
                                       *READER_LOADERS, GrouvelRawLoader)
}

HELPER_READERS: dict[str, str] = {
    'grouvel_raw': 'Geneva gait with the IMU streams rebuilt from the raw session logs (mis-assembled streams, '
                   'error 10)',
    'bonfiglio_deposited': "Trento upper limb with the deposit's own trial pairing, read for the trial-pairing "
                           'check (error 8)',
}
PAPER_DATASETS: list[str] = ['tuk6min', 'opensense', 'fullbody17', 'camargo', 'wmcg', 'compliant_terrain',
                             'hip_rom_y', 'is2b', 'grouvel', 'bonfiglio', 'grimmer_stairs', 'ultramocap', 'diodem',
                             'dfjimu', 'mfimt_d01', 'reemc', 'openaxes', 'repoimu']
assert set(LOADERS) == set(PAPER_DATASETS) | set(HELPER_READERS)


def load_dataset(name: str, **options) -> DatasetLoader:
    """The loader for a dataset by its short name; options go to the loader (e.g. root=..., imu='nonrigid')."""
    if name not in LOADERS:
        raise KeyError(f'No loader for {name!r}. Available: {", ".join(LOADERS)}')
    return LOADERS[name](**options)


__all__ = ['DatasetLoader', 'LOADERS', 'PAPER_DATASETS', 'HELPER_READERS', 'load_dataset',
           'DfjimuLoader', 'GrouvelRawLoader', 'MfimtLoader', 'TukLoader', 'DiodemLoader', 'ReaderLoader']
