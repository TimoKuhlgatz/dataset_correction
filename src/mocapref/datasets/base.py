"""The interface every dataset loader implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from ..recording import Recording

DATA_ROOT = Path('/home/data/imu-orientation/raw')


class DatasetLoader(ABC):
    """Reads one public dataset in its own deposit format and returns `Recording` objects.

    A loader only translates the format: it converts units to rad/s and m/s^2, attaches each IMU to its segment
    and reference, and states the documented conventions. It never repairs anything; every error stays in the
    data so the error checks can find it. Data are read where they lie and never copied.

    Usage:
        loader = DfjimuLoader()
        loader.trials()               # list of trial identifiers
        rec = loader.load('trial01')  # one Recording
        for rec in loader: ...        # all trials
    """

    name: str = ''  # short dataset name, the key in datasets.LOADERS
    default_root: Path = DATA_ROOT

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else Path(self.default_root)
        if not self.root.exists():
            raise FileNotFoundError(f'{self.name}: data folder {self.root} not found')

    @abstractmethod
    def trials(self) -> list[str]:
        """Identifiers of all trials in the deposit, sorted."""

    @abstractmethod
    def load(self, trial: str) -> Recording:
        """Read one trial."""

    def participant_of(self, trial: str) -> str:
        """The participant a trial belongs to, derived from the trial identifier without reading the file."""
        return trial

    def session_of(self, trial: str) -> str:
        """The session (one mounting of the sensors) a trial belongs to. Default: every trial is its own session,
        the safe choice when the documentation does not say that sensors stayed in place between trials."""
        return trial

    def sessions(self) -> dict[str, list[str]]:
        """All trials grouped by session."""
        groups: dict[str, list[str]] = {}
        for trial in self.trials():
            groups.setdefault(self.session_of(trial), []).append(trial)
        return groups

    def __iter__(self) -> Iterator[Recording]:
        for trial in self.trials():
            yield self.load(trial)

    def __len__(self):
        return len(self.trials())

    def __repr__(self):
        return f'{type(self).__name__}({self.root})'
