"""A tiny in-memory loader of synthetic recordings, for the dataset-level tests of the error handlers."""
from __future__ import annotations


class MemoryLoader:
    """trials() / load() / participant_of() / session_of() over a dict {trial: Recording}; load returns a copy."""

    def __init__(self, recordings: dict, participant: dict | None = None, session: dict | None = None):
        self.recordings = recordings
        self.participant = participant or {}
        self.session = session or {}
        self.name = 'synthetic'

    def trials(self):
        return list(self.recordings)

    def load(self, trial):
        return self.recordings[trial].copy()

    def participant_of(self, trial):
        return self.participant.get(trial, 'P1')

    def session_of(self, trial):
        return self.session.get(trial, trial)

    def __iter__(self):
        for t in self.trials():
            yield self.load(t)
