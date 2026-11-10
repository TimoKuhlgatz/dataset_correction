"""Dataset readers, one per dataset: each turns a deposit's files into the Recording of
``characterisation/schema.py``; ``mocapref.datasets._bridge`` translates that into ``mocapref.Recording``.

Layout: ``characterisation/adapters/`` (one reader per dataset), ``characterisation/measures/`` (reference-side
measures used by the readers), ``datasets/opensense/`` (the OpenSense reference rebuilt from the raw markers) and
``scripts/`` (the Geneva IMU rebuild from the raw session logs).
"""
