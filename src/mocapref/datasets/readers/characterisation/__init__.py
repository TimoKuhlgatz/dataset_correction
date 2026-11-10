"""The dataset readers' common layer: ``schema.py`` (the in-memory Recording every reader produces), ``adapters/``
(one reader per dataset, reading the files where they live, ``paths.py``) and ``measures/`` (reference-side
measures; the cluster-to-sensor fit of the Trento reader uses ``measures.alignment``). No orientation estimator
runs anywhere in this package; constant rotations and offsets fitted by least squares over a whole trial are
measurements and are named as such.
"""
