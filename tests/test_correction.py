"""The order of correction: every handled error 1-22 has a module whose PARTS entry carries its own number, and the
readers switch where the order says (Trento re-paired after error 8, Geneva rebuilt from error 8's check on)."""
from mocapref import correction as C


def test_every_error_has_its_parts_under_its_own_number():
    assert sorted(C.ORDER) == list(range(1, 23))
    for n in C.ORDER:
        parts = C.parts(n)
        assert parts['spec'].number == n
        assert parts['spec'].kind in ('fix', 'mask', 'report')
        assert callable(parts['check_dataset']) and callable(parts['prevalence'])


def test_readers_switch_where_the_order_says():
    assert C.base_reader('bonfiglio', 8) == 'bonfiglio_deposited'
    assert C.base_reader('bonfiglio', 9) == 'bonfiglio'
    assert C.base_reader('grouvel', 7) == 'grouvel'
    assert C.base_reader('grouvel', 8) == 'grouvel_raw'
    assert C.base_reader('camargo', 1) == 'camargo'
