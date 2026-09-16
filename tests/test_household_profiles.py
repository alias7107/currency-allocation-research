from src.household.profiles import P1_LOCAL_CONTROL, P4_MULTI_CURRENCY, strategy_catalog


def test_shares_sum_to_one():
    for p in (P1_LOCAL_CONTROL, P4_MULTI_CURRENCY):
        assert abs(p.qI_kzt + p.qI_usd + p.qI_eur - 1) < 1e-12
        assert abs(p.qE_kzt + p.qE_usd + p.qE_eur - 1) < 1e-12


def test_p1_deduplicates_matched_strategies():
    names = [n for n, _ in strategy_catalog(P1_LOCAL_CONTROL)]
    assert names.count("B1_all_kzt") == 1
    assert "B5_expense_matched" not in names
    assert "B6_income_matched" not in names


def test_p4_keeps_distinct_matched_strategies():
    names = [n for n, _ in strategy_catalog(P4_MULTI_CURRENCY)]
    assert "B5_expense_matched" in names
    assert "B6_income_matched" in names
    assert len(names) == 6
