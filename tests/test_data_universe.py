

def test_membership_csv_from_raw_strings_resolves_point_in_time(live_cfg, tmp_path):
    """A CSV read straight from disk carries strings, not Timestamps.

    Comparing those to a Timestamp raises, which would break the one mechanism
    that can remove survivorship bias -- and it would break at the moment
    someone finally transcribed the reconstitution circulars.
    """
    import datetime as dt

    import pandas as pd

    from prosignal.data.store import DataStore
    from prosignal.data.universe import UniverseResolver

    store = DataStore(live_cfg.paths.curated, live_cfg.paths.snapshots)
    csv = pd.DataFrame(
        [
            {"index_name": "NIFTY 200", "symbol": "STAYS",
             "effective_from": "2019-01-01", "effective_to": ""},
            {"index_name": "NIFTY 200", "symbol": "LEAVES",
             "effective_from": "2019-01-01", "effective_to": "2023-06-30"},
        ]
    )
    resolver = UniverseResolver(store, live_cfg.params)

    before = resolver.resolve("NIFTY 200", dt.date(2022, 6, 1), membership_csv=csv)
    after = resolver.resolve("NIFTY 200", dt.date(2024, 6, 1), membership_csv=csv)

    assert "LEAVES" in set(before.symbols)
    assert "LEAVES" not in set(after.symbols)
    assert "STAYS" in set(after.symbols)
    # Effective-dated membership is genuinely point-in-time, so no risk flag.
    assert before.survivorship_risk is False
