from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from src.feature_engineering import compute_causal_features


def graph():
    return pd.DataFrame(
        {
            "u": [1, 2, 1, 1, 2, 3],
            "i": [10, 10, 11, 10, 11, 10],
            "ts": [1.0, 2.0, 3.0, 4.0, 4.0, 8.0],
        }
    )


def test_temporal_features_are_manual_and_equal_time_strict():
    values = compute_causal_features(graph(), 3.0).values
    assert values["time_since_source_event"].tolist() == [0, 0, 2, 1, 2, 0]
    assert values["time_since_destination_event"].tolist() == [0, 1, 0, 2, 1, 4]
    assert values["source_events_recent"].tolist() == [0, 0, 1, 2, 1, 0]
    assert values["destination_events_recent"].tolist() == [0, 1, 0, 2, 1, 0]
    assert values["pair_events_recent"].tolist() == [0, 0, 0, 1, 0, 0]


def test_graph_features_are_manual_and_equal_time_strict():
    values = compute_causal_features(graph(), 3.0).values
    assert values["source_degree"].tolist() == [0, 0, 1, 2, 1, 0]
    assert values["destination_degree"].tolist() == [0, 1, 0, 2, 1, 2]
    assert values["prior_edge_exists"].tolist() == [0, 0, 0, 1, 0, 0]
    assert values["source_two_hop_reach"].tolist() == [0, 0, 1, 1, 1, 0]
    assert values["destination_two_hop_reach"].tolist() == [0, 0, 0, 1, 1, 1]


def test_future_changes_and_deletions_do_not_change_earlier_features():
    original = graph()
    expected = compute_causal_features(original, 3.0).values.iloc[:4]
    changed = original.copy()
    changed.loc[4:, ["u", "i", "ts"]] = [[99, 199, 20.0], [98, 198, 21.0]]
    actual_changed = compute_causal_features(changed, 3.0).values.iloc[:4]
    actual_deleted = compute_causal_features(original.iloc[:4], 3.0).values
    pdt.assert_frame_equal(expected, actual_changed)
    pdt.assert_frame_equal(expected, actual_deleted)

