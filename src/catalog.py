"""Binding feature metadata used by validation and the public API."""

from __future__ import annotations

FEATURE_CATALOG = {
    "temporal": {
        "time_since_source_event": {
            "description": "Elapsed time between the current event and the source node's most recent previous interaction.",
            "interpretation": "Larger values mean the source has been inactive longer; 0 means no prior source event.",
            "uses_future_information": False,
        },
        "time_since_destination_event": {
            "description": "Elapsed time between the current event and the destination node's most recent previous interaction.",
            "interpretation": "Larger values mean the destination has been inactive longer; 0 means no prior destination event.",
            "uses_future_information": False,
        },
        "source_events_recent": {
            "description": "Number of source-node interactions within the predefined recent time window.",
            "interpretation": "Recent activity count for the source before the event.",
            "uses_future_information": False,
        },
        "destination_events_recent": {
            "description": "Number of destination-node interactions within the predefined recent time window.",
            "interpretation": "Recent activity count for the destination before the event.",
            "uses_future_information": False,
        },
        "pair_events_recent": {
            "description": "Number of previous interactions between the same source-destination pair within the predefined time window.",
            "interpretation": "Recent repeated-interaction count for this ordered pair.",
            "uses_future_information": False,
        },
    },
    "graph": {
        "source_degree": {
            "description": "Number of unique destination nodes connected to the source before the current interaction.",
            "interpretation": "Historical source-side degree immediately before the event.",
            "uses_future_information": False,
        },
        "destination_degree": {
            "description": "Number of unique source nodes connected to the destination before the current interaction.",
            "interpretation": "Historical destination-side degree immediately before the event.",
            "uses_future_information": False,
        },
        "prior_edge_exists": {
            "description": "Binary indicator showing whether the source and destination have interacted previously.",
            "interpretation": "1 marks a repeated ordered edge; 0 marks a first interaction.",
            "uses_future_information": False,
        },
        "source_two_hop_reach": {
            "description": "Number of unique source-side nodes reachable through the source's previously connected destination nodes.",
            "interpretation": "Other sources sharing at least one historical destination with this source.",
            "uses_future_information": False,
        },
        "destination_two_hop_reach": {
            "description": "Number of unique destination-side nodes reachable through the destination's previously connected source nodes.",
            "interpretation": "Other destinations sharing at least one historical source with this destination.",
            "uses_future_information": False,
        },
    },
}

ALL_FEATURES = tuple(
    name for category in FEATURE_CATALOG.values() for name in category
)


def feature_category(name: str) -> str:
    for category, entries in FEATURE_CATALOG.items():
        if name in entries:
            return category
    raise ValueError(
        f"Unknown feature: {name}. Choose a feature returned by get_feature_catalog()."
    )


def validate_features(names: list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(names, str):
        raise TypeError("selected_features must be a list of feature names, not a string")
    result = list(names)
    duplicates = sorted({name for name in result if result.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate feature(s): {', '.join(duplicates)}")
    for name in result:
        feature_category(name)
    return result
