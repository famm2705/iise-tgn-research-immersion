# Approved Feature Catalog

## Temporal

- `time_since_source_event`: elapsed time since the source node's most recent
  earlier interaction. It is `0` when no earlier interaction exists.
- `time_since_destination_event`: elapsed time since the destination node's
  most recent earlier interaction. It is `0` when no earlier interaction exists.
- `source_events_recent`: number of earlier source interactions in the centrally
  configured recent-time window.
- `destination_events_recent`: number of earlier destination interactions in
  the same window.
- `pair_events_recent`: number of earlier interactions of the same ordered
  source-destination pair in the same window.

## Graph

- `source_degree`: number of unique destinations connected to the source earlier.
- `destination_degree`: number of unique sources connected to the destination earlier.
- `prior_edge_exists`: `1` if the ordered pair interacted earlier, otherwise `0`.
- `source_two_hop_reach`: number of unique other sources that share at least one
  earlier destination neighbor with the source.
- `destination_two_hop_reach`: number of unique other destinations that share at
  least one earlier source neighbor with the destination.

All definitions use only timestamps strictly earlier than the current event.
