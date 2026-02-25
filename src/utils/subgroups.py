def make_subgroup_indices(n_players: int, n_subgroups: int) -> list[list[int]]:
    """
    Split n_players into n_subgroups contiguous buckets.
    Players must be pre-sorted by bounding-box x-coordinate (left → right).

    n_subgroups=1 → [all players]
    n_subgroups=2 → [left_team, right_team]
    n_subgroups=4 → [left_back, left_front, right_back, right_front]
    """
    base  = n_players // n_subgroups
    extra = n_players  % n_subgroups
    indices, start = [], 0
    for m in range(n_subgroups):
        end = start + base + (1 if m < extra else 0)
        indices.append(list(range(start, end)))
        start = end
    return indices