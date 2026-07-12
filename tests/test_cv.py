"""CPCV: split counts, purge/embargo proofs, path reconstruction, canary."""
import numpy as np
import pytest

from spydt.cv import Split, assert_no_leakage, cpcv_splits, make_groups, paths_from_splits


def test_split_count_and_sizes():
    splits = cpcv_splits(600, n_groups=6, k_test=2, purge_days=1, embargo_days=5)
    assert len(splits) == 15  # C(6,2)
    for s in splits:
        assert len(s.test_idx) == 200  # 2 groups of 100
        assert len(np.intersect1d(s.train_idx, s.test_idx)) == 0


def test_purge_and_embargo_boundaries():
    splits = cpcv_splits(600, n_groups=6, k_test=2, purge_days=2, embargo_days=5)
    for s in splits:
        t = np.sort(s.test_idx)
        blocks = np.split(t, np.where(np.diff(t) > 1)[0] + 1)
        for b in blocks:
            # no train day within purge BEFORE the block
            assert not ((s.train_idx >= b.min() - 2) & (s.train_idx < b.min())).any()
            # no train day within purge+embargo AFTER the block
            assert not ((s.train_idx > b.max()) & (s.train_idx <= b.max() + 7)).any()


def test_every_group_tested_equally():
    n_groups, k = 6, 2
    splits = cpcv_splits(600, n_groups=n_groups, k_test=k, purge_days=1, embargo_days=3)
    counts = {g: 0 for g in range(n_groups)}
    for s in splits:
        for g in s.test_groups:
            counts[g] += 1
    assert all(v == 5 for v in counts.values())  # C(5,1)


def test_paths_cover_all_groups_disjointly():
    paths = paths_from_splits(6, 2)
    assert len(paths) == 5
    for p in paths:
        assert sorted(p.keys()) == list(range(6))
    # each (group, split) assignment used exactly once across paths
    used = set()
    for p in paths:
        for g, sid in p.items():
            assert (g, sid) not in used
            used.add((g, sid))


def test_canary_verifies_embargo_too():
    splits = cpcv_splits(600, n_groups=6, k_test=2, purge_days=1, embargo_days=3)
    assert_no_leakage(splits, purge_days=1, embargo_days=3)  # clean passes
    bad = Split(0, splits[0].test_groups,
                np.concatenate([splits[0].train_idx,
                                [int(np.sort(splits[0].test_idx)[-1]) + 2]]),
                splits[0].test_idx)
    # a train day 2 days after a test block violates purge(1)+embargo(3)
    with pytest.raises(AssertionError):
        assert_no_leakage([bad], purge_days=1, embargo_days=3)


def test_canary_catches_injected_future_leak():
    splits = cpcv_splits(600, n_groups=6, k_test=2, purge_days=1, embargo_days=3)
    assert_no_leakage(splits, purge_days=1)  # clean passes
    bad = splits[0]
    corrupted = Split(
        bad.split_id,
        bad.test_groups,
        np.concatenate([bad.train_idx, bad.test_idx[:1]]),  # inject a test day
        bad.test_idx,
    )
    with pytest.raises(AssertionError):
        assert_no_leakage([corrupted], purge_days=1)


def test_groups_are_contiguous_and_balanced():
    g = make_groups(1000, 10)
    assert (np.diff(g) >= 0).all()
    _, counts = np.unique(g, return_counts=True)
    assert counts.min() >= 99 and counts.max() <= 101
