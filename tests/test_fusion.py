import pytest

from retrieval.fusion import rank_by_score, reciprocal_rank_fusion


def test_rrf_uses_one_over_k_plus_rank():
    fused = reciprocal_rank_fusion([[5, 7]], k=10)

    assert fused == pytest.approx({5: 1 / 11, 7: 1 / 12})


def test_rrf_rewards_agreement_between_retrievers():
    # Item 1 is second in both lists; item 0 is first in only one list.
    fused = reciprocal_rank_fusion([[0, 1], [2, 1]], k=60)

    assert fused[1] > fused[0]
    assert fused[1] > fused[2]


def test_rrf_ignores_score_scale():
    # Only ranks matter, so these two rankings are equivalent whatever the raw scores were.
    assert reciprocal_rank_fusion([[3, 1, 2]]) == reciprocal_rank_fusion([[3, 1, 2], []])


def test_rrf_weights_scale_each_ranking():
    fused = reciprocal_rank_fusion([[0], [1]], k=60, weights=[2.0, 0.0])

    assert fused[0] == pytest.approx(2 / 61)
    assert fused[1] == 0.0


def test_rrf_on_no_rankings_is_empty():
    assert reciprocal_rank_fusion([]) == {}


def test_rrf_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[0]], k=-1)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[0], [1]], weights=[1.0])


def test_rank_by_score_drops_items_with_no_signal():
    assert rank_by_score([0.0, 2.0, 1.0, 2.0]) == [1, 3, 2]
