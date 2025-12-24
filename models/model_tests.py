#!/usr/bin/env python3
"""
Tests for the Commonality Model.

Mostly behavior tests, with a few sanity checks on model setup.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model import (
    CommonalityModel,
    load_factor_loadings,
    load_question_means,
)


class TestModelSetup:
    """Sanity checks that model is configured correctly."""

    def test_k_stored(self):
        """Model should store k."""
        assert CommonalityModel(k=4).k == 4
        assert CommonalityModel(k=0).k == 0

    def test_epsilon_clipped(self):
        """Epsilon should be clipped to [0, 1]."""
        assert CommonalityModel(epsilon=1.5).epsilon == 1.0
        assert CommonalityModel(epsilon=-0.5).epsilon == 0.0

    def test_repr(self):
        """Repr should show key parameters."""
        assert 'k=4' in repr(CommonalityModel(k=4))
        assert 'Bayesian' in repr(CommonalityModel(k=4, lambda_mix=0.0))
        assert 'SimilarityProjection' in repr(CommonalityModel(lambda_mix=1.0))
        assert 'λ=0.50' in repr(CommonalityModel(k=4, lambda_mix=0.5))


class TestPredictions:
    """Test prediction behavior."""

    def test_output_shape(self):
        """Should return 35 predictions."""
        model = CommonalityModel(k=4)
        preds = model.predict(0, 3.0, np.random.uniform(1, 5, 35))
        assert preds.shape == (35,)

    def test_output_range(self):
        """Predictions should be probabilities in [0, 1]."""
        model = CommonalityModel(k=4)
        preds = model.predict(0, 3.0, np.random.uniform(1, 5, 35))
        assert np.all(preds >= 0) and np.all(preds <= 1)

    def test_deterministic(self):
        """Same inputs should give same outputs."""
        model = CommonalityModel(k=4)
        r_self = np.random.uniform(1, 5, 35)
        preds1 = model.predict(0, 3.0, r_self)
        preds2 = model.predict(0, 3.0, r_self)
        assert np.allclose(preds1, preds2)

    def test_epsilon_compresses_to_half(self):
        """Higher epsilon should push predictions toward 0.5."""
        r_self = np.random.uniform(1, 5, 35)
        preds_low = CommonalityModel(k=4, epsilon=0.0).predict(0, 3.0, r_self)
        preds_high = CommonalityModel(k=4, epsilon=0.8).predict(0, 3.0, r_self)

        dist_low = np.abs(preds_low - 0.5).mean()
        dist_high = np.abs(preds_high - 0.5).mean()
        assert dist_high < dist_low


class TestModelBehavior:
    """Test that different configurations behave differently."""

    def test_lambda_affects_predictions(self):
        """λ=0 (Bayesian) vs λ=1 (SimilarityProjection) should give different predictions."""
        r_self = np.array([1.0] * 17 + [5.0] * 18)  # Polarized responses
        preds_bayes = CommonalityModel(k=5, lambda_mix=0.0).predict(0, 3.0, r_self)
        preds_proj = CommonalityModel(k=5, lambda_mix=1.0).predict(0, 3.0, r_self)
        assert not np.allclose(preds_bayes, preds_proj)

    def test_lambda_clipped(self):
        """λ should be clipped to [0, 1]."""
        assert CommonalityModel(lambda_mix=1.5).lambda_mix == 1.0
        assert CommonalityModel(lambda_mix=-0.5).lambda_mix == 0.0

    def test_lambda_mixture_interpolates(self):
        """λ=0.5 should give predictions between λ=0 and λ=1."""
        r_self = np.random.uniform(1, 5, 35)
        preds_0 = CommonalityModel(k=5, lambda_mix=0.0, epsilon=0.0).predict(0, 3.0, r_self)
        preds_1 = CommonalityModel(k=5, lambda_mix=1.0, epsilon=0.0).predict(0, 3.0, r_self)
        preds_half = CommonalityModel(k=5, lambda_mix=0.5, epsilon=0.0).predict(0, 3.0, r_self)
        # Mixture should be close to average
        expected = 0.5 * preds_0 + 0.5 * preds_1
        assert np.allclose(preds_half, expected, atol=0.01)

    def test_similarity_projection_uses_self_structure(self):
        """Similarity projection (λ=1) should produce gradients based on self-response similarity."""
        # Self responds identically on q0 and q1, differently on q2
        r_self = np.array([3.0, 3.0, 1.0] + [3.0] * 32)
        model = CommonalityModel(lambda_mix=1.0, epsilon=0.0)  # Pure similarity projection
        preds = model.predict(0, 3.0, r_self)  # Observe match on q0
        # q1 (same self-response as q0) should have higher prediction than q2 (different)
        assert preds[1] > preds[2], "Self-similar questions should have higher predictions"
        # Predictions should vary (not uniform)
        assert preds.std() > 0.01, "Predictions should vary based on self-similarity"

    def test_k5_creates_gradient(self):
        """With k>0 and λ=0, predictions should vary across questions."""
        model = CommonalityModel(k=5, lambda_mix=0.0, epsilon=0.0)
        r_self = np.random.uniform(1, 5, 35)
        preds = model.predict(0, r_self[0], r_self)
        assert preds.std() > 0.01  # Non-trivial variation

    def test_works_for_all_k(self):
        """Model should work for k=0,1,4,35."""
        r_self = np.random.uniform(1, 5, 35)
        for k in [0, 1, 4, 35]:
            preds = CommonalityModel(k=k).predict(0, 3.0, r_self)
            assert np.all(np.isfinite(preds))

    def test_works_for_all_questions(self):
        """Should work when observing any question."""
        model = CommonalityModel(k=4)
        r_self = np.random.uniform(1, 5, 35)
        for obs_q in [0, 17, 34]:
            preds = model.predict(obs_q, 3.0, r_self)
            assert np.all(np.isfinite(preds))


class TestBatchPrediction:
    """Test batched prediction."""

    def test_batch_shape(self):
        """Batch predict should return (N, 35) array."""
        model = CommonalityModel(k=4)
        n = 10
        obs_qs = np.zeros(n, dtype=int)
        r_partners = np.ones(n) * 3.0
        r_selves = np.random.uniform(1, 5, (n, 35))
        preds = model.predict_batch(obs_qs, r_partners, r_selves)
        assert preds.shape == (n, 35)

    def test_batch_matches_single(self):
        """Batch predict should match individual predictions."""
        model = CommonalityModel(k=4)
        r_selves = np.random.uniform(1, 5, (3, 35))
        obs_qs = np.array([0, 5, 10])
        r_partners = np.array([2.0, 3.0, 4.0])

        batch_preds = model.predict_batch(obs_qs, r_partners, r_selves)
        for i in range(3):
            single_pred = model.predict(obs_qs[i], r_partners[i], r_selves[i])
            assert np.allclose(batch_preds[i], single_pred, rtol=1e-5)


class TestEdgeCases:
    """Test boundary conditions."""

    def test_extreme_responses(self):
        """Should handle all-1s or all-5s responses."""
        model = CommonalityModel(k=4)
        assert np.all(np.isfinite(model.predict(0, 5.0, np.ones(35))))
        assert np.all(np.isfinite(model.predict(0, 1.0, np.ones(35) * 5)))

    def test_identical_responses(self):
        """Should handle identical self-responses."""
        model = CommonalityModel(k=4)
        preds = model.predict(0, 3.0, np.ones(35) * 3.0)
        assert np.all(np.isfinite(preds))


class TestDataLoading:
    """Test data loading functions."""

    def test_loadings_shape(self):
        assert load_factor_loadings(k=4).shape == (35, 4)
        assert load_factor_loadings().shape == (35, 35)

    def test_means_shape_and_range(self):
        means = load_question_means()
        assert means.shape == (35,)
        assert np.all((means >= 1) & (means <= 5))


class TestPerceivedResponse:
    """Test that we use perceived (not ground truth) response for chat condition."""

    def test_chat_uses_perceived_response(self):
        """Chat condition should use postChatResponse (perceived), not partner_response (ground truth)."""
        import pandas as pd
        from models.model import prepare_evaluation_data

        df = pd.read_csv('data/responses.csv', low_memory=False)
        chat_data = df[df['experiment'] == 'chat']

        # Find a participant where perceived != ground truth
        for pid in chat_data['pid'].unique():
            subj = chat_data[chat_data['pid'] == pid]
            obs_row = subj[subj['question_type'] == 'observed']
            if len(obs_row) == 0:
                continue
            perceived = obs_row['postChatResponse'].iloc[0]
            ground_truth = obs_row['partner_response'].iloc[0]
            if pd.notna(perceived) and pd.notna(ground_truth) and perceived != ground_truth:
                # Found a mismatch - run evaluation on just this participant
                eval_data = prepare_evaluation_data(subj)
                if len(eval_data['r_partners']) > 0:
                    # Should use perceived, not ground truth
                    assert eval_data['r_partners'][0] == perceived
                    assert eval_data['r_partners'][0] != ground_truth
                    return  # Test passed
        pytest.skip("No chat participants with perceived != ground truth found")

    def test_nochat_uses_ground_truth(self):
        """No-chat condition should use partner_response (ground truth)."""
        import pandas as pd
        from models.model import prepare_evaluation_data

        df = pd.read_csv('data/responses.csv', low_memory=False)
        nochat_data = df[df['experiment'] == 'no-chat']
        eval_data = prepare_evaluation_data(nochat_data)

        # Should have some participants
        assert len(eval_data['r_partners']) > 0
        # All values should be valid Likert responses (1-5)
        assert np.all((eval_data['r_partners'] >= 1) & (eval_data['r_partners'] <= 5))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
