"""Guard for the SREF content-shared-pair contrastive (BACKLOG SREF-INFONCE-VOID / SREF-PAIR-VS-BANK).

This is the test that would have caught the void Stage-0.5 objective before an 8000-step run wedged
the machine and returned a false NO-GO on the flagship hypothesis. Its job is to assert, hermetically,
that a REFERENCE-BLIND model cannot score well — where "reference-blind" means *ignores the reference
while still depending on its input*, which is the collapse we actually observe. Output-constancy is a
strictly weaker failure and every prior objective punished only that.

The key property under test: for any z_a == z_b, the pair loss is >= ln 2, with equality iff the
output is equidistant from the two references. A reference-aware model drives it to ~0.
"""
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # train/

from ip_adapter.loss import pair_contrastive_loss, pair_row_accuracy

LN2 = float(np.log(2.0))
TAU = 0.1
D = 64


def _l2(v):
    return v / mx.maximum(mx.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _refs(B, seed):
    """Two batches of distinct, frozen, L2-normalised reference descriptors."""
    mx.random.seed(seed)
    return _l2(mx.random.normal((B, D))), _l2(mx.random.normal((B, D)))


# --------------------------------------------------------------------------------------
# The floor: no reference-blind model can beat ln 2.
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(12))
def test_reference_blind_is_bounded_below_by_ln2(seed):
    """z_a == z_b (the collapse) => loss >= ln 2, for ARBITRARY z. This is the anti-collapse
    guarantee the old self-target InfoNCE and the correct-ref-only bank both lack."""
    s_a, s_b = _refs(8, seed)
    mx.random.seed(seed + 1000)
    z = _l2(mx.random.normal((8, D)))                    # any reference-blind output whatsoever
    loss = float(pair_contrastive_loss(z, z, s_a, s_b, tau=TAU))
    assert loss >= LN2 - 1e-6, f"reference-blind scored {loss:.6f} < ln2 ({LN2:.6f})"


def test_reference_blind_floor_is_attained_exactly_at_the_symmetric_point():
    """Equality case: an output equidistant from both references pays exactly ln 2. Confirms the
    bound is tight (not a loose inequality that a collapsed model could sit far below)."""
    s_a, s_b = _refs(16, 7)
    z = _l2(s_a + s_b)                                   # equidistant by symmetry
    loss = float(pair_contrastive_loss(z, z, s_a, s_b, tau=TAU))
    assert abs(loss - LN2) < 1e-5, f"symmetric reference-blind output scored {loss:.6f}, want ln2"


def test_constant_output_is_just_a_special_case_of_reference_blind():
    """The degenerate collapse (a single constant vector for every sample) is subsumed by the same
    bound. Prior objectives punished ONLY this case, which is why they missed the real one."""
    s_a, s_b = _refs(8, 3)
    mx.random.seed(99)
    z = mx.broadcast_to(_l2(mx.random.normal((1, D))), (8, D))
    assert float(pair_contrastive_loss(z, z, s_a, s_b, tau=TAU)) >= LN2 - 1e-6


def test_perfect_reference_aware_model_scores_near_zero():
    """A model that adopts whichever style it is handed separates cleanly. Without this the ln2
    bound would be vacuous (a loss no model can ever beat is not an objective)."""
    s_a, s_b = _refs(16, 11)
    loss = float(pair_contrastive_loss(s_a, s_b, s_a, s_b, tau=TAU))
    assert loss < 0.05, f"reference-aware model scored {loss:.6f}; expected ~0"
    assert loss < LN2 / 10


def test_partial_reference_use_lands_between_the_bounds():
    """Monotone in how much the branches diverge: interpolating from blind to aware must decrease
    the loss, so gradient descent has a path out of the collapsed basin."""
    s_a, s_b = _refs(16, 13)
    prev = None
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        z_a = _l2((1 - w) * (s_a + s_b) + w * 2.0 * s_a)
        z_b = _l2((1 - w) * (s_a + s_b) + w * 2.0 * s_b)
        loss = float(pair_contrastive_loss(z_a, z_b, s_a, s_b, tau=TAU))
        if prev is not None:
            assert loss < prev + 1e-9, f"loss rose ({prev:.4f} -> {loss:.4f}) while using the ref more"
        prev = loss
    assert prev < 0.05


def test_gradient_at_the_collapsed_point_is_nonzero():
    """A bound is useless if the collapsed point is a saddle with zero gradient. Check the loss
    actually pushes the two branches apart from z_a == z_b."""
    s_a, s_b = _refs(8, 17)
    mx.random.seed(5)
    z = _l2(mx.random.normal((8, D)))

    def f(za, zb):
        return pair_contrastive_loss(za, zb, s_a, s_b, tau=TAU)

    g_a, g_b = mx.grad(f, argnums=(0, 1))(z, z)
    assert float(mx.linalg.norm(g_a)) > 1e-3
    assert float(mx.linalg.norm(g_b)) > 1e-3
    # descending the gradient must separate the branches (they start identical)
    za2, zb2 = _l2(z - 0.1 * g_a), _l2(z - 0.1 * g_b)
    assert float(mx.mean(mx.sum(za2 * zb2, axis=1))) < 1.0 - 1e-4


# --------------------------------------------------------------------------------------
# The diagnostic: foreign-row accuracy is chance under collapse.
# --------------------------------------------------------------------------------------

def test_row_accuracies_sum_to_one_under_reference_blindness():
    """For z_a == z_b exactly one row is correct, so mean row accuracy is exactly 0.5 = chance.
    This is the in-loop collapse metric. Contrast with a correct-ref-only negative bank, whose
    top-1 accuracy reads ~98% on this same collapsed model (SREF-PAIR-VS-BANK)."""
    s_a, s_b = _refs(256, 23)
    mx.random.seed(31)
    z = _l2(mx.random.normal((256, D)))
    acc_a = float(pair_row_accuracy(z, s_a, s_b))
    acc_b = float(pair_row_accuracy(z, s_b, s_a))
    assert abs((acc_a + acc_b) - 1.0) < 1e-6
    assert abs(0.5 * (acc_a + acc_b) - 0.5) < 1e-6


def test_row_accuracy_is_one_for_a_reference_aware_model():
    s_a, s_b = _refs(64, 29)
    assert float(pair_row_accuracy(s_a, s_a, s_b)) == pytest.approx(1.0)
    assert float(pair_row_accuracy(s_b, s_b, s_a)) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Regression guard: encode WHY the old objective was void, so it cannot be reintroduced.
# --------------------------------------------------------------------------------------

def _self_target_infonce(x0_pred, latent, tau=TAU):
    """The SUPERSEDED Stage-0.5 term: contrast each prediction against its OWN target's style stats.
    Reproduced here only to assert its defect. Do not use it."""
    def stats(x):
        mu = x.mean(axis=(2, 3))
        sd = mx.sqrt(((x - mu[:, :, None, None]) ** 2).mean(axis=(2, 3)) + 1e-5)
        return _l2(mx.concatenate([mu, sd], axis=1))

    logits = (stats(x0_pred) @ stats(latent).T) / tau
    B = latent.shape[0]
    lse = mx.logsumexp(logits, axis=1)
    diag = logits[mx.arange(B), mx.arange(B)]
    acc = float(mx.mean((mx.argmax(logits, axis=1) == mx.arange(B)).astype(mx.float32)))
    return float(mx.mean(lse - diag)), acc


def test_self_target_infonce_is_won_by_a_reference_blind_denoiser():
    """SREF-INFONCE-VOID, locked in. A perfect but reference-blind denoiser (x0_pred == x0) attains
    the OLD objective's global minimum for the batch and classifies at 100% — the reference is not
    even an argument of that loss. Meanwhile the pair loss assigns the same model >= ln 2.

    If this test ever fails, someone has changed the old term's semantics; if it passes, the old
    term remains unusable as an anti-collapse signal and must not be reintroduced."""
    mx.random.seed(0)
    x0 = mx.random.normal((8, 4, 16, 16)) * mx.random.normal((8, 4, 1, 1))  # per-sample style scale

    blind_loss, blind_acc = _self_target_infonce(x0, x0)     # ref-blind, perfect: x0_pred == x0
    assert blind_acc == pytest.approx(1.0), "a reference-blind denoiser should ace the old objective"

    # ...and it sits at the term's floor: any perturbation of the prediction can only raise it.
    mx.random.seed(1)
    worse, _ = _self_target_infonce(x0 + 0.5 * mx.random.normal(x0.shape), x0)
    assert worse >= blind_loss - 1e-6

    # The pair loss, on the same reference-blind model, refuses to award it anything below ln 2.
    s_a, s_b = _refs(8, 41)
    mx.random.seed(2)
    z = _l2(mx.random.normal((8, D)))
    assert float(pair_contrastive_loss(z, z, s_a, s_b, tau=TAU)) >= LN2 - 1e-6
