"""
Held-out cond_gap based checkpoint selection + early stopping for IP-adapter training.

cond_gap = loss_null - loss_cond on a HELD-OUT set is the conditioning-quality metric.
Training loss falls monotonically while held-out cond_gap can DEGRADE (over-training: the
adapter keeps fitting the flow objective but loses reference conditioning — observed in the
warmup flywheel, cond_gap +0.0273 -> -0.0057 while train_loss fell). So checkpoint SELECTION
and EARLY-STOP must key on held-out cond_gap, NEVER train_loss (or even raw val_loss, which
also falls). This module is the pure decision logic — no I/O, stdlib only — ready to wire
into the trainer's held-out eval loop (T-05). See BACKLOG PROD-1/PROD-2.

It mirrors the doctor's cond_gap-stall / over-training detector so that training-time
stopping and post-hoc analysis agree on what "over-training" means.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class CondGapPoint:
    """One held-out evaluation during training."""
    step: int
    cond_gap: float
    train_loss: Optional[float] = None


def best_point(history: Sequence[CondGapPoint]) -> Optional[CondGapPoint]:
    """The checkpoint to SELECT: highest held-out cond_gap, earliest on ties (the
    least-trained checkpoint at a given quality is preferred). None for empty history."""
    best: Optional[CondGapPoint] = None
    for p in history:
        if best is None or p.cond_gap > best.cond_gap:
            best = p
    return best


def evals_since_best(history: Sequence[CondGapPoint]) -> int:
    """Evaluations since the best cond_gap (0 ⇒ the latest eval is the best)."""
    if not history:
        return 0
    best_i = 0
    for i, p in enumerate(history):
        if p.cond_gap > history[best_i].cond_gap:
            best_i = i
    return (len(history) - 1) - best_i


def should_stop(history: Sequence[CondGapPoint], patience: int = 3,
                min_delta: float = 0.0) -> bool:
    """Early-stop when held-out cond_gap has not set a new high (by more than `min_delta`)
    for `patience` consecutive evaluations. Requires at least `patience`+1 evals so an
    early run never stops prematurely."""
    if len(history) < patience + 1:
        return False
    last_improve = 0
    running = history[0].cond_gap
    for i in range(1, len(history)):
        if history[i].cond_gap > running + min_delta:
            running = history[i].cond_gap
            last_improve = i
    return (len(history) - 1) - last_improve >= patience


def is_overtraining(history: Sequence[CondGapPoint], window: int = 3) -> bool:
    """True when the last `window` evals show cond_gap STRICTLY DECREASING while train_loss
    is non-increasing (and strictly lower end-to-end) — the over-training signature (fitting
    the flow objective while losing conditioning). Distinct from a flat plateau; matches the
    doctor's detector. Requires train_loss on every windowed point."""
    if len(history) < window:
        return False
    win = history[-window:]
    cg = [p.cond_gap for p in win]
    tl = [p.train_loss for p in win]
    if any(t is None for t in tl):
        return False
    cg_down = all(cg[i] < cg[i - 1] for i in range(1, len(cg)))
    tl_down = all(tl[i] <= tl[i - 1] for i in range(1, len(tl))) and tl[-1] < tl[0]
    return cg_down and tl_down
