# FLUX.2 Web UI — Improvement Backlog

## Tier 1 — High Impact, Expose Existing C Binary Capability

- [x] **1. Guidance scale slider** — Essential creative control for base models; already in C binary (`-g` flag, `flux_params.guidance`). Auto-selects 1.0 distilled / 4.0 base.
- [x] **2. Increase steps max to 50+** — Base models need 50 steps; slider currently capped at 8. C binary supports up to 256 (`FLUX_MAX_STEPS`).
- [x] **3. Schedule selection** (linear/power/sigmoid) — Dramatically different results; already in C binary (`--linear`, `--power`, `--power-alpha`).
- [x] **4. Model info display** — Users don't know what model is loaded (4B/9B, distilled/base). Add `/model-info` endpoint + header display.
- [ ] **5. Embedding cache in server mode** — Skip ~1s text encoding on repeated prompts. CLI uses 4-bit quantized cache (`embcache.c`). Server re-encodes every time.

## Tier 2 — High Impact, New UX Features

- [x] **6. Ctrl+Enter to generate** — Standard UX pattern in AI tools. Trivial to add.
- [x] **7. ETA display** — Has step_time + remaining steps; just needs math in progress handler.
- [x] **8. History search/filter** — Finding past generations gets hard with many images. Add text search input.
- [x] **9. Lightbox metadata overlay** — See prompt/seed/params without going back to grid.
- [x] **10. Image caching headers** — Immutable images should cache; saves bandwidth on history grid scroll.

## Tier 3 — Medium Impact

- [x] **11. Store generation time in history** — Track performance, know what to expect.
- [x] **12. Persist style preset & variations count** in localStorage — Settings lost on reload.
- [x] **13. Auto-restart on process crash** — Hung server requires manual restart.
- [x] **14. Variation grid/mosaic view** — See all variations at once instead of one-by-one in history.
- [x] **15. Side-by-side comparison** — Compare two images (before/after, different seeds).

## Tier 4 — Nice to Have

- [x] **16. Lightbox delete + download buttons** — Common actions require leaving lightbox currently.
- [ ] **17. Prompt templates** — Reusable prompt structures with variable placeholders.
- [ ] **18. Batch prompt generation** — Submit a list of different prompts to generate in sequence.
- [x] **19. Reference image reorder** — Drag to reorder slots (order matters for multi-ref T_offset).
- [ ] **20. Per-job timeout** — Prevent hung generations from blocking the queue forever.

## Tier 5 — Extended Features

- [x] **26. Dark/light theme toggle** — Toggle between dark and light themes with localStorage persistence.
