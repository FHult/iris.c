# Plan: Hot precompute cache (defer-rmtree LRU) — cut redundant cold↔hot transfers

**Status:** design / not started. A precompute-throughput optimization. Land in a
non-campaign window (touches the live orchestrator publish/cleanup path + cache
resolution). Independent of, and complementary to, the proxy VAE and
`--subsample-per-shard`.

## Problem (measured)

Each flywheel iteration currently:
1. stages selected shards' precompute cold→hot (DataStager — a real **copy** because
   hot=`2TBSSD` and cold=`16TBCold` are different physical devices, so symlinks don't apply),
2. precomputes any cache-miss shards on hot,
3. publishes precompute to cold (versioned `current`) **and `shutil.rmtree`s the per-iter hot
   precompute** (orchestrator.py ~3234/3247, "data is now durable on cold").

So a shard precomputed (or staged) in iter N is deleted from hot, then **re-copied cold→hot**
when a later iter re-selects it. That round-trip is pure waste for recurring shards.

**Measured reuse (flywheel_history.db `iterations.selected_shards`, warmup-run1, 13 iters,
520 selections, 361 unique):**

| Cache window | Mean per-iter shard reuse |
|---|---|
| immediately prior iter (LRU-1) | **30.6%** |
| last 3 iters (LRU-3) | 31.0% |
| any prior iter (unbounded) | 33.1% |

Overall reuse fraction = 30.6%; per-iter trend rises **30% → 40%** as a campaign matures.
warmup-run2 (2 iters) agrees (~29%).

## Key insight: a 1-iteration cache captures ~all of it

LRU-1 (30.6%) ≈ LRU-3 (31.0%) ≈ unbounded (33.1%) — **almost all reuse is from the
immediately preceding iteration.** A 1-iter cache yields ~92% of the maximum benefit. So:
- no need to hold the union of all iters on hot,
- no need to predict the adaptive shard schedule (it's a feedback loop — selection depends on
  prior iters' training results — so it can't be reliably mapped at the start anyway),
- the change is almost surgical: **don't eagerly `rmtree`; keep one iteration resident.**

## Design

**Decouple the two operations the publish step currently fuses:**
- **Archive to cold** = durability. Keep doing it every iter (unchanged).
- **Delete from hot** = eviction. **Defer it.** Instead of `rmtree` after publish, register
  the per-iter precompute dir in a hot cache index and evict by LRU under a budget.

**Hot cache layout.** A single hot dir, e.g. `<hot>/precompute_cache/<encoder>/<version>/`,
holding `{shard}_{idx}.npz` keyed by the same content-addressed identity the cold cache uses
(PRECOMP-3 `encoder_config_subset`). Per-iter staging **links** (hardlink within the hot
device; symlink otherwise) the needed records from the cache into the iter's working dir,
rather than copying from cold. Only cache-*misses* are copied cold→hot (and, once computed,
land in the cache too).

**Staging path (per iter):**
1. Resolve each selected shard's records against the hot cache index.
2. Cache hit → link from `precompute_cache` into the iter dir (free).
3. Cache miss → copy cold→hot (existing DataStager path) **into the cache**, then link.
4. After train + publish-to-cold: update LRU timestamps; do **not** rmtree the cache;
   evict LRU entries only if the cache exceeds its budget.

**Eviction.** LRU on shard granularity with a hard hot budget (config
`storage.hot_precompute_cache_gb`, default ~1.5× one iter's footprint so LRU-1 always fits;
0 = disabled → current behaviour). Evicting a cache entry is safe — it's durable on cold;
a future hit just re-copies. Never evict the in-flight iter's working set.

**Cache resolution.** Training reads precompute via `effective_dir` / the `current`
resolution. Add the hot cache as a first lookup tier: **hot cache → cold `current` → newest
complete cold version.** The content-addressing makes coherence cheap (same identity ⇒ same
bytes); a hot entry is always a valid substitute for the cold one.

## Footprint (measured from the cold VAE `manifest.json`)

VAE latent `[32,64,64]` f32 = **0.5 MB/record exactly**. The cold VAE version manifest reports
**747,739 records across 152 shards ≈ 4,919 records/shard → ~2.5 GB/shard VAE**. Adding the
(4-bit-quantized, smaller) qwen3 + siglip caches → **~4–5 GB/shard all three encoders**, so a
~42-shard iter is **~170–210 GB**. That's the same order the pipeline already stages
transiently each iter today, so an LRU-1 cache adds ~0 steady-state hot usage — it just *keeps*
what it already copies (peak ≈ 1.3× one iter ≈ ~260 GB). Comfortable on the 2 TB hot tier
(~10–13%); set `hot_precompute_cache_gb` accordingly with headroom for LRU-2 if wanted.
(qwen3/siglip per-record sizes are estimated, not exact — confirm with a `du` on a cold
version dir at idle; the VAE figure is exact.)

## Scope / non-goals

- Saves ~30% of the **precompute-npz cold→hot copy volume** (cache-hit shards). Rising to ~40%
  as campaigns mature, and higher across campaigns (more of the dataset precomputed).
- Does **not** help cache-*miss* shards (fresh tar-in + encode + publish-out) or the
  encode/training compute. It is a transfer-I/O win, not 30% of wall-clock.
- No change to durability (cold remains the source of truth) or to the content-addressed
  cache identity.

## Risks

- **Cache/cold coherence** — mitigated by content-addressing (PRECOMP-3); a hot entry equals
  its cold counterpart by construction. Guard: verify identity (version dir) on link, not bytes.
- **Hot budget blowout** — the hard `hot_precompute_cache_gb` cap + LRU is the safety net;
  default-disabled so it can't surprise an operator.
- **Stale links** — links into per-iter working dirs must be cleaned per iter (the working dir
  is still ephemeral; only the cache persists).
- **Cross-device hardlink** — hardlinks only work within the hot device; fall back to symlink
  into the cache (still zero-copy). Cache and iter working dirs must be same-device for
  hardlinks (they are — both on hot).

## Effort / sequencing

- Moderate: changes in DataStager (link-from-cache path), orchestrator publish/cleanup
  (defer rmtree + LRU index + budget), and `effective_dir` (hot-cache tier). ~1–2 days +
  a dry-run validation that an iter with known overlap copies only the cache-miss shards.
- **Land in a non-campaign window** (publish/cleanup is live-pipeline code).
- Measure-as-you-go: log per-iter `copied_shards` vs `linked_shards` to confirm the realized
  ~30% saving in production.
