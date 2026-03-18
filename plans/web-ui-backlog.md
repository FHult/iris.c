# Feature Implementation Plans
*Written 2026-03-04 after deep code review*

---

## Feature 1: Negative Prompt

### Goal
Expose a negative prompt field in the UI and pass it through to the C binary.

### Current State
- No `--negative` CLI flag in `main.c`
- No `negative_prompt` field in `/generate` (server.py)
- No `Job.negative_prompt` attribute
- No textarea in `index.html`

### Implementation

#### Step 1 — C binary (`main.c`, `flux_qwen3.c`)
The distilled 4B/9B models have no CFG, so a true negative prompt isn't
meaningful. The base models (4B-base, 9B-base) do CFG, so negative prompt
should replace the empty string passed as `v_uncond`. Two options:

**Option A — Simple (recommended):** Store `negative_prompt` alongside the job
but don't pass it to the C binary yet. Show it in history metadata. Implement
actual negative conditioning later when base-model performance is optimised.

**Option B — Full:** Add `--negative TEXT` flag to C binary. In `flux_qwen3.c`
encode the negative text the same way as positive. Pass to `flux_generate` as a
separate `negative_embeddings` argument. Use in CFG step:
```c
v = v_uncond + guidance * (v_cond - v_uncond)
```
where `v_uncond` uses negative embeddings instead of empty-string embeddings.

Only Option B adds real value; Option A is metadata-only. Plan for Option B:

**`main.c`:**
```c
// In long_options:
{"negative",  required_argument, 0, 'N'},

// In JSON server-mode parse:
const char *negative_prompt = get_json_string(req, "negative_prompt", "");

// CLI:
case 'N': negative_prompt = optarg; break;
```

**`flux.h` / `flux.c`:** Add `negative_prompt` field to `FluxGenerateParams`.

**`flux_qwen3.c`:** Encode separately — reuse existing `flux_encode_text()`.

**`flux_sample.c`:** CFG step already exists for base models; substitute
negative embeddings for the second (uncond) forward pass.

**Note:** For distilled models, negative_prompt is silently ignored (no CFG).

#### Step 2 — Server (`web/server.py`)
```python
# In /generate handler, add:
negative_prompt = data.get('negative_prompt', '') or ''

# In Job class:
self.negative_prompt = negative_prompt  # new field

# In to_history_dict():
'negative_prompt': self.negative_prompt,

# In FluxServer.generate() / queue_generation(), pass through:
request_data['negative_prompt'] = negative_prompt  # if not empty
```

#### Step 3 — HTML (`web/static/index.html`)
Add after the main prompt `<div class="form-group">`:
```html
<div class="form-group" id="negative-prompt-group" style="display:none">
    <label for="negative-prompt">Negative Prompt
        <span class="label-hint">(base models only)</span>
    </label>
    <textarea id="negative-prompt" name="negative_prompt" rows="2"
        placeholder="blurry, low quality, deformed..."></textarea>
</div>
```
Show/hide based on active model: only visible when model is `flux-klein-4b-base`
or `flux-klein-9b-base`.

#### Step 4 — App.js (`web/static/app.js`)
```javascript
// In getFormParams():
negative_prompt: document.getElementById('negative-prompt').value.trim() || null,

// In addToQueue() body sent to /generate:
negative_prompt: params.negative_prompt || null,

// Show/hide negative prompt on model change:
function updateNegativePromptVisibility(modelInfo) {
    const isBase = modelInfo && modelInfo.model_type === 'base';
    document.getElementById('negative-prompt-group').style.display =
        isBase ? '' : 'none';
}
// Call after /model-info response.

// In history vary / remix: pass negative_prompt from item if present.
```

#### Step 5 — History persistence
Add `negative_prompt` to `to_history_dict()`. When loading history items for
vary/remix, pass `negative_prompt` through.

### Effort: ~3–4 hours (server + UI). Full C implementation adds ~4 hours more.

---

## Feature 2: Per-Slot Reference Strength + Style Reference Mode

### Goal
Each reference image slot gets:
- A **strength slider** (0.0–1.0, how much it influences composition)
- A **mode toggle**: "Composition" vs "Style Reference"

Style Reference mode changes how the reference is consumed: lower weight,
influences colour/texture/style without constraining composition. Implemented
in practice as a lower img2img strength with the reference weighted as an
additional conditioning token rather than the primary latent initialisation.

### Current State
- Single global `img2img_strength` — applied to all references uniformly
- No per-slot metadata in `referenceImageData[]`
- `referenceImageData` is a flat array of data URLs

### Implementation

#### Step 1 — Refactor `referenceImageData` (app.js)
Change from `[null, null, null, null]` (data URLs) to objects:
```javascript
// Was:
let referenceImageData = [null, null, null, null];

// Becomes:
let referenceImageData = [
    { dataUrl: null, strength: 0.85, mode: 'composition' },
    { dataUrl: null, strength: 0.85, mode: 'composition' },
    { dataUrl: null, strength: 0.85, mode: 'composition' },
    { dataUrl: null, strength: 0.85, mode: 'composition' },
];
```
Update every function that reads/writes `referenceImageData[i]` to use
`.dataUrl` instead of the raw value, and check `!= null` on `.dataUrl`.

Functions to update: `setReferenceImage()`, `clearReferenceImage()`,
`renderAllSlots()`, `getFormParams()`, drag handlers.

#### Step 2 — Slot UI overlay (index.html + style.css)
When a slot has an image, show a small overlay at the bottom:
```html
<!-- Inside .reference-image-slot, shown only when has-image -->
<div class="ref-slot-controls" style="display:none">
    <select class="ref-mode-select" data-slot="0">
        <option value="composition">Comp</option>
        <option value="style">Style</option>
    </select>
    <input type="range" class="ref-strength-slider" data-slot="0"
           min="0.1" max="1.0" step="0.05" value="0.85">
</div>
```
CSS: small translucent bar at bottom of slot thumbnail. Only visible when slot
has image.

#### Step 3 — Wiring slot controls (app.js)
```javascript
// In renderAllSlots():
const slotEl = document.querySelector(`[data-slot="${i}"]`);
const controls = slotEl.querySelector('.ref-slot-controls');
const modeSelect = slotEl.querySelector('.ref-mode-select');
const strengthSlider = slotEl.querySelector('.ref-strength-slider');

if (referenceImageData[i].dataUrl) {
    controls.style.display = '';
    modeSelect.value = referenceImageData[i].mode;
    strengthSlider.value = referenceImageData[i].strength;
    modeSelect.onchange = e => { referenceImageData[i].mode = e.target.value; };
    strengthSlider.oninput = e => { referenceImageData[i].strength = parseFloat(e.target.value); };
} else {
    controls.style.display = 'none';
}
```

#### Step 4 — Pass per-slot data in form submission (app.js)
```javascript
// In getFormParams():
referenceImages: referenceImageData
    .filter(s => s.dataUrl !== null)
    .map(s => ({ data: s.dataUrl, strength: s.strength, mode: s.mode })),
```

#### Step 5 — Server handling (server.py)
```python
# /generate: accept objects instead of bare base64 strings
raw_refs = data.get('reference_images', [])
reference_images = []
for ref in raw_refs:
    if isinstance(ref, dict):
        reference_images.append({
            'data': ref['data'],
            'strength': float(ref.get('strength', 0.85)),
            'mode': ref.get('mode', 'composition'),
        })
    else:
        # backward compat: bare base64 string
        reference_images.append({'data': ref, 'strength': 0.85, 'mode': 'composition'})
```

**Style mode mapping:**
- `composition` → pass as reference image normally (existing path)
- `style` → add to reference but set `img2img_strength` lower (e.g. 0.35) and
  mark as style-only conditioning

For now style mode can simply lower strength automatically; a deeper
conditioning approach would require separate model support.

#### Step 6 — C binary (main.c / flux.c)
For composition mode with per-image strength: currently the C binary takes a
single `--img2img-strength` for all references. Extend the JSON protocol:
```json
{
    "reference_images": [
        {"path": "/tmp/ref0.png", "strength": 0.85, "mode": "composition"},
        {"path": "/tmp/ref1.png", "strength": 0.4, "mode": "style"}
    ]
}
```

In `flux.c`, `flux_generate_params_t` gains per-image strength and mode arrays.
The style-mode images are stacked after composition images with lower weight
applied during attention pooling.

**Note:** The C-side change for true per-image conditioning is more involved.
As a safe first step, server.py can compute an effective global `img2img_strength`
as the minimum/average of all slot strengths and pass that, while tracking the
per-slot data for future full implementation.

### Effort: UI only (slots + server averaging) ~3h. Full per-image C conditioning ~8h.

---

## Feature 3: Enhanced Vary-from-History

### Goal
Make the history "Vary" flow match the output-area "Vary Subtle" UX:
- Adjustable strength slider (not hardcoded 0.7)
- Preserve guidance/schedule in history so they round-trip correctly
- Show vary controls in lightbox (not just history grid)

### Current State
- `history-vary-btn` exists and works but:
  - `img2img_strength` hardcoded to 0.7
  - `guidance` and `schedule` not persisted in `to_history_dict()`
  - Lightbox has no "Vary" button — only "Use as Input" (manual)
  - Vary strength slider doesn't appear on history items

### Implementation

#### Step 1 — Persist guidance + schedule in history (server.py)
```python
# Job class — add attributes:
self.guidance = guidance  # already set but...
self.schedule = schedule  # already set but...

# to_history_dict() — add:
'guidance': self.guidance,
'schedule': self.schedule,
'img2img_strength': self.img2img_strength,  # track what was used
```

History items will now carry enough data to accurately reconstruct the vary job.

#### Step 2 — Vary strength in history item UI (app.js + index.html)
In the history grid card, replace the static "~ Vary" button with a compound
button (same pattern as output-area vary-subtle-group):
```html
<div class="history-vary-group">
    <button class="history-quick-btn history-vary-btn">~ Vary</button>
    <input type="range" class="history-vary-strength"
           min="0.1" max="0.95" step="0.05" value="0.7">
    <span class="history-vary-strength-label">0.70</span>
</div>
```
CSS: compact inline slider, hidden until hover or expanded.

In app.js history-vary-btn handler:
```javascript
const strength = card.querySelector('.history-vary-strength').value;
await addToQueue({
    ...varyParams,
    img2img_strength: parseFloat(strength),
    guidance: item.guidance || null,
    schedule: item.schedule || null,
});
```

#### Step 3 — Lightbox Vary button
Add a "Vary" button to the lightbox toolbar (next to "Upscale 2×"):
```html
<button type="button" id="lightbox-vary-btn" title="Vary this image">Vary</button>
```
With an inline strength selector dropdown or popover.

In app.js lightbox-vary-btn handler:
```javascript
lightboxVaryBtn.addEventListener('click', async () => {
    const item = lightboxItems[lightboxIndex];
    if (!item) return;
    // fetch image as base64
    const base64 = await fetchImageAsBase64(lightboxImage.src);
    await addToQueue({
        prompt: item.prompt,
        width: item.width,
        height: item.height,
        steps: item.steps,
        seed: null,
        referenceImages: [base64],
        style: item.style || null,
        guidance: item.guidance || null,
        schedule: item.schedule || null,
        lora: item.lora || null,
        lora_scale: item.lora_scale || 1.0,
        img2img_strength: lightboxVaryStrength,  // from inline slider
    });
    closeLightbox();
});
```

#### Step 4 — "Re-seed exact" option
Add a checkbox to history card: "Same seed" — when checked, passes the
original seed. Useful for exploring a specific composition with minor prompt
tweaks (currently requires manually copying the seed).

### Effort: ~2–3 hours (all UI + server persistence).

---

## Feature 4: Outpaint UI

### Goal
Allow the user to extend an image's canvas in one or more directions and
fill the new area with generated content, using the outpaint LoRA.

Midjourney calls this "Zoom Out" (symmetric) or "Pan" (directional).

### Current State
- `fal-outpaint` LoRA already in curated list (server.py)
- img2img + LoRA pipeline already works end-to-end
- No dedicated outpaint canvas UI
- No canvas-padding logic anywhere in server or client

### Architecture

The outpaint flow is:
1. User selects an existing image (from output or history)
2. User chooses: direction(s) + expansion amount (e.g. +50% right)
3. Server-side: pad image on chosen sides with grey/blurred edge fill
4. Submit to /generate as img2img with outpaint LoRA, `img2img_strength=0.85`
5. The LoRA fills the grey areas with coherent content

### Implementation

#### Step 1 — Server-side image padding (server.py)
Add a `/outpaint-prep` endpoint that takes a job_id or uploaded image and
padding spec:
```python
@app.route("/outpaint-prep", methods=["POST"])
def outpaint_prep():
    data = request.json
    image_data = data['image']  # base64 PNG
    left = data.get('left', 0)   # extra pixels
    right = data.get('right', 0)
    top = data.get('top', 0)
    bottom = data.get('bottom', 0)

    img = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))
    orig_w, orig_h = img.size
    new_w = orig_w + left + right
    new_h = orig_h + top + bottom

    # Snap to 16px grid
    new_w = round(new_w / 16) * 16
    new_h = round(new_h / 16) * 16

    # Create padded canvas — grey fill (neutral for outpaint conditioning)
    padded = Image.new('RGB', (new_w, new_h), (128, 128, 128))
    padded.paste(img, (left, top))

    # Optional: mirror-pad edges for smoother seam
    # (copy a strip from the image edge into the padding zone, blurred)

    buffer = BytesIO()
    padded.save(buffer, format='PNG')
    encoded = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    return jsonify({'padded_image': encoded, 'width': new_w, 'height': new_h})
```

#### Step 2 — Outpaint UI panel (index.html)

Add to the lightbox toolbar (and as a standalone panel accessible from output area):
```html
<button type="button" id="lightbox-outpaint-btn" title="Outpaint — extend canvas">Outpaint</button>

<!-- Outpaint popover/panel -->
<div id="outpaint-panel" class="outpaint-panel" style="display:none">
    <div class="outpaint-header">Extend Canvas</div>
    <div class="outpaint-grid">
        <!-- Directional expansion controls -->
        <div></div>
        <div class="outpaint-dir">
            <label>Top</label>
            <select class="outpaint-amount" data-dir="top">
                <option value="0">No expand</option>
                <option value="0.25">+25%</option>
                <option value="0.5">+50%</option>
                <option value="1.0">+100%</option>
            </select>
        </div>
        <div></div>
        <div class="outpaint-dir">
            <label>Left</label>
            <select class="outpaint-amount" data-dir="left">...</select>
        </div>
        <!-- Center shows image preview thumbnail -->
        <div class="outpaint-preview-thumb" id="outpaint-thumb"></div>
        <div class="outpaint-dir">
            <label>Right</label>
            <select class="outpaint-amount" data-dir="right">...</select>
        </div>
        <div></div>
        <div class="outpaint-dir">
            <label>Bottom</label>
            <select class="outpaint-amount" data-dir="bottom">...</select>
        </div>
        <div></div>
    </div>
    <div class="outpaint-new-size" id="outpaint-new-size">New size: —</div>
    <div class="outpaint-form-group">
        <label>Prompt (optional override)</label>
        <input type="text" id="outpaint-prompt" placeholder="Leave empty to continue original prompt">
    </div>
    <button type="button" id="outpaint-submit-btn" class="btn-primary">Outpaint</button>
    <button type="button" id="outpaint-cancel-btn" class="btn-secondary">Cancel</button>
</div>
```

#### Step 3 — App.js outpaint handler

```javascript
outpaintSubmitBtn.addEventListener('click', async () => {
    const item = lightboxItems[lightboxIndex]; // or currentGeneration
    const image = await fetchImageAsBase64(lightboxImage.src);

    // Compute pixel amounts from percentage selections
    const topPct = parseFloat(document.querySelector('[data-dir="top"]').value);
    const bottomPct = parseFloat(document.querySelector('[data-dir="bottom"]').value);
    const leftPct = parseFloat(document.querySelector('[data-dir="left"]').value);
    const rightPct = parseFloat(document.querySelector('[data-dir="right"]').value);

    const top    = Math.round(item.height * topPct / 16) * 16;
    const bottom = Math.round(item.height * bottomPct / 16) * 16;
    const left   = Math.round(item.width * leftPct / 16) * 16;
    const right  = Math.round(item.width * rightPct / 16) * 16;

    // Prep padded image
    const prep = await fetch('/outpaint-prep', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ image, top, bottom, left, right })
    }).then(r => r.json());

    if (prep.width > 1792 || prep.height > 1792) {
        showError('Resulting size exceeds maximum (1792px)'); return;
    }

    // Ensure outpaint LoRA is available; prompt to download if not
    const outpaintLora = 'flux-outpaint-lora.safetensors';
    if (!availableLoras.find(l => l.filename === outpaintLora)) {
        showError('Outpaint LoRA not downloaded. Use + LoRA panel to download fal-outpaint.');
        return;
    }

    const prompt = document.getElementById('outpaint-prompt').value.trim()
                || item.prompt;

    await addToQueue({
        prompt,
        width: prep.width,
        height: prep.height,
        steps: item.steps,
        seed: null,
        referenceImages: [prep.padded_image],
        img2img_strength: 0.85,
        lora: outpaintLora,
        lora_scale: 1.0,
        style: null,
        guidance: null,
        schedule: null,
    });
    closeOutpaintPanel();
    closeLightbox();
});
```

#### Step 4 — Live size preview
As the user adjusts directional expansion selects, compute and show the new
canvas dimensions in `#outpaint-new-size`. Warn if > 1792px.

#### Step 5 — Outpaint LoRA auto-suggest
If user clicks Outpaint and the LoRA is missing, show a one-click download
prompt: "The outpaint LoRA is needed. Download now? (234 MB)". This calls the
existing download flow for the `fal-outpaint` curated LoRA.

### Effort: ~5–7 hours (server padding endpoint + UI panel + app.js flow).

---

## Implementation Order

| # | Feature | Effort | Value |
|---|---|---|---|
| 3 | Vary-from-history enhanced | 2–3h | High — fixes existing friction immediately |
| 1 | Negative prompt (UI + server) | 3–4h | Medium — needs base model to be useful |
| 2 | Per-slot reference strength | 3h (UI+server avg) / 8h (full C) | High — useful for all img2img |
| 4 | Outpaint UI | 5–7h | High — biggest feature gap vs Midjourney |

**Recommended order:** 3 → 2 (UI only) → 4 → 1 (C backend last since distilled
models need no CFG).

---

## Shared Utilities Needed

Several features above need the same helper — extract once:

```javascript
// Fetch an image URL and return as base64 data URL
async function fetchImageAsBase64(url) {
    const resp = await fetch(url.split('?')[0]);
    const blob = await resp.blob();
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}
```

Currently duplicated in: upscaleImage, varySubtleBtn handler, history vary
handler, lightbox use-btn handler. Consolidate before adding more callers.
