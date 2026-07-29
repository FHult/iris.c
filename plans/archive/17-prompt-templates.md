# Plan: Prompt Templates (Backlog #17)

## Context
Users often craft similar prompt structures repeatedly (portraits, landscapes, product shots, etc.)
with only a few details changing. Prompt templates let users define reusable prompt patterns with
`{variable}` placeholders that get filled in via input fields. This is a **pure frontend feature** —
no server changes needed, templates stored in localStorage.

## UI Flow
1. Click template button (📋) in the prompt action buttons area (next to enhance & history)
2. Dropdown opens showing built-in + user templates grouped by category
3. Select a template → prompt textarea fills with pattern, variable input fields appear below
4. Type in variable fields → prompt updates live in textarea
5. "Save Current" button saves current prompt (must contain `{variables}`) as user template
6. User templates can be deleted; built-in templates are permanent

## Files to Modify

### 1. `web/static/index.html`

**Add template button** in `.prompt-buttons` (after history button, ~line 57):
- New `<button id="template-btn">` with document/plus SVG icon

**Add template dropdown** after `.prompt-wrapper` (~line 59):
- `#template-dropdown` div with header ("Templates" title + "Save Current" button) and content area
- `#template-variables` div with header ("Fill template variables:" + clear ×) and `#template-variables-inputs` container

### 2. `web/static/app.js`

**State variables** (near other globals):
- `activeTemplate`, `templateVariableValues`, `TEMPLATE_STORAGE_KEY`

**Built-in templates array** (~8 templates):
| Category | Name | Pattern |
|----------|------|---------|
| Characters | Portrait | `Portrait of a {subject}, {lighting} lighting, {background} background` |
| Characters | Character Design | `{character_type} character, {clothing}, {pose}, {expression} expression` |
| Scenes | Landscape | `A {location} landscape, {time_of_day}, {weather} weather, {mood} atmosphere` |
| Scenes | Interior | `{room_type} interior, {style} style, {lighting} lighting, {mood} atmosphere` |
| Artistic | Style Transfer | `A {subject} in the style of {artist}, {medium} art, {mood} mood` |
| Artistic | Abstract | `Abstract {subject}, {colors} colors, {composition} composition` |
| Commercial | Product Shot | `{product} product photography, {angle} angle, {lighting} lighting` |
| Fantasy | Creature | `A {creature} with {features}, in a {environment} environment` |

**Core functions:**
- `extractVariables(pattern)` — regex `/{(\w+)}/g`, returns unique variable names
- `loadTemplates()` — load built-in + user templates from localStorage
- `saveUserTemplates()` — persist to localStorage (`flux_prompt_templates` key)
- `applyTemplate(template)` — set textarea to pattern, render variable inputs
- `renderTemplateVariables()` — create input fields for each `{variable}`, grid layout
- `updatePromptFromTemplate()` — replace `{vars}` with input values, update textarea live
- `clearTemplate()` — remove variable inputs, reset state
- `renderTemplateDropdown()` — populate dropdown with categorized template list
- `saveCurrentAsTemplate()` — validate has `{variables}`, prompt for name, save
- `deleteUserTemplate(id)` — remove from localStorage, re-render

**Event handlers:**
- Template button click → toggle dropdown
- Template item click → apply template, close dropdown
- Variable input → live-update prompt
- Clear button → clear template
- Manual prompt edit → clear template if structure changed
- Close dropdown on outside click

### 3. `web/static/style.css`

**Template dropdown** (mirrors prompt-history-dropdown pattern):
- `.template-dropdown` — positioned below prompt, max-height 400px, scrollable
- `.template-dropdown-header` — sticky header with title + save button
- `.template-section-header` — uppercase category labels
- `.template-item` — two-line: name (bold) + pattern preview (gray italic), hover highlight
- `.template-item-delete` — red × button, visible on hover only

**Template variables panel:**
- `.template-variables` — subtle accent-tinted background with border, below prompt
- `#template-variables-inputs` — CSS grid, `repeat(auto-fill, minmax(200px, 1fr))`
- `.template-variable-input` — label shows `{varName}` in monospace, text input below

## Verification
1. Click template button → dropdown shows 8 built-in templates in 5 categories
2. Select "Portrait" → textarea shows pattern, 3 variable inputs appear
3. Type "a young woman" in subject → prompt updates live to "Portrait of a young woman, {lighting} lighting..."
4. Fill all variables → complete prompt ready to generate
5. Click clear → variables panel disappears
6. Write a prompt with `{variables}`, click "Save Current" → appears in "Your Templates" section
7. Delete user template → removed from dropdown
8. Reload page → user templates persist
9. Generate with a composed prompt → works normally with style presets
