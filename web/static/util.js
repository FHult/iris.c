// FLUX.2 Web UI - shared image helpers.
//
// Extracted from app.js so additional frontend modules can share a single copy
// of these helpers (prerequisite for the advanced Web UI features in BACKLOG).
// These are intentionally global (top-level) functions so any script loaded
// after this one can call them. No behavior change vs. the previous in-app.js
// definitions.

function blobToDataURL(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

async function fetchImageAsBase64(url) {
    const resp = await fetch(url.split('?')[0]);
    return blobToDataURL(await resp.blob());
}
