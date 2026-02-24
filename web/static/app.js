// FLUX.2 Web UI - Frontend Logic

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('generate-form');
    const generateBtn = document.getElementById('generate-btn');
    const progressSection = document.getElementById('progress-section');
    const progressPrompt = document.getElementById('progress-prompt');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const outputArea = document.getElementById('output-area');
    const imageInfo = document.getElementById('image-info');
    const seedDisplay = document.getElementById('seed-display');
    const stepsInput = document.getElementById('steps');
    const stepsValue = document.getElementById('steps-value');
    const showStepsCheckbox = document.getElementById('show-steps');
    const remixBtn = document.getElementById('remix-btn');
    const historySection = document.getElementById('history-section');
    const historyGrid = document.getElementById('history-grid');
    const aspectPreset = document.getElementById('aspect-preset');
    const widthSelect = document.getElementById('width');
    const heightSelect = document.getElementById('height');
    const queueSection = document.getElementById('queue-section');
    const queueList = document.getElementById('queue-list');
    const queueCount = document.getElementById('queue-count');
    const clearAllImagesBtn = document.getElementById('clear-all-images');

    const lightboxModal = document.getElementById('lightbox-modal');
    const lightboxImage = document.getElementById('lightbox-image');
    const lightboxClose = document.getElementById('lightbox-close');
    const lightboxContent = document.getElementById('lightbox-content');
    const cancelBtn = document.getElementById('cancel-btn');
    const useAsInputBtn = document.getElementById('use-as-input-btn');
    const slideshowBtn = document.getElementById('slideshow-btn');
    const lightboxPrev = document.getElementById('lightbox-prev');
    const lightboxNext = document.getElementById('lightbox-next');

    // Cropping elements
    const cropOverlay = document.getElementById('crop-overlay');
    const cropSelection = document.getElementById('crop-selection');
    const lightboxToolbar = document.getElementById('lightbox-toolbar');
    const cropToolbar = document.getElementById('crop-toolbar');
    const lightboxCropBtn = document.getElementById('lightbox-crop-btn');
    const lightboxUseBtn = document.getElementById('lightbox-use-btn');
    const cropApplyBtn = document.getElementById('crop-apply-btn');
    const cropCancelBtn = document.getElementById('crop-cancel-btn');
    const cropDimensions = document.getElementById('crop-dimensions');
    const cropPreset = document.getElementById('crop-preset');
    const copySeedBtn = document.getElementById('copy-seed-btn');
    const downloadBtn = document.getElementById('download-btn');
    const promptHistoryBtn = document.getElementById('prompt-history-btn');
    const promptHistoryDropdown = document.getElementById('prompt-history-dropdown');
    const stylePresetSelect = document.getElementById('style-preset');
    const styleHint = document.getElementById('style-hint');
    const stepsHint = document.getElementById('steps-hint');
    const enhanceBtn = document.getElementById('enhance-btn');
    const promptTextarea = document.getElementById('prompt');
    const variationsValue = document.getElementById('variations-value');
    const variationsInc = document.getElementById('variations-inc');
    const variationsDec = document.getElementById('variations-dec');
    const guidanceInput = document.getElementById('guidance');
    const guidanceValue = document.getElementById('guidance-value');
    const scheduleSelect = document.getElementById('schedule');
    const loraSelect = document.getElementById('lora-select');
    const loraBrowseBtn = document.getElementById('lora-browse-btn');
    const loraScaleInput = document.getElementById('lora-scale');
    const loraScaleValue = document.getElementById('lora-scale-value');
    const loraScaleGroup = document.getElementById('lora-scale-group');
    const loraPanel = document.getElementById('lora-panel');
    const loraPanelClose = document.getElementById('lora-panel-close');
    const loraPanelBody = document.getElementById('lora-panel-body');
    const loraPanelBackdrop = document.getElementById('lora-panel-backdrop');
    const advancedControls = document.getElementById('advanced-controls');
    const toggleAdvancedBtn = document.getElementById('toggle-advanced');
    const modelInfoEl = document.getElementById('model-info');
    const historySearch = document.getElementById('history-search');
    const lightboxMetadata = document.getElementById('lightbox-metadata');
    const lightboxDownloadBtn = document.getElementById('lightbox-download-btn');
    const lightboxDeleteBtn = document.getElementById('lightbox-delete-btn');
    const lightboxVariationsBtn = document.getElementById('lightbox-variations-btn');
    const compareModal = document.getElementById('compare-modal');
    const compareGrid = document.getElementById('compare-grid');
    const compareClose = document.getElementById('compare-close');
    const compareBar = document.getElementById('compare-bar');
    const compareBarText = document.getElementById('compare-bar-text');
    const compareBarBtn = document.getElementById('compare-bar-btn');
    const compareBarClear = document.getElementById('compare-bar-clear');

    let currentEventSource = null;
    let referenceImageData = [null, null, null, null]; // Up to 4 reference images
    let availableLoras = [];   // [{name, filename, size_mb}] from /available-loras
    let curatedLoras = [];     // [{...curated entry, downloaded: bool}]
    let activeDownloads = {};  // {dl_id: {percent, done, error, interval}}
    let currentGeneration = null; // Store current generation params for remix
    let generationQueue = [];
    let isGenerating = false;
    let currentJobId = null;
    let lightboxIndex = -1;
    let lightboxItems = [];
    let slideshowTimer = null;
    let cachedHistory = [];
    let historyPageSize = 20;
    let historyVisible = historyPageSize;

    // Style presets and step guidance
    let stylePresets = {};
    let stepGuidance = {};
    let isPromptEnhanced = false;
    let originalPrompt = '';

    // Variations count
    let variationsCount = 1;

    // Template management
    const templateBtn = document.getElementById('template-btn');
    const templateDropdown = document.getElementById('template-dropdown');
    const templateDropdownContent = document.getElementById('template-dropdown-content');
    const templateSaveBtn = document.getElementById('template-save-btn');
    const templateVariablesDiv = document.getElementById('template-variables');
    const templateVariablesInputs = document.getElementById('template-variables-inputs');
    const templateClearBtn = document.getElementById('template-clear-btn');
    let activeTemplate = null;
    let templateVariableValues = {};
    let userTemplates = [];
    const TEMPLATE_STORAGE_KEY = 'flux_prompt_templates';

    // Compare selection state
    let selectedForCompare = [];

    // Theme toggle
    const themeToggle = document.getElementById('theme-toggle');
    const themeIconSun = document.getElementById('theme-icon-sun');
    const themeIconMoon = document.getElementById('theme-icon-moon');

    function applyTheme(theme) {
        if (theme === 'light') {
            document.documentElement.setAttribute('data-theme', 'light');
            themeIconSun.style.display = '';
            themeIconMoon.style.display = 'none';
        } else {
            document.documentElement.removeAttribute('data-theme');
            themeIconSun.style.display = 'none';
            themeIconMoon.style.display = '';
        }
    }

    // Load saved theme
    applyTheme(localStorage.getItem('theme') || 'dark');

    themeToggle.addEventListener('click', () => {
        const isLight = document.documentElement.hasAttribute('data-theme');
        const newTheme = isLight ? 'dark' : 'light';
        localStorage.setItem('theme', newTheme);
        applyTheme(newTheme);
    });

    function updateVariationsDisplay() {
        variationsValue.textContent = variationsCount;
        // Update generate button text hint
        if (!isGenerating) {
            generateBtn.textContent = variationsCount > 1 ? `Generate ${variationsCount}` : 'Generate';
        }
    }

    variationsInc.addEventListener('click', () => {
        if (variationsCount < 8) {
            variationsCount++;
            updateVariationsDisplay();
            saveSettings();
        }
    });

    variationsDec.addEventListener('click', () => {
        if (variationsCount > 1) {
            variationsCount--;
            updateVariationsDisplay();
            saveSettings();
        }
    });

    // Advanced controls toggle
    toggleAdvancedBtn.addEventListener('click', () => {
        const visible = advancedControls.style.display !== 'none';
        advancedControls.style.display = visible ? 'none' : 'grid';
        toggleAdvancedBtn.textContent = visible ? 'Advanced options' : 'Hide advanced';
    });

    // Guidance slider
    guidanceInput.addEventListener('input', () => {
        const val = parseFloat(guidanceInput.value);
        guidanceValue.textContent = val === 0 ? 'auto' : val.toFixed(1);
    });

    // Load model info into header
    fetch('/model-info').then(r => r.json()).then(data => {
        if (data.model) modelInfoEl.textContent = data.model;
    }).catch(() => {});

    // Ctrl+Enter to generate
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            form.dispatchEvent(new Event('submit'));
        }
    });

    // History search
    let historySearchQuery = '';
    historySearch.addEventListener('input', () => {
        historySearchQuery = historySearch.value.trim().toLowerCase();
        historyVisible = historyPageSize;
        renderHistory();
    });

    // Cropping state
    let isCropping = false;
    let cropStartX = 0, cropStartY = 0;
    let cropRect = { x: 0, y: 0, width: 0, height: 0 };
    let isDraggingCrop = false;
    let isResizingCrop = false;
    let resizeHandle = null;
    let dragStartX = 0, dragStartY = 0;
    let initialCropRect = null;

    // Load history on page load
    loadHistory();

    // Load style presets and step guidance
    loadStylePresets();

    // Load available LoRA adapters
    loadAvailableLoras();

    // Load saved settings from localStorage
    loadSavedSettings();

    // Recover any active jobs (running/queued) after page reload
    recoverActiveJobs();

    // Auto-focus prompt field
    document.getElementById('prompt').focus();

    // Lightbox functionality
    function openLightbox(imageSrc, items, index) {
        lightboxImage.src = imageSrc;
        lightboxModal.classList.add('active');
        document.body.style.overflow = 'hidden';
        if (items && index !== undefined) {
            lightboxItems = items;
            lightboxIndex = index;
            lightboxPrev.style.display = items.length > 1 ? '' : 'none';
            lightboxNext.style.display = items.length > 1 ? '' : 'none';
            updateLightboxMetadata(items[index]);
        } else {
            lightboxItems = [];
            lightboxIndex = -1;
            lightboxPrev.style.display = 'none';
            lightboxNext.style.display = 'none';
            lightboxMetadata.innerHTML = '';
        }
    }

    function updateLightboxMetadata(item) {
        if (!item || !lightboxMetadata) return;
        const timeStr = item.generation_time ? ` &middot; ${item.generation_time.toFixed(1)}s` : '';
        const styleName = item.style && stylePresets[item.style] ? stylePresets[item.style].name : null;
        const styleStr = styleName ? ` &middot; ${escapeHtml(styleName)}` : '';
        lightboxMetadata.innerHTML = `
            <div class="meta-prompt">${escapeHtml(item.prompt)}</div>
            <div class="meta-details">${item.width}x${item.height} &middot; ${item.steps} steps &middot; seed ${item.seed || '?'}${styleStr}${timeStr}</div>
        `;
        // Show/hide variations button
        if (item.batch_id) {
            const batchItems = cachedHistory.filter(h => h.batch_id === item.batch_id);
            if (batchItems.length > 1) {
                lightboxVariationsBtn.style.display = '';
                lightboxVariationsBtn.textContent = `Variations (${batchItems.length})`;
            } else {
                lightboxVariationsBtn.style.display = 'none';
            }
        } else {
            lightboxVariationsBtn.style.display = 'none';
        }
    }

    function closeLightbox() {
        lightboxModal.classList.remove('active');
        document.body.style.overflow = '';
        stopSlideshow();
        if (isCropping) {
            exitCropMode();
        }
    }

    function lightboxNavigate(direction) {
        if (lightboxItems.length === 0) return;

        // Sync lightbox items with current history to avoid showing deleted images
        if (cachedHistory.length > 0 && lightboxItems !== cachedHistory) {
            const currentItem = lightboxItems[lightboxIndex];
            lightboxItems = cachedHistory;
            // Try to find current item in updated history
            if (currentItem) {
                const newIndex = lightboxItems.findIndex(h => h.id === currentItem.id);
                if (newIndex !== -1) {
                    lightboxIndex = newIndex;
                } else {
                    // Current item was deleted, start from beginning
                    lightboxIndex = 0;
                }
            }
        }

        if (lightboxItems.length === 0) {
            closeLightbox();
            return;
        }

        lightboxIndex = (lightboxIndex + direction + lightboxItems.length) % lightboxItems.length;
        const item = lightboxItems[lightboxIndex];
        lightboxImage.src = `${item.image_url}?t=${item.created_at}`;
        updateLightboxMetadata(item);
        // Also select in form
        selectHistoryItem(item);
    }

    lightboxPrev.addEventListener('click', (e) => {
        e.stopPropagation();
        lightboxNavigate(-1);
    });
    lightboxNext.addEventListener('click', (e) => {
        e.stopPropagation();
        lightboxNavigate(1);
    });

    // Close lightbox on button click
    lightboxClose.addEventListener('click', closeLightbox);

    // Close lightbox when clicking outside the image
    lightboxModal.addEventListener('click', (e) => {
        if (e.target === lightboxModal) {
            closeLightbox();
        }
    });

    // Keyboard navigation in lightbox
    document.addEventListener('keydown', (e) => {
        if (!lightboxModal.classList.contains('active')) return;
        if (e.key === 'Escape') {
            if (isCropping) {
                exitCropMode();
            } else {
                closeLightbox();
            }
        } else if (e.key === 'ArrowLeft' && !isCropping) {
            lightboxNavigate(-1);
        } else if (e.key === 'ArrowRight' && !isCropping) {
            lightboxNavigate(1);
        } else if (e.key === 'c' && !isCropping) {
            enterCropMode();
        } else if (e.key === 'Enter' && isCropping) {
            applyCrop();
        }
    });

    // Slideshow
    function startSlideshow() {
        if (cachedHistory.length === 0) return;
        openLightbox(
            `${cachedHistory[0].image_url}?t=${cachedHistory[0].created_at}`,
            cachedHistory,
            0
        );
        slideshowBtn.classList.add('active');
        slideshowTimer = setInterval(() => {
            lightboxNavigate(1);
        }, 3000);
    }

    function stopSlideshow() {
        if (slideshowTimer) {
            clearInterval(slideshowTimer);
            slideshowTimer = null;
        }
        slideshowBtn.classList.remove('active');
    }

    slideshowBtn.addEventListener('click', () => {
        if (slideshowTimer) {
            stopSlideshow();
            closeLightbox();
        } else {
            startSlideshow();
        }
    });

    // ========== Cropping Functionality ==========

    function enterCropMode() {
        isCropping = true;
        lightboxModal.classList.add('cropping');
        cropOverlay.style.display = 'block';
        lightboxToolbar.style.display = 'none';
        cropToolbar.style.display = 'flex';
        cropPreset.value = ''; // Reset preset dropdown
        stopSlideshow();

        // Initialize crop selection to center of image
        const imgRect = lightboxImage.getBoundingClientRect();
        const contentRect = lightboxContent.getBoundingClientRect();

        // Image position relative to content container
        const imgOffsetX = imgRect.left - contentRect.left;
        const imgOffsetY = imgRect.top - contentRect.top;

        // Default crop to 50% of image, centered
        const cropW = imgRect.width * 0.5;
        const cropH = imgRect.height * 0.5;
        const cropX = imgOffsetX + (imgRect.width - cropW) / 2;
        const cropY = imgOffsetY + (imgRect.height - cropH) / 2;

        cropRect = { x: cropX, y: cropY, width: cropW, height: cropH };
        updateCropSelection();
    }

    function exitCropMode() {
        isCropping = false;
        lightboxModal.classList.remove('cropping');
        cropOverlay.style.display = 'none';
        lightboxToolbar.style.display = 'flex';
        cropToolbar.style.display = 'none';
    }

    function updateCropSelection() {
        cropSelection.style.left = `${cropRect.x}px`;
        cropSelection.style.top = `${cropRect.y}px`;
        cropSelection.style.width = `${cropRect.width}px`;
        cropSelection.style.height = `${cropRect.height}px`;

        // Update crop dimensions display (in actual pixels)
        const imgRect = lightboxImage.getBoundingClientRect();
        const contentRect = lightboxContent.getBoundingClientRect();
        const imgOffsetX = imgRect.left - contentRect.left;
        const imgOffsetY = imgRect.top - contentRect.top;

        const scaleX = lightboxImage.naturalWidth / imgRect.width;
        const scaleY = lightboxImage.naturalHeight / imgRect.height;

        const actualW = Math.round(Math.min(lightboxImage.naturalWidth, cropRect.width * scaleX));
        const actualH = Math.round(Math.min(lightboxImage.naturalHeight, cropRect.height * scaleY));

        cropDimensions.textContent = `${actualW} × ${actualH}`;
    }

    async function applyCrop() {
        // Get the image and crop coordinates
        const imgRect = lightboxImage.getBoundingClientRect();
        const contentRect = lightboxContent.getBoundingClientRect();

        // Calculate image offset within content
        const imgOffsetX = imgRect.left - contentRect.left;
        const imgOffsetY = imgRect.top - contentRect.top;

        // Calculate crop coordinates relative to image
        const scaleX = lightboxImage.naturalWidth / imgRect.width;
        const scaleY = lightboxImage.naturalHeight / imgRect.height;

        const cropX = Math.max(0, (cropRect.x - imgOffsetX) * scaleX);
        const cropY = Math.max(0, (cropRect.y - imgOffsetY) * scaleY);
        const cropW = Math.min(lightboxImage.naturalWidth - cropX, cropRect.width * scaleX);
        const cropH = Math.min(lightboxImage.naturalHeight - cropY, cropRect.height * scaleY);

        // Create canvas and crop the image
        const canvas = document.createElement('canvas');
        canvas.width = cropW;
        canvas.height = cropH;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(lightboxImage, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

        // Get cropped image data URL
        const croppedDataUrl = canvas.toDataURL('image/png');

        // Get original image metadata from lightbox item (if available)
        let originalItem = null;
        if (lightboxItems.length > 0 && lightboxIndex >= 0 && lightboxIndex < lightboxItems.length) {
            originalItem = lightboxItems[lightboxIndex];
        }

        // Save cropped image to history
        try {
            const response = await fetch('/save-crop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: croppedDataUrl,
                    original_id: originalItem?.id || null,
                    prompt: originalItem ? `Cropped: ${originalItem.prompt}` : 'Cropped image',
                    seed: originalItem?.seed || null,
                    width: Math.round(cropW),
                    height: Math.round(cropH),
                }),
            });
            if (response.ok) {
                // Refresh history to show the new cropped image
                await loadHistory();
            }
        } catch (err) {
            console.error('Failed to save crop to history:', err);
        }

        // Add to first empty reference slot
        const slot = findFirstEmptySlot();
        setReferenceImage(slot, croppedDataUrl);

        exitCropMode();
        closeLightbox();
    }

    // Crop button handler
    lightboxCropBtn.addEventListener('click', enterCropMode);
    cropCancelBtn.addEventListener('click', exitCropMode);
    cropApplyBtn.addEventListener('click', applyCrop);

    // Crop preset handler
    cropPreset.addEventListener('change', () => {
        const value = cropPreset.value;
        if (!value) return;

        const [targetW, targetH] = value.split('x').map(Number);
        const imgRect = lightboxImage.getBoundingClientRect();
        const contentRect = lightboxContent.getBoundingClientRect();
        const imgOffsetX = imgRect.left - contentRect.left;
        const imgOffsetY = imgRect.top - contentRect.top;

        // Calculate scale from image natural size to display size
        const scaleX = imgRect.width / lightboxImage.naturalWidth;
        const scaleY = imgRect.height / lightboxImage.naturalHeight;

        // Target size in display pixels
        let cropW = targetW * scaleX;
        let cropH = targetH * scaleY;

        // Constrain to image bounds
        cropW = Math.min(cropW, imgRect.width);
        cropH = Math.min(cropH, imgRect.height);

        // Center the crop
        const cropX = imgOffsetX + (imgRect.width - cropW) / 2;
        const cropY = imgOffsetY + (imgRect.height - cropH) / 2;

        cropRect = { x: cropX, y: cropY, width: cropW, height: cropH };
        updateCropSelection();
    });

    // Lightbox use as input button
    lightboxUseBtn.addEventListener('click', () => {
        const slot = findFirstEmptySlot();
        loadImageIntoSlot(lightboxImage.src, slot);
        closeLightbox();
    });

    // Crop selection dragging
    cropSelection.addEventListener('mousedown', (e) => {
        if (e.target.classList.contains('crop-handle')) {
            isResizingCrop = true;
            resizeHandle = e.target.dataset.handle;
        } else {
            isDraggingCrop = true;
        }
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        initialCropRect = { ...cropRect };
        e.stopPropagation();
    });

    // Crop overlay click to create new selection
    cropOverlay.addEventListener('mousedown', (e) => {
        if (e.target === cropOverlay) {
            const contentRect = lightboxContent.getBoundingClientRect();
            cropStartX = e.clientX - contentRect.left;
            cropStartY = e.clientY - contentRect.top;
            cropRect = { x: cropStartX, y: cropStartY, width: 0, height: 0 };
            updateCropSelection();
            isDraggingCrop = false;
            isResizingCrop = true;
            resizeHandle = 'se';
            dragStartX = e.clientX;
            dragStartY = e.clientY;
            initialCropRect = { ...cropRect };
        }
    });

    document.addEventListener('mousemove', (e) => {
        if (!isCropping) return;

        const imgRect = lightboxImage.getBoundingClientRect();
        const contentRect = lightboxContent.getBoundingClientRect();
        const imgOffsetX = imgRect.left - contentRect.left;
        const imgOffsetY = imgRect.top - contentRect.top;

        const dx = e.clientX - dragStartX;
        const dy = e.clientY - dragStartY;

        if (isDraggingCrop && initialCropRect) {
            // Move the selection
            let newX = initialCropRect.x + dx;
            let newY = initialCropRect.y + dy;

            // Constrain to image bounds
            newX = Math.max(imgOffsetX, Math.min(newX, imgOffsetX + imgRect.width - cropRect.width));
            newY = Math.max(imgOffsetY, Math.min(newY, imgOffsetY + imgRect.height - cropRect.height));

            cropRect.x = newX;
            cropRect.y = newY;
            updateCropSelection();
        } else if (isResizingCrop && initialCropRect) {
            // Resize the selection
            let { x, y, width, height } = initialCropRect;

            if (resizeHandle.includes('e')) {
                width = Math.max(50, initialCropRect.width + dx);
            }
            if (resizeHandle.includes('w')) {
                const newW = Math.max(50, initialCropRect.width - dx);
                x = initialCropRect.x + initialCropRect.width - newW;
                width = newW;
            }
            if (resizeHandle.includes('s')) {
                height = Math.max(50, initialCropRect.height + dy);
            }
            if (resizeHandle.includes('n')) {
                const newH = Math.max(50, initialCropRect.height - dy);
                y = initialCropRect.y + initialCropRect.height - newH;
                height = newH;
            }

            // Constrain to image bounds
            x = Math.max(imgOffsetX, x);
            y = Math.max(imgOffsetY, y);
            width = Math.min(width, imgOffsetX + imgRect.width - x);
            height = Math.min(height, imgOffsetY + imgRect.height - y);

            cropRect = { x, y, width, height };
            updateCropSelection();
        }
    });

    document.addEventListener('mouseup', () => {
        isDraggingCrop = false;
        isResizingCrop = false;
        resizeHandle = null;
        initialCropRect = null;
    });

    // ========== End Cropping Functionality ==========

    // Make output area images clickable
    outputArea.addEventListener('click', (e) => {
        if (e.target.tagName === 'IMG') {
            openLightbox(e.target.src);
        }
    });

    // Cancel button handler
    cancelBtn.addEventListener('click', async () => {
        if (!currentJobId) return;

        try {
            await fetch(`/cancel/${currentJobId}`, { method: 'POST' });
        } catch (err) {
            console.error('Failed to cancel:', err);
        }

        // Close the event source and reset UI
        if (currentEventSource) {
            currentEventSource.close();
            currentEventSource = null;
        }
        currentJobId = null;
        setGenerating(false);
        showProgress('Cancelled', 0);

        // Chain to next server-queued job if any
        if (generationQueue.length > 0) {
            recoverNextServerJob();
        }
    });

    // Queue functionality
    function getFormParams() {
        // Collect non-null reference images
        const refs = referenceImageData.filter(img => img !== null);
        const guidance = parseFloat(guidanceInput.value);
        return {
            prompt: document.getElementById('prompt').value.trim(),
            width: parseInt(widthSelect.value),
            height: parseInt(heightSelect.value),
            steps: parseInt(stepsInput.value),
            seed: document.getElementById('seed').value.trim() || null,
            referenceImages: refs.length > 0 ? refs : null,
            style: stylePresetSelect ? stylePresetSelect.value || null : null,
            guidance: guidance > 0 ? guidance : null,
            schedule: scheduleSelect.value || null,
            lora: loraSelect ? loraSelect.value || null : null,
            lora_scale: loraScaleInput ? parseFloat(loraScaleInput.value) : 1.0,
        };
    }

    async function addToQueue(params) {
        // Submit to server immediately — it will queue the job if busy
        const { prompt, width, height, steps, seed, referenceImages, style, guidance, schedule, batchId, lora, lora_scale } = params;
        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt,
                    width,
                    height,
                    steps,
                    seed: seed ? parseInt(seed) : null,
                    reference_images: referenceImages,
                    show_steps: showStepsCheckbox.checked,
                    style: style || null,
                    guidance: guidance || null,
                    schedule: schedule || null,
                    batch_id: batchId || null,
                    lora: lora || null,
                    lora_scale: lora_scale || 1.0,
                }),
            });
            const data = await response.json();
            if (!response.ok) {
                showError(data.error || 'Failed to queue job');
                return;
            }
            // Track in local queue display with server's job_id
            generationQueue.push({
                id: data.job_id,
                prompt,
                width,
                height,
                steps,
                seed,
                serverJob: true,
            });
            renderQueue();
        } catch (err) {
            showError(`Failed to queue: ${err.message}`);
        }
        // Clear prompt for next entry
        document.getElementById('prompt').value = '';
    }

    function removeFromQueue(id) {
        generationQueue = generationQueue.filter(item => item.id !== id);
        renderQueue();
    }

    function renderQueue() {
        queueCount.textContent = generationQueue.length;
        queueSection.style.display = generationQueue.length > 0 ? 'block' : 'none';

        queueList.innerHTML = '';
        generationQueue.forEach((item) => {
            const li = document.createElement('li');
            li.className = 'queue-item';
            li.innerHTML = `
                <span class="queue-item-prompt" title="${escapeHtml(item.prompt)}">${escapeHtml(item.prompt)}</span>
                <span class="queue-item-info">${item.width}x${item.height}, ${item.steps} steps</span>
                <button class="queue-item-remove" data-id="${item.id}" title="Remove">&times;</button>
            `;
            queueList.appendChild(li);
        });

        // Add remove/cancel handlers
        queueList.querySelectorAll('.queue-item-remove').forEach(btn => {
            btn.addEventListener('click', async () => {
                const id = btn.dataset.id;
                const item = generationQueue.find(item => String(item.id) === id);
                if (item && item.serverJob) {
                    // Server-managed job: cancel via API
                    try {
                        await fetch(`/cancel/${id}`, { method: 'POST' });
                    } catch (err) {
                        console.error('Failed to cancel queued job:', err);
                    }
                    generationQueue = generationQueue.filter(item => String(item.id) !== id);
                    renderQueue();
                } else {
                    removeFromQueue(parseInt(id));
                }
            });
        });
    }

    async function processNextInQueue() {
        if (generationQueue.length === 0) return;

        const item = generationQueue.shift();
        renderQueue();

        try {
            await generateImage(item);
        } catch (error) {
            showError(`Queue item failed: ${error.message}`);
        }

        // Process next item if any
        if (generationQueue.length > 0) {
            processNextInQueue();
        }
    }

    // Handle aspect preset change - populate width/height dropdowns
    aspectPreset.addEventListener('change', () => {
        const value = aspectPreset.value;
        if (value && value.includes('x')) {
            const [w, h] = value.split('x').map(Number);
            widthSelect.value = w;
            heightSelect.value = h;
        }
    });

    // Update steps display and hint
    stepsInput.addEventListener('input', () => {
        const steps = parseInt(stepsInput.value);
        stepsValue.textContent = steps;
        updateStepHint(steps);
    });

    // Style preset change handler
    if (stylePresetSelect) {
        stylePresetSelect.addEventListener('change', () => {
            const value = stylePresetSelect.value;
            if (value && stylePresets[value]) {
                const preset = stylePresets[value];
                if (styleHint) {
                    styleHint.textContent = preset.description;
                }
                // Update recommended steps
                if (preset.recommended_steps) {
                    stepsInput.value = preset.recommended_steps;
                    stepsValue.textContent = preset.recommended_steps;
                    updateStepHint(preset.recommended_steps);
                }
            } else {
                if (styleHint) {
                    styleHint.textContent = '';
                }
            }
            // Reset enhanced state when style changes
            isPromptEnhanced = false;
            originalPrompt = '';
            if (enhanceBtn) {
                enhanceBtn.classList.remove('enhanced');
                enhanceBtn.title = 'Enhance prompt with style modifiers';
            }
        });
    }

    // Enhance button handler
    if (enhanceBtn) {
        enhanceBtn.addEventListener('click', () => {
            if (isPromptEnhanced) {
                revertPrompt();
            } else {
                enhancePrompt();
            }
        });
    }

    // Reset enhanced state when prompt is manually edited
    if (promptTextarea) {
        promptTextarea.addEventListener('input', () => {
            if (isPromptEnhanced) {
                // User is editing, mark as no longer enhanced
                isPromptEnhanced = false;
                originalPrompt = '';
                if (enhanceBtn) {
                    enhanceBtn.classList.remove('enhanced');
                    enhanceBtn.title = 'Enhance prompt with style modifiers';
                }
            }
        });
    }

    // Reference image slots handling
    const refSlots = document.querySelectorAll('.reference-image-slot');
    const refInputs = document.querySelectorAll('.ref-image-input');
    const refPreviews = document.querySelectorAll('.ref-preview');
    const refClearBtns = document.querySelectorAll('.btn-clear-ref');

    // Update visibility of clear all button
    function updateClearAllButton() {
        const hasAnyImage = referenceImageData.some(img => img !== null);
        clearAllImagesBtn.style.display = hasAnyImage ? 'block' : 'none';
    }

    // Set reference image for a specific slot
    function setReferenceImage(slot, dataUrl, autoSetDimensions = true) {
        referenceImageData[slot] = dataUrl;
        const preview = refPreviews[slot];
        const clearBtn = refClearBtns[slot];
        const slotEl = refSlots[slot];

        preview.innerHTML = `<img src="${dataUrl}" alt="Reference image ${slot + 1}">`;
        clearBtn.style.display = 'block';
        slotEl.classList.add('has-image');
        slotEl.setAttribute('draggable', 'true');
        updateClearAllButton();

        // Auto-set dimensions based on first image (slot 0)
        if (autoSetDimensions && slot === 0) {
            autoSetDimensionsFromImage(dataUrl);
        }
    }

    // Clear reference image for a specific slot and compact remaining images
    function clearReferenceImage(slot) {
        // Shift all images after this slot up to fill the gap
        for (let i = slot; i < 3; i++) {
            referenceImageData[i] = referenceImageData[i + 1];
        }
        referenceImageData[3] = null; // Last slot is now empty
        renderAllSlots();
    }

    // Find first empty slot
    function findFirstEmptySlot() {
        for (let i = 0; i < 4; i++) {
            if (referenceImageData[i] === null) return i;
        }
        return 0; // Default to first slot if all full
    }

    // Auto-set dimensions from image and reset steps to default
    function autoSetDimensionsFromImage(dataUrl) {
        const img = new Image();
        img.onload = function() {
            const w = img.width;
            const h = img.height;

            // Find closest available dimension options
            const widthOptions = Array.from(widthSelect.options).map(o => parseInt(o.value));
            const heightOptions = Array.from(heightSelect.options).map(o => parseInt(o.value));

            // Find closest width (divisible by 16)
            let closestWidth = widthOptions.reduce((prev, curr) =>
                Math.abs(curr - w) < Math.abs(prev - w) ? curr : prev
            );

            // Find closest height (divisible by 16)
            let closestHeight = heightOptions.reduce((prev, curr) =>
                Math.abs(curr - h) < Math.abs(prev - h) ? curr : prev
            );

            setDimensions(closestWidth, closestHeight);

            // Reset steps to default (4)
            stepsInput.value = 4;
            stepsValue.textContent = '4';
        };
        img.src = dataUrl;
    }

    // Handle image upload for each slot
    refInputs.forEach((input, slot) => {
        input.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    setReferenceImage(slot, event.target.result);
                };
                reader.readAsDataURL(file);
            }
        });
    });

    // Drag and drop support for each slot (file drops + internal reorder)
    let dragSourceSlot = null;

    function renderAllSlots() {
        for (let i = 0; i < 4; i++) {
            const preview = refPreviews[i];
            const clearBtn = refClearBtns[i];
            const slotEl = refSlots[i];
            const input = refInputs[i];

            if (referenceImageData[i]) {
                preview.innerHTML = `<img src="${referenceImageData[i]}" alt="Reference image ${i + 1}">`;
                clearBtn.style.display = 'block';
                slotEl.classList.add('has-image');
                slotEl.setAttribute('draggable', 'true');
            } else {
                preview.innerHTML = `<span>Image ${i + 1}</span>`;
                clearBtn.style.display = 'none';
                slotEl.classList.remove('has-image');
                slotEl.removeAttribute('draggable');
            }
            input.value = '';
        }
        updateClearAllButton();
    }

    refSlots.forEach((slotEl, slot) => {
        slotEl.addEventListener('dragstart', (e) => {
            if (!referenceImageData[slot]) { e.preventDefault(); return; }
            dragSourceSlot = slot;
            slotEl.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/x-ref-slot', String(slot));
        });

        slotEl.addEventListener('dragend', () => {
            slotEl.classList.remove('dragging');
            dragSourceSlot = null;
        });

        slotEl.addEventListener('dragover', (e) => {
            e.preventDefault();
            if (dragSourceSlot !== null) {
                slotEl.classList.add('drag-over');
            } else {
                refPreviews[slot].style.borderColor = 'var(--accent)';
                refPreviews[slot].style.backgroundColor = 'rgba(99, 102, 241, 0.05)';
            }
        });

        slotEl.addEventListener('dragleave', () => {
            slotEl.classList.remove('drag-over');
            refPreviews[slot].style.borderColor = '';
            refPreviews[slot].style.backgroundColor = '';
        });

        slotEl.addEventListener('drop', (e) => {
            e.preventDefault();
            slotEl.classList.remove('drag-over');
            refPreviews[slot].style.borderColor = '';
            refPreviews[slot].style.backgroundColor = '';

            // Internal reorder: swap source and target slot data
            const sourceSlotStr = e.dataTransfer.getData('text/x-ref-slot');
            if (sourceSlotStr !== '') {
                const src = parseInt(sourceSlotStr);
                if (src !== slot) {
                    const tmp = referenceImageData[src];
                    referenceImageData[src] = referenceImageData[slot];
                    referenceImageData[slot] = tmp;
                    renderAllSlots();
                }
                return;
            }

            // Check for image URL from drag
            const imageUrl = e.dataTransfer.getData('text/plain');
            if (imageUrl && imageUrl.startsWith('/image/')) {
                loadImageIntoSlot(imageUrl, slot);
                return;
            }

            // Fall back to file drop
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    setReferenceImage(slot, event.target.result);
                };
                reader.readAsDataURL(file);
            }
        });
    });

    // Clear button for each slot
    refClearBtns.forEach((btn, slot) => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            clearReferenceImage(slot);
        });
    });

    // Clear all images button
    clearAllImagesBtn.addEventListener('click', () => {
        referenceImageData = [null, null, null, null];
        renderAllSlots();
    });

    // Remix button - regenerate with same settings but new seed
    remixBtn.addEventListener('click', () => {
        if (!currentGeneration) return;

        // Fill form with current generation params
        document.getElementById('prompt').value = currentGeneration.prompt;
        setDimensions(currentGeneration.width, currentGeneration.height);
        document.getElementById('steps').value = currentGeneration.steps;
        stepsValue.textContent = currentGeneration.steps;
        document.getElementById('seed').value = ''; // Clear seed for random

        // Trigger generation
        form.dispatchEvent(new Event('submit'));
    });

    // Use current output image as img2img input
    useAsInputBtn.addEventListener('click', () => {
        const img = outputArea.querySelector('img');
        if (!img) return;
        const slot = findFirstEmptySlot();
        loadImageIntoSlot(img.src, slot);
        // Reset steps to default when using as input
        stepsInput.value = 4;
        stepsValue.textContent = '4';
    });

    // Fetch an image URL and set it into a specific slot
    async function loadImageIntoSlot(url, slot) {
        try {
            const response = await fetch(url);
            const blob = await response.blob();
            const reader = new FileReader();
            reader.onload = () => {
                setReferenceImage(slot, reader.result);
            };
            reader.readAsDataURL(blob);
        } catch (err) {
            console.error('Failed to load image into slot:', err);
        }
    }

    // Backwards compatibility: load image as input (into first empty slot)
    async function loadImageAsInput(url) {
        const slot = findFirstEmptySlot();
        await loadImageIntoSlot(url, slot);
    }

    // Helper to set dimensions and update preset selector
    function setDimensions(width, height) {
        widthSelect.value = width;
        heightSelect.value = height;

        // Try to find a matching preset
        const presetValue = `${width}x${height}`;
        const option = aspectPreset.querySelector(`option[value="${presetValue}"]`);
        if (option) {
            aspectPreset.value = presetValue;
        } else {
            aspectPreset.value = '';
        }
    }

    // Generate image with given parameters (returns a promise)
    function generateImage(params) {
        return new Promise((resolve, reject) => {
            // Close any existing event source
            if (currentEventSource) {
                currentEventSource.close();
                currentEventSource = null;
            }

            const { prompt, width, height, steps, seed, referenceImages, style, guidance, schedule, batchId, lora, lora_scale } = params;

            // Store current generation params for remix
            currentGeneration = { prompt, width, height, steps };

            // Disable form and show progress
            setGenerating(true);
            progressPrompt.textContent = prompt;
            progressPrompt.title = prompt;
            showProgress('Starting generation...', 0);

            // Clear output area to show placeholder while generating
            outputArea.innerHTML = `
                <div class="placeholder">
                    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <circle cx="8.5" cy="8.5" r="1.5"></circle>
                        <polyline points="21 15 16 10 5 21"></polyline>
                    </svg>
                    <p>Generating...</p>
                </div>
            `;
            imageInfo.style.display = 'none';

            fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt,
                    width,
                    height,
                    steps,
                    seed: seed ? parseInt(seed) : null,
                    reference_images: referenceImages,
                    show_steps: showStepsCheckbox.checked,
                    style: style || null,
                    guidance: guidance || null,
                    schedule: schedule || null,
                    batch_id: batchId || null,
                    lora: lora || null,
                    lora_scale: lora_scale || 1.0,
                }),
            })
            .then(response => response.json().then(data => ({ response, data })))
            .then(({ response, data }) => {
                if (!response.ok) {
                    throw new Error(data.error || 'Generation failed');
                }
                // Connect to SSE for progress updates
                currentJobId = data.job_id;
                connectToProgress(data.job_id, steps, resolve, reject);
            })
            .catch(error => {
                showError(error.message);
                setGenerating(false);
                reject(error);
            });
        });
    }

    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const params = getFormParams();
        if (!params.prompt) {
            showError('Please enter a prompt');
            return;
        }

        // Generate batch ID for variation groups
        const batchId = variationsCount > 1 ? crypto.randomUUID().slice(0, 12) : null;
        if (batchId) params.batchId = batchId;

        const extraVariations = variationsCount - 1;

        if (isGenerating) {
            // Already generating — queue everything (main first, then extras)
            addToQueue(params);
            for (let i = 0; i < extraVariations; i++) {
                addToQueue({ ...params, seed: null });
            }
        } else {
            // Generate main job first, then queue extras after it's submitted
            try {
                const mainPromise = generateImage(params);
                // Queue extras after main job's fetch has been sent
                for (let i = 0; i < extraVariations; i++) {
                    addToQueue({ ...params, seed: null });
                }
                await mainPromise;
            } catch (error) {
                // Error already shown
            }
            // Chain to next server-queued job if any were added while generating
            if (generationQueue.length > 0) {
                recoverNextServerJob();
            }
        }
    });

    function connectToProgress(jobId, totalSteps, onComplete, onError) {
        currentEventSource = new EventSource(`/progress/${jobId}`);

        currentEventSource.addEventListener('progress', (e) => {
            const data = JSON.parse(e.data);
            const percent = (data.progress / data.total_steps) * 100;
            const stepTime = data.step_time ? ` (${data.step_time.toFixed(1)}s/step)` : '';
            const remaining = data.total_steps - data.progress;
            const eta = (data.step_time && remaining > 0) ? ` ~${(data.step_time * remaining).toFixed(0)}s left` : '';
            showProgress(`${data.phase} - Step ${data.progress}/${data.total_steps}${stepTime}${eta}`, percent);
        });

        currentEventSource.addEventListener('step_image', (e) => {
            const data = JSON.parse(e.data);
            const url = `${data.image_url}?t=${Date.now()}`;
            outputArea.innerHTML = `
                <div class="step-preview">
                    <img src="${url}" alt="Step ${data.step} preview">
                    <span class="step-badge">Step ${data.step}/${data.total}</span>
                </div>
            `;
        });

        currentEventSource.addEventListener('status', (e) => {
            const data = JSON.parse(e.data);
            if (data.seed) {
                seedDisplay.textContent = `Seed: ${data.seed}`;
                if (currentGeneration) {
                    currentGeneration.seed = data.seed;
                }
            }
            if (data.phase_done) {
                const phaseTime = data.phase_time ? ` (${data.phase_time.toFixed(1)}s)` : '';
                showProgress(`${data.phase_done} done${phaseTime}`, null);
            } else if (data.phase) {
                const elapsed = data.elapsed ? ` - ${data.elapsed.toFixed(1)}s elapsed` : '';
                showProgress(`${data.phase}...${elapsed}`, null);
            } else if (data.status === 'running') {
                showProgress('Starting...', 0);
            }
        });

        currentEventSource.addEventListener('queued', (e) => {
            const data = JSON.parse(e.data);
            showProgress(`Queued (position ${data.position})`, 0);
        });

        currentEventSource.addEventListener('queue_position', (e) => {
            const data = JSON.parse(e.data);
            showProgress(`Queued (position ${data.position}/${data.total})`, 0);
        });

        currentEventSource.addEventListener('complete', (e) => {
            const data = JSON.parse(e.data);
            const totalTime = data.total_time ? ` in ${data.total_time.toFixed(1)}s` : '';
            showProgress(`Complete${totalTime}!`, 100);
            displayImage(data.image_url);
            setGenerating(false);
            currentEventSource.close();
            currentEventSource = null;
            currentJobId = null;
            // Refresh history
            loadHistory();
            if (onComplete) onComplete();
        });

        currentEventSource.addEventListener('error', (e) => {
            let errorMsg = 'Connection lost';
            if (e.data) {
                const data = JSON.parse(e.data);
                errorMsg = data.message;
                if (errorMsg !== 'Cancelled') {
                    showError(errorMsg);
                }
            } else {
                showError(errorMsg);
            }
            setGenerating(false);
            currentEventSource.close();
            currentEventSource = null;
            currentJobId = null;
            if (onError) onError(new Error(errorMsg));
        });

        currentEventSource.onerror = () => {
            currentJobId = null;
            // Connection error - try to get final status
            setTimeout(async () => {
                try {
                    const response = await fetch(`/status/${jobId}`);
                    const data = await response.json();
                    if (data.status === 'complete' && data.image_url) {
                        showProgress('Complete!', 100);
                        displayImage(data.image_url);
                        loadHistory();
                        if (onComplete) onComplete();
                    } else if (data.status === 'error') {
                        showError(data.error || 'Generation failed');
                        if (onError) onError(new Error(data.error || 'Generation failed'));
                    }
                } catch (err) {
                    if (onError) onError(err);
                }
                setGenerating(false);
            }, 500);

            if (currentEventSource) {
                currentEventSource.close();
                currentEventSource = null;
            }
        };
    }

    async function recoverActiveJobs() {
        try {
            const response = await fetch('/active-jobs');
            if (!response.ok) return;
            const data = await response.json();

            // Recover queued jobs into the queue display
            if (data.queued && data.queued.length > 0) {
                data.queued.forEach(job => {
                    generationQueue.push({
                        id: job.id,
                        prompt: job.prompt,
                        width: job.width,
                        height: job.height,
                        steps: job.steps,
                        seed: job.seed,
                        serverJob: true,
                    });
                });
                renderQueue();
            }

            // Recover running job
            if (data.running) {
                const job = data.running;
                currentJobId = job.id;

                setGenerating(true);
                progressPrompt.textContent = job.prompt;
                progressPrompt.title = job.prompt;

                if (job.progress > 0 && job.total_steps > 0) {
                    const percent = (job.progress / job.total_steps) * 100;
                    showProgress(
                        `${job.phase} - Step ${job.progress}/${job.total_steps}`,
                        percent
                    );
                } else {
                    showProgress(`${job.phase || 'Reconnecting'}...`, 0);
                }

                currentGeneration = {
                    prompt: job.prompt,
                    width: job.width,
                    height: job.height,
                    steps: job.steps,
                };

                connectToProgress(job.id, job.steps,
                    () => {
                        // When running job completes, check for next queued job
                        recoverNextServerJob();
                    },
                    () => {
                        recoverNextServerJob();
                    }
                );
            }
        } catch (err) {
            console.error('Failed to recover active jobs:', err);
        }
    }

    async function recoverNextServerJob() {
        // Remove the first server-managed queue item (server has already started it)
        const idx = generationQueue.findIndex(item => item.serverJob);
        if (idx !== -1) {
            const item = generationQueue.splice(idx, 1)[0];
            renderQueue();

            // Give the server a moment to start the job, then check its status
            await new Promise(resolve => setTimeout(resolve, 500));

            try {
                const response = await fetch(`/status/${item.id}`);
                if (!response.ok) return;
                const job = await response.json();

                if (job.status === 'complete') {
                    loadHistory();
                    recoverNextServerJob();
                    return;
                }
                if (job.status === 'error' || job.status === 'cancelled') {
                    recoverNextServerJob();
                    return;
                }

                // Job is running — track it
                currentJobId = item.id;
                setGenerating(true);
                progressPrompt.textContent = item.prompt;
                progressPrompt.title = item.prompt;
                showProgress(`${job.phase || 'Starting'}...`, 0);

                currentGeneration = {
                    prompt: item.prompt,
                    width: item.width,
                    height: item.height,
                    steps: item.steps,
                };

                connectToProgress(item.id, item.steps,
                    () => recoverNextServerJob(),
                    () => recoverNextServerJob()
                );
            } catch (err) {
                console.error('Failed to track next queued job:', err);
            }
        } else if (generationQueue.length > 0) {
            // Only client-queued items remain, process them normally
            processNextInQueue();
        }
    }

    function setGenerating(generating) {
        isGenerating = generating;
        if (generating) {
            generateBtn.textContent = 'Queue';
        } else {
            generateBtn.textContent = variationsCount > 1 ? `Generate ${variationsCount}` : 'Generate';
        }
        form.classList.toggle('generating', generating);
        progressSection.style.display = generating ? 'block' : 'none';
    }

    function showProgress(text, percent) {
        progressText.textContent = text;
        if (percent !== null) {
            progressFill.style.width = `${percent}%`;
        }
    }

    function showError(message) {
        // Remove any existing error
        const existingError = document.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }

        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        form.appendChild(errorDiv);

        // Remove after 5 seconds
        setTimeout(() => errorDiv.remove(), 5000);
    }

    function displayImage(imageUrl) {
        // Add timestamp to bust cache
        const url = `${imageUrl}?t=${Date.now()}`;
        outputArea.innerHTML = `<img src="${url}" alt="Generated image">`;
        imageInfo.style.display = 'flex';
    }

    // Load style presets from server
    // =========================================================================
    // LoRA Support
    // =========================================================================

    async function loadAvailableLoras() {
        try {
            const resp = await fetch('/available-loras');
            const data = await resp.json();
            availableLoras = data.loras || [];
            curatedLoras = data.curated || [];
            rebuildLoraDropdown();
        } catch (e) {
            console.error('Failed to load LoRAs:', e);
        }
    }

    function rebuildLoraDropdown() {
        if (!loraSelect) return;
        const prev = loraSelect.value;
        loraSelect.innerHTML = '<option value="">None</option>';
        availableLoras.forEach(lora => {
            const opt = document.createElement('option');
            opt.value = lora.filename;
            opt.textContent = lora.name + (lora.size_mb ? ` (${lora.size_mb} MB)` : '');
            loraSelect.appendChild(opt);
        });
        if (prev && availableLoras.some(l => l.filename === prev)) {
            loraSelect.value = prev;
        }
    }

    function openLoraPanel() {
        renderLoraPanel();
        loraPanel.style.display = 'flex';
        loraPanelBackdrop.style.display = 'block';
    }

    function closeLoraPanel() {
        loraPanel.style.display = 'none';
        loraPanelBackdrop.style.display = 'none';
    }

    function renderLoraPanel() {
        if (!loraPanelBody) return;
        loraPanelBody.innerHTML = '';

        // Curated section
        if (curatedLoras.length > 0) {
            const h = document.createElement('div');
            h.className = 'lora-section-header';
            h.textContent = 'Klein LoRA Collection';
            loraPanelBody.appendChild(h);

            curatedLoras.forEach(lora => {
                const card = document.createElement('div');
                card.className = 'lora-card';
                card.dataset.id = lora.id;

                const badges = lora.models.map(m =>
                    `<span class="lora-badge lora-badge-${m}">${m.toUpperCase()}</span>`
                ).join('');
                const trigger = lora.trigger
                    ? `<span class="lora-trigger" title="Add this word to your prompt">${lora.trigger}</span>`
                    : '';
                const dlState = activeDownloads[lora.id];
                const sourceBadge = lora.source === 'civitai'
                    ? '<span class="lora-source-badge lora-source-civitai">Civitai</span>'
                    : '<span class="lora-source-badge lora-source-hf">HuggingFace</span>';
                const strengthLabel = lora.strength
                    ? `<span class="lora-trigger" title="Recommended strength">⚡ ${lora.strength}</span>`
                    : '';
                let actionHtml;
                if (dlState && !dlState.done && !dlState.error) {
                    actionHtml = `<div class="lora-progress-wrap"><div class="lora-progress-bar" style="width:${dlState.percent || 0}%"></div></div>`;
                } else if (lora.downloaded) {
                    actionHtml = `<button class="lora-use-btn" data-filename="${lora.filename}">Use</button>`;
                } else if (lora.source === 'civitai') {
                    actionHtml = `<button class="lora-dl-btn"
                        data-id="${lora.id}"
                        data-source="civitai"
                        data-civitai-model-id="${lora.civitai_model_id}"
                        data-civitai-version-filter="${lora.civitai_version_filter || ''}"
                        data-filename="${lora.filename}">Download</button>`;
                } else {
                    actionHtml = `<button class="lora-dl-btn"
                        data-id="${lora.id}"
                        data-source="huggingface"
                        data-repo="${lora.repo}"
                        data-filename="${lora.filename}">Download</button>`;
                }

                card.innerHTML = `
                    <div class="lora-card-main">
                        <div class="lora-card-info">
                            <div class="lora-card-name-row">
                                <span class="lora-card-name">${lora.name}</span>
                                ${sourceBadge}
                            </div>
                            <span class="lora-card-desc">${lora.description}</span>
                            <div class="lora-card-tags">${badges}${trigger}${strengthLabel}</div>
                        </div>
                        <div class="lora-card-action">${actionHtml}</div>
                    </div>
                `;
                loraPanelBody.appendChild(card);
            });
        }

        // Local LoRAs not in curated list
        const localOnly = availableLoras.filter(l =>
            !curatedLoras.some(c => c.filename === l.filename)
        );
        if (localOnly.length > 0) {
            const h = document.createElement('div');
            h.className = 'lora-section-header';
            h.textContent = 'Local LoRAs';
            loraPanelBody.appendChild(h);
            localOnly.forEach(lora => {
                const card = document.createElement('div');
                card.className = 'lora-card';
                card.innerHTML = `
                    <div class="lora-card-main">
                        <div class="lora-card-info">
                            <span class="lora-card-name">${lora.name}</span>
                            <span class="lora-card-desc">${lora.size_mb} MB</span>
                        </div>
                        <div class="lora-card-action">
                            <button class="lora-use-btn" data-filename="${lora.filename}">Use</button>
                        </div>
                    </div>
                `;
                loraPanelBody.appendChild(card);
            });
        }

        if (curatedLoras.length === 0 && localOnly.length === 0) {
            loraPanelBody.innerHTML = '<p class="lora-empty">No LoRAs found. Click Download to get one.</p>';
        }

        // Wire up buttons
        loraPanelBody.querySelectorAll('.lora-dl-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                startLoraDownload(btn.dataset.id, {
                    source: btn.dataset.source || 'huggingface',
                    repo: btn.dataset.repo,
                    filename: btn.dataset.filename,
                    civitai_model_id: btn.dataset.civitaiModelId ? Number(btn.dataset.civitaiModelId) : undefined,
                    civitai_version_filter: btn.dataset.civitaiVersionFilter || undefined,
                });
            });
        });
        loraPanelBody.querySelectorAll('.lora-use-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                if (loraSelect) {
                    loraSelect.value = btn.dataset.filename;
                    loraSelect.dispatchEvent(new Event('change'));
                }
                closeLoraPanel();
            });
        });
    }

    async function startLoraDownload(dlId, entry) {
        activeDownloads[dlId] = { percent: 0, done: false, error: null };
        renderLoraPanel();

        try {
            const resp = await fetch('/download-lora', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    id: dlId,
                    source: entry.source || 'huggingface',
                    repo: entry.repo,
                    filename: entry.filename,
                    civitai_model_id: entry.civitai_model_id,
                    civitai_version_filter: entry.civitai_version_filter,
                }),
            });
            const data = await resp.json();
            if (!resp.ok || data.error) {
                activeDownloads[dlId].error = data.error || 'Download failed';
                renderLoraPanel();
                return;
            }
            if (data.already_exists) {
                delete activeDownloads[dlId];
                await loadAvailableLoras();
                renderLoraPanel();
                return;
            }
        } catch (e) {
            activeDownloads[dlId].error = String(e);
            renderLoraPanel();
            return;
        }

        // Poll progress
        const interval = setInterval(async () => {
            try {
                const pr = await fetch(`/download-lora/progress/${dlId}`);
                const state = await pr.json();
                activeDownloads[dlId].percent = state.percent || 0;
                activeDownloads[dlId].done = state.done || false;
                activeDownloads[dlId].error = state.error || null;

                // Update progress bar in place without full re-render
                const card = loraPanelBody ? loraPanelBody.querySelector(`[data-id="${dlId}"]`) : null;
                if (card) {
                    const bar = card.querySelector('.lora-progress-bar');
                    if (bar) bar.style.width = `${state.percent || 0}%`;
                }

                if (state.done || state.error) {
                    clearInterval(interval);
                    delete activeDownloads[dlId];
                    await loadAvailableLoras();
                    renderLoraPanel();
                }
            } catch (e) {
                clearInterval(interval);
            }
        }, 500);
    }

    // LoRA event listeners
    if (loraSelect) {
        loraSelect.addEventListener('change', () => {
            if (loraScaleGroup) {
                loraScaleGroup.style.display = loraSelect.value ? 'block' : 'none';
            }
        });
    }
    if (loraScaleInput && loraScaleValue) {
        loraScaleInput.addEventListener('input', () => {
            loraScaleValue.textContent = parseFloat(loraScaleInput.value).toFixed(2);
        });
    }
    if (loraBrowseBtn) loraBrowseBtn.addEventListener('click', openLoraPanel);
    if (loraPanelClose) loraPanelClose.addEventListener('click', closeLoraPanel);
    if (loraPanelBackdrop) loraPanelBackdrop.addEventListener('click', closeLoraPanel);

    async function loadStylePresets() {
        try {
            const response = await fetch('/style-presets');
            const data = await response.json();
            stylePresets = data.presets || {};
            stepGuidance = data.step_guidance || {};

            // Populate style preset dropdown with optgroups by category
            if (stylePresetSelect) {
                const groups = {};
                Object.entries(stylePresets).forEach(([key, preset]) => {
                    const cat = preset.category || 'Other';
                    if (!groups[cat]) groups[cat] = [];
                    groups[cat].push({ key, preset });
                });
                Object.entries(groups).forEach(([category, items]) => {
                    const optgroup = document.createElement('optgroup');
                    optgroup.label = category;
                    items.forEach(({ key, preset }) => {
                        const option = document.createElement('option');
                        option.value = key;
                        option.textContent = preset.name;
                        optgroup.appendChild(option);
                    });
                    stylePresetSelect.appendChild(optgroup);
                });
            }

            // Restore saved style preset if pending
            if (stylePresetSelect && stylePresetSelect.dataset.pendingStyle) {
                stylePresetSelect.value = stylePresetSelect.dataset.pendingStyle;
                delete stylePresetSelect.dataset.pendingStyle;
            }

            // Update initial step hint
            updateStepHint(parseInt(stepsInput.value));
        } catch (error) {
            console.error('Failed to load style presets:', error);
        }
    }

    // Update step hint based on current value
    function updateStepHint(steps) {
        if (stepsHint && stepGuidance[steps]) {
            stepsHint.textContent = `(${stepGuidance[steps].label})`;
        }
    }

    // Enhance prompt with selected style
    async function enhancePrompt() {
        const prompt = promptTextarea.value.trim();
        if (!prompt) return;

        const style = stylePresetSelect ? stylePresetSelect.value : '';

        try {
            const response = await fetch('/enhance-prompt', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    style: style || null,
                    auto_enhance: !style // Only auto-enhance if no style selected
                })
            });
            const data = await response.json();

            if (data.enhanced && data.enhanced !== prompt) {
                originalPrompt = prompt;
                promptTextarea.value = data.enhanced;
                isPromptEnhanced = true;
                if (enhanceBtn) {
                    enhanceBtn.classList.add('enhanced');
                    enhanceBtn.title = 'Prompt enhanced! Click to revert';
                }

                // Update steps if recommended
                if (data.recommended_steps && data.recommended_steps !== parseInt(stepsInput.value)) {
                    stepsInput.value = data.recommended_steps;
                    stepsValue.textContent = data.recommended_steps;
                    updateStepHint(data.recommended_steps);
                }
            }
        } catch (error) {
            console.error('Failed to enhance prompt:', error);
        }
    }

    // Revert enhanced prompt to original
    function revertPrompt() {
        if (isPromptEnhanced && originalPrompt) {
            promptTextarea.value = originalPrompt;
            isPromptEnhanced = false;
            originalPrompt = '';
            if (enhanceBtn) {
                enhanceBtn.classList.remove('enhanced');
                enhanceBtn.title = 'Enhance prompt with style modifiers';
            }
        }
    }

    async function loadHistory() {
        try {
            const response = await fetch('/history');
            const history = await response.json();

            cachedHistory = history;

            // Sync lightbox items with history if lightbox is open
            if (lightboxModal.classList.contains('active') && lightboxItems.length > 0) {
                // Check if current lightbox image still exists
                const currentItem = lightboxItems[lightboxIndex];
                if (currentItem) {
                    const stillExists = history.some(h => h.id === currentItem.id);
                    if (!stillExists) {
                        // Current image was deleted, close lightbox
                        closeLightbox();
                    } else {
                        // Update lightbox items to current history
                        lightboxItems = history;
                        // Find new index of current item
                        const newIndex = history.findIndex(h => h.id === currentItem.id);
                        if (newIndex !== -1) {
                            lightboxIndex = newIndex;
                        }
                    }
                }
            }

            if (history.length === 0) {
                historySection.style.display = 'none';
                return;
            }

            historySection.style.display = 'block';
            renderHistory();
        } catch (err) {
            console.error('Failed to load history:', err);
        }
    }

    function renderHistory() {
        const history = historySearchQuery
            ? cachedHistory.filter(h => h.prompt.toLowerCase().includes(historySearchQuery))
            : cachedHistory;
        historyGrid.innerHTML = '';

        // Build batch groups lookup
        const batchGroups = {};
        history.forEach(item => {
            if (item.batch_id) {
                if (!batchGroups[item.batch_id]) batchGroups[item.batch_id] = [];
                batchGroups[item.batch_id].push(item);
            }
        });
        const seenBatches = new Set();

        const visible = history.slice(0, historyVisible);
        for (let i = 0; i < visible.length; i++) {
            const item = visible[i];
            const div = document.createElement('div');
            div.className = 'history-item';

            // Batch variation indicator
            if (item.batch_id && batchGroups[item.batch_id]?.length > 1) {
                div.classList.add('batch-item');
            }

            // Compare selection indicator
            if (selectedForCompare.some(s => s.id === item.id)) {
                div.classList.add('selected-compare');
            }

            // Use thumbnail for grid, full image for lightbox
            const thumbUrl = item.thumb_url ? `${item.thumb_url}?t=${item.created_at}` : `${item.image_url}?t=${item.created_at}`;
            const fullUrl = `${item.image_url}?t=${item.created_at}`;
            div.setAttribute('draggable', 'true');
            div.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('text/plain', item.image_url);
                e.dataTransfer.effectAllowed = 'copy';
            });

            // Build inner HTML with optional batch badge
            let badgeHtml = '';
            if (item.batch_id && batchGroups[item.batch_id]?.length > 1 && !seenBatches.has(item.batch_id)) {
                seenBatches.add(item.batch_id);
                const count = batchGroups[item.batch_id].length;
                badgeHtml = `<div class="batch-badge" data-batch-id="${item.batch_id}" title="View all ${count} variations">${count}</div>`;
            }

            div.innerHTML = `
                ${badgeHtml}
                <img src="${thumbUrl}" alt="${escapeHtml(item.prompt)}" data-fullsize="${fullUrl}" draggable="false">
                <button class="history-item-delete" title="Delete">&times;</button>
                <div class="history-item-overlay">
                    <div class="history-item-prompt">${escapeHtml(item.prompt)}</div>
                </div>
            `;

            // Batch badge click → open compare modal with batch
            const badge = div.querySelector('.batch-badge');
            if (badge) {
                badge.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const batchId = badge.dataset.batchId;
                    const batchItems = cachedHistory.filter(h => h.batch_id === batchId);
                    openCompareModal(batchItems);
                });
            }

            const deleteBtn = div.querySelector('.history-item-delete');
            deleteBtn.addEventListener('click', async (e) => {
                e.stopPropagation();
                try {
                    const resp = await fetch(`/history/${item.id}`, { method: 'DELETE' });
                    if (!resp.ok) {
                        console.error('Delete failed:', resp.status, await resp.text());
                    }
                    loadHistory();
                } catch (err) {
                    console.error('Failed to delete:', err);
                }
            });
            const itemIndex = i;
            div.addEventListener('click', (e) => {
                if (e.target.closest('.history-item-delete')) return;
                if (e.target.closest('.batch-badge')) return;
                // Shift+click toggles compare selection
                if (e.shiftKey) {
                    toggleCompareSelection(item);
                    return;
                }
                selectHistoryItem(item);
            });
            div.addEventListener('dblclick', (e) => {
                if (e.target.closest('.history-item-delete')) return;
                if (e.target.closest('.batch-badge')) return;
                openLightbox(
                    `${item.image_url}?t=${item.created_at}`,
                    cachedHistory,
                    cachedHistory.indexOf(item)
                );
            });
            historyGrid.appendChild(div);
        }

        // Remove existing show-more button
        const existing = historySection.querySelector('.history-show-more');
        if (existing) existing.remove();

        // Add show more / show less buttons
        if (history.length > historyPageSize) {
            const btnRow = document.createElement('div');
            btnRow.className = 'history-show-more';

            if (historyVisible < history.length) {
                const moreBtn = document.createElement('button');
                moreBtn.className = 'btn-secondary';
                moreBtn.textContent = `Show more (${history.length - historyVisible} remaining)`;
                moreBtn.addEventListener('click', () => {
                    historyVisible += historyPageSize;
                    renderHistory();
                });
                btnRow.appendChild(moreBtn);
            }

            if (historyVisible > historyPageSize) {
                const lessBtn = document.createElement('button');
                lessBtn.className = 'btn-secondary';
                lessBtn.textContent = 'Show less';
                lessBtn.addEventListener('click', () => {
                    historyVisible = historyPageSize;
                    renderHistory();
                    historySection.scrollIntoView({ behavior: 'smooth' });
                });
                btnRow.appendChild(lessBtn);
            }

            historySection.appendChild(btnRow);
        }
    }

    function selectHistoryItem(item) {
        // Display the image
        displayImage(item.image_url);
        seedDisplay.textContent = `Seed: ${item.seed}`;

        // Fill form with the item's parameters (base prompt, no style suffix)
        document.getElementById('prompt').value = item.prompt;
        setDimensions(item.width, item.height);
        document.getElementById('steps').value = item.steps;
        stepsValue.textContent = item.steps;
        document.getElementById('seed').value = item.seed || '';

        // Restore style preset if item was generated with one
        if (stylePresetSelect) {
            stylePresetSelect.value = item.style || '';
        }

        // Restore LoRA settings
        if (loraSelect) {
            loraSelect.value = item.lora || '';
            if (loraScaleGroup) {
                loraScaleGroup.style.display = loraSelect.value ? '' : 'none';
            }
        }
        if (loraScaleInput && item.lora_scale != null) {
            loraScaleInput.value = item.lora_scale;
            if (loraScaleValue) loraScaleValue.textContent = Number(item.lora_scale).toFixed(2);
        }

        // Store for remix
        currentGeneration = {
            prompt: item.prompt,
            width: item.width,
            height: item.height,
            steps: item.steps,
            seed: item.seed,
        };
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ========== Copy Seed Button ==========
    copySeedBtn.addEventListener('click', async () => {
        if (!currentGeneration || !currentGeneration.seed) return;

        try {
            await navigator.clipboard.writeText(String(currentGeneration.seed));
            copySeedBtn.classList.add('copied');
            setTimeout(() => copySeedBtn.classList.remove('copied'), 1500);
        } catch (err) {
            console.error('Failed to copy seed:', err);
        }
    });

    // ========== Download Button ==========
    downloadBtn.addEventListener('click', () => {
        const img = outputArea.querySelector('img');
        if (!img) return;

        const link = document.createElement('a');
        // Remove cache-busting query param for cleaner filename
        const imageUrl = img.src.split('?')[0];
        link.href = imageUrl;

        // Generate filename from prompt
        const prompt = currentGeneration?.prompt || 'image';
        const seed = currentGeneration?.seed || 'unknown';
        const safePrompt = prompt.substring(0, 50).replace(/[^a-zA-Z0-9]/g, '_');
        link.download = `flux_${safePrompt}_${seed}.png`;

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // ========== Paste Image from Clipboard (Ctrl+V) ==========
    document.addEventListener('paste', async (e) => {
        // Don't intercept paste in text inputs
        if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') {
            return;
        }

        const items = e.clipboardData?.items;
        if (!items) return;

        for (const item of items) {
            if (item.type.startsWith('image/')) {
                e.preventDefault();
                const blob = item.getAsFile();
                if (!blob) continue;

                const reader = new FileReader();
                reader.onload = (event) => {
                    const slot = findFirstEmptySlot();
                    setReferenceImage(slot, event.target.result);
                };
                reader.readAsDataURL(blob);
                break;
            }
        }
    });

    // ========== LocalStorage Settings Persistence ==========
    const SETTINGS_KEY = 'flux_settings';

    function loadSavedSettings() {
        try {
            const saved = localStorage.getItem(SETTINGS_KEY);
            if (!saved) return;

            const settings = JSON.parse(saved);
            if (settings.width) widthSelect.value = settings.width;
            if (settings.height) heightSelect.value = settings.height;
            if (settings.steps) {
                stepsInput.value = settings.steps;
                stepsValue.textContent = settings.steps;
            }
            if (settings.showSteps !== undefined) {
                showStepsCheckbox.checked = settings.showSteps;
            }
            if (settings.style && stylePresetSelect) {
                // Defer style restore until presets are loaded
                stylePresetSelect.dataset.pendingStyle = settings.style;
            }
            if (settings.variations && settings.variations > 1) {
                variationsCount = Math.min(8, Math.max(1, settings.variations));
                updateVariationsDisplay();
            }
            if (settings.guidance !== undefined && guidanceInput) {
                guidanceInput.value = settings.guidance;
                const val = parseFloat(settings.guidance);
                guidanceValue.textContent = val === 0 ? 'auto' : val.toFixed(1);
            }
            if (settings.schedule && scheduleSelect) {
                scheduleSelect.value = settings.schedule;
            }
        } catch (err) {
            console.error('Failed to load settings:', err);
        }
    }

    function saveSettings() {
        try {
            const settings = {
                width: widthSelect.value,
                height: heightSelect.value,
                steps: stepsInput.value,
                showSteps: showStepsCheckbox.checked,
                style: stylePresetSelect ? stylePresetSelect.value : '',
                variations: variationsCount,
                guidance: guidanceInput ? guidanceInput.value : '0',
                schedule: scheduleSelect ? scheduleSelect.value : '',
            };
            localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
        } catch (err) {
            console.error('Failed to save settings:', err);
        }
    }

    // Save settings when they change
    widthSelect.addEventListener('change', saveSettings);
    heightSelect.addEventListener('change', saveSettings);
    stepsInput.addEventListener('change', saveSettings);
    showStepsCheckbox.addEventListener('change', saveSettings);
    if (guidanceInput) guidanceInput.addEventListener('change', saveSettings);
    if (scheduleSelect) scheduleSelect.addEventListener('change', saveSettings);
    if (stylePresetSelect) stylePresetSelect.addEventListener('change', saveSettings);

    // ========== Prompt History ==========
    const PROMPT_HISTORY_KEY = 'flux_prompt_history';
    const MAX_PROMPT_HISTORY = 20;

    function getPromptHistory() {
        try {
            const saved = localStorage.getItem(PROMPT_HISTORY_KEY);
            return saved ? JSON.parse(saved) : [];
        } catch {
            return [];
        }
    }

    function savePromptToHistory(prompt) {
        if (!prompt || prompt.trim().length === 0) return;
        prompt = prompt.trim();

        try {
            let history = getPromptHistory();
            // Remove if already exists (to move to top)
            history = history.filter(p => p !== prompt);
            // Add to beginning
            history.unshift(prompt);
            // Limit size
            if (history.length > MAX_PROMPT_HISTORY) {
                history = history.slice(0, MAX_PROMPT_HISTORY);
            }
            localStorage.setItem(PROMPT_HISTORY_KEY, JSON.stringify(history));
        } catch (err) {
            console.error('Failed to save prompt history:', err);
        }
    }

    function renderPromptHistory() {
        const history = getPromptHistory();
        promptHistoryDropdown.innerHTML = '';

        if (history.length === 0) {
            promptHistoryDropdown.innerHTML = '<div class="prompt-history-empty">No recent prompts</div>';
            return;
        }

        history.forEach(prompt => {
            const item = document.createElement('div');
            item.className = 'prompt-history-item';
            item.textContent = prompt;
            item.title = prompt;
            item.addEventListener('click', () => {
                document.getElementById('prompt').value = prompt;
                promptHistoryDropdown.classList.remove('active');
                clearTemplate();
            });
            promptHistoryDropdown.appendChild(item);
        });
    }

    // Toggle prompt history dropdown
    promptHistoryBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        renderPromptHistory();
        promptHistoryDropdown.classList.toggle('active');
        templateDropdown.classList.remove('active');
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!promptHistoryDropdown.contains(e.target) && e.target !== promptHistoryBtn) {
            promptHistoryDropdown.classList.remove('active');
        }
    });

    // Save prompt when generation starts (hook into form submit)
    const originalFormSubmit = form.onsubmit;
    form.addEventListener('submit', () => {
        const prompt = document.getElementById('prompt').value;
        savePromptToHistory(prompt);
    });

    // ========== Lightbox Download Button (#16) ==========
    lightboxDownloadBtn.addEventListener('click', () => {
        if (lightboxItems.length === 0 || lightboxIndex < 0) return;
        const item = lightboxItems[lightboxIndex];
        const link = document.createElement('a');
        link.href = item.image_url;
        const safePrompt = item.prompt.substring(0, 50).replace(/[^a-zA-Z0-9]/g, '_');
        link.download = `flux_${safePrompt}_${item.seed || 'unknown'}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // ========== Lightbox Delete Button (#16) ==========
    lightboxDeleteBtn.addEventListener('click', async () => {
        if (lightboxItems.length === 0 || lightboxIndex < 0) return;
        const item = lightboxItems[lightboxIndex];
        try {
            const resp = await fetch(`/history/${item.id}`, { method: 'DELETE' });
            if (!resp.ok) return;
            await loadHistory();
            // Navigate to next or close
            if (cachedHistory.length === 0) {
                closeLightbox();
            } else {
                lightboxItems = cachedHistory;
                lightboxIndex = Math.min(lightboxIndex, lightboxItems.length - 1);
                const next = lightboxItems[lightboxIndex];
                lightboxImage.src = `${next.image_url}?t=${next.created_at}`;
                updateLightboxMetadata(next);
            }
        } catch (err) {
            console.error('Failed to delete from lightbox:', err);
        }
    });

    // ========== Lightbox Variations Button (#14) ==========
    lightboxVariationsBtn.addEventListener('click', () => {
        if (lightboxItems.length === 0 || lightboxIndex < 0) return;
        const item = lightboxItems[lightboxIndex];
        if (!item.batch_id) return;
        const batchItems = cachedHistory.filter(h => h.batch_id === item.batch_id);
        if (batchItems.length > 1) {
            closeLightbox();
            openCompareModal(batchItems);
        }
    });

    // ========== Compare Modal (#14, #15) ==========
    function openCompareModal(items) {
        compareGrid.innerHTML = '';
        const n = Math.min(items.length, 4);
        compareGrid.className = 'compare-grid cols-' + n;

        items.slice(0, 4).forEach(item => {
            const div = document.createElement('div');
            div.className = 'compare-item';
            const timeStr = item.generation_time ? `${item.generation_time.toFixed(1)}s` : '';
            div.innerHTML = `
                <img src="${item.image_url}?t=${item.created_at}" alt="${escapeHtml(item.prompt)}">
                <div class="compare-item-info">
                    Seed: ${item.seed || '?'}${timeStr ? ' &middot; ' + timeStr : ''}
                </div>
            `;
            // Click image to open in lightbox
            div.querySelector('img').addEventListener('click', () => {
                closeCompareModal();
                const idx = cachedHistory.findIndex(h => h.id === item.id);
                openLightbox(
                    `${item.image_url}?t=${item.created_at}`,
                    cachedHistory,
                    idx >= 0 ? idx : 0
                );
            });
            compareGrid.appendChild(div);
        });

        compareModal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeCompareModal() {
        compareModal.classList.remove('active');
        document.body.style.overflow = '';
    }

    compareClose.addEventListener('click', closeCompareModal);
    compareModal.addEventListener('click', (e) => {
        if (e.target === compareModal) closeCompareModal();
    });

    // Escape key closes compare modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && compareModal.classList.contains('active')) {
            closeCompareModal();
        }
    });

    // ========== Compare Selection in History (#15) ==========
    function toggleCompareSelection(item) {
        const idx = selectedForCompare.findIndex(s => s.id === item.id);
        if (idx >= 0) {
            selectedForCompare.splice(idx, 1);
        } else if (selectedForCompare.length < 4) {
            selectedForCompare.push(item);
        }
        renderHistory();
        updateCompareBar();
    }

    function updateCompareBar() {
        if (selectedForCompare.length >= 2) {
            compareBar.style.display = 'flex';
            compareBarText.textContent = `${selectedForCompare.length} selected`;
        } else {
            compareBar.style.display = selectedForCompare.length > 0 ? 'flex' : 'none';
            compareBarText.textContent = `${selectedForCompare.length} selected (pick ${2 - selectedForCompare.length} more)`;
        }
        compareBarBtn.disabled = selectedForCompare.length < 2;
        compareBarBtn.style.opacity = selectedForCompare.length < 2 ? '0.5' : '1';
    }

    compareBarBtn.addEventListener('click', () => {
        if (selectedForCompare.length >= 2) {
            openCompareModal(selectedForCompare);
        }
    });

    compareBarClear.addEventListener('click', () => {
        selectedForCompare = [];
        renderHistory();
        updateCompareBar();
    });

    // ========== Prompt Templates (#17) ==========

    const BUILTIN_TEMPLATES = [
        {
            id: 'portrait',
            name: 'Portrait',
            pattern: 'Portrait of a {subject}, {lighting} lighting, {background} background',
            category: 'Characters',
        },
        {
            id: 'character',
            name: 'Character Design',
            pattern: '{character_type} character, {clothing}, {pose}, {expression} expression',
            category: 'Characters',
        },
        {
            id: 'landscape',
            name: 'Landscape',
            pattern: 'A {location} landscape, {time_of_day}, {weather} weather, {mood} atmosphere',
            category: 'Scenes',
        },
        {
            id: 'interior',
            name: 'Interior',
            pattern: '{room_type} interior, {style} style, {lighting} lighting, {mood} atmosphere',
            category: 'Scenes',
        },
        {
            id: 'style-transfer',
            name: 'Style Transfer',
            pattern: 'A {subject} in the style of {artist}, {medium} art, {mood} mood',
            category: 'Artistic',
        },
        {
            id: 'abstract',
            name: 'Abstract',
            pattern: 'Abstract {subject}, {colors} colors, {composition} composition',
            category: 'Artistic',
        },
        {
            id: 'product',
            name: 'Product Shot',
            pattern: '{product} product photography, {angle} angle, {lighting} lighting, {background} background',
            category: 'Commercial',
        },
        {
            id: 'creature',
            name: 'Fantasy Creature',
            pattern: 'A {creature} with {features}, in a {environment} environment',
            category: 'Fantasy',
        },
    ];

    function extractVariables(pattern) {
        const matches = pattern.matchAll(/\{(\w+)\}/g);
        return [...new Set(Array.from(matches, m => m[1]))];
    }

    function loadUserTemplates() {
        try {
            const saved = localStorage.getItem(TEMPLATE_STORAGE_KEY);
            userTemplates = saved ? JSON.parse(saved) : [];
        } catch (err) {
            console.error('Failed to load user templates:', err);
            userTemplates = [];
        }
    }

    function saveUserTemplates() {
        try {
            localStorage.setItem(TEMPLATE_STORAGE_KEY, JSON.stringify(userTemplates));
        } catch (err) {
            console.error('Failed to save user templates:', err);
        }
    }

    function renderTemplateDropdown() {
        templateDropdownContent.innerHTML = '';

        // Group by category
        const categories = {};
        BUILTIN_TEMPLATES.forEach(t => {
            if (!categories[t.category]) categories[t.category] = [];
            categories[t.category].push({ ...t, type: 'builtin' });
        });
        if (userTemplates.length > 0) {
            categories['Your Templates'] = userTemplates.map(t => ({ ...t, type: 'user' }));
        }

        for (const [category, templates] of Object.entries(categories)) {
            const header = document.createElement('div');
            header.className = 'template-section-header';
            header.textContent = category;
            templateDropdownContent.appendChild(header);

            templates.forEach(template => {
                const item = document.createElement('div');
                item.className = 'template-item';

                const name = document.createElement('span');
                name.className = 'template-item-name';
                name.textContent = template.name;
                item.appendChild(name);

                const pattern = document.createElement('span');
                pattern.className = 'template-item-pattern';
                pattern.textContent = template.pattern.length > 70
                    ? template.pattern.substring(0, 70) + '...'
                    : template.pattern;
                item.appendChild(pattern);

                if (template.type === 'user') {
                    const del = document.createElement('button');
                    del.className = 'template-item-delete';
                    del.innerHTML = '&times;';
                    del.title = 'Delete template';
                    del.addEventListener('click', (e) => {
                        e.stopPropagation();
                        deleteUserTemplate(template.id);
                    });
                    item.appendChild(del);
                }

                item.addEventListener('click', () => {
                    applyTemplate(template);
                    templateDropdown.classList.remove('active');
                });

                templateDropdownContent.appendChild(item);
            });
        }
    }

    function applyTemplate(template) {
        const vars = extractVariables(template.pattern);
        activeTemplate = { ...template, variables: vars };
        templateVariableValues = {};
        promptTextarea.value = template.pattern;

        // Clear enhanced state
        isPromptEnhanced = false;
        originalPrompt = '';
        if (enhanceBtn) enhanceBtn.classList.remove('enhanced');

        renderTemplateVariables();
    }

    function renderTemplateVariables() {
        if (!activeTemplate || activeTemplate.variables.length === 0) {
            templateVariablesDiv.style.display = 'none';
            return;
        }

        templateVariablesDiv.style.display = 'block';
        templateVariablesInputs.innerHTML = '';

        activeTemplate.variables.forEach(varName => {
            const div = document.createElement('div');
            div.className = 'template-variable-input';

            const label = document.createElement('label');
            label.setAttribute('for', 'tvar-' + varName);
            label.textContent = '{' + varName + '}';
            div.appendChild(label);

            const input = document.createElement('input');
            input.type = 'text';
            input.id = 'tvar-' + varName;
            input.placeholder = varName.replace(/_/g, ' ');
            input.value = templateVariableValues[varName] || '';
            input.addEventListener('input', (e) => {
                templateVariableValues[varName] = e.target.value;
                updatePromptFromTemplate();
            });
            div.appendChild(input);

            templateVariablesInputs.appendChild(div);
        });

        // Focus first input
        const first = templateVariablesInputs.querySelector('input');
        if (first) first.focus();
    }

    function updatePromptFromTemplate() {
        if (!activeTemplate) return;
        let composed = activeTemplate.pattern;
        activeTemplate.variables.forEach(varName => {
            const value = templateVariableValues[varName];
            if (value) {
                composed = composed.replace(new RegExp('\\{' + varName + '\\}', 'g'), value);
            }
        });
        promptTextarea.value = composed;
    }

    function clearTemplate() {
        activeTemplate = null;
        templateVariableValues = {};
        templateVariablesDiv.style.display = 'none';
    }

    function saveCurrentAsTemplate() {
        const prompt = promptTextarea.value.trim();
        if (!prompt) return;

        const vars = extractVariables(prompt);
        if (vars.length === 0) {
            alert('Template must contain at least one {variable} placeholder.\n\nExample: A {subject} in {style} style');
            return;
        }

        const name = window.prompt('Template name:');
        if (!name || !name.trim()) return;

        userTemplates.unshift({
            id: Date.now().toString(),
            name: name.trim(),
            pattern: prompt,
        });
        saveUserTemplates();
        renderTemplateDropdown();
    }

    function deleteUserTemplate(id) {
        userTemplates = userTemplates.filter(t => t.id !== id);
        saveUserTemplates();
        renderTemplateDropdown();
        if (activeTemplate && activeTemplate.id === id) clearTemplate();
    }

    // Template event handlers
    templateBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        renderTemplateDropdown();
        templateDropdown.classList.toggle('active');
        promptHistoryDropdown.classList.remove('active');
    });

    templateSaveBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        saveCurrentAsTemplate();
    });

    templateClearBtn.addEventListener('click', clearTemplate);

    // Close template dropdown on outside click
    document.addEventListener('click', (e) => {
        if (!templateDropdown.contains(e.target) && e.target !== templateBtn) {
            templateDropdown.classList.remove('active');
        }
    });

    // Clear template when user manually edits prompt in a way that changes its structure
    promptTextarea.addEventListener('input', () => {
        if (!activeTemplate) return;
        // If the user's text no longer contains any of the original template variables
        // and doesn't match the composed output, they're free-editing
        const currentText = promptTextarea.value;
        const composedText = (() => {
            let c = activeTemplate.pattern;
            activeTemplate.variables.forEach(v => {
                const val = templateVariableValues[v];
                if (val) c = c.replace(new RegExp('\\{' + v + '\\}', 'g'), val);
            });
            return c;
        })();
        if (currentText !== composedText) {
            clearTemplate();
        }
    });

    // Load user templates on startup
    loadUserTemplates();
});
