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

    let currentEventSource = null;
    let referenceImageData = [null, null, null, null]; // Up to 4 reference images
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
        } else {
            lightboxItems = [];
            lightboxIndex = -1;
            lightboxPrev.style.display = 'none';
            lightboxNext.style.display = 'none';
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

        // Process next in queue if any
        if (generationQueue.length > 0) {
            processNextInQueue();
        }
    });

    // Queue functionality
    function getFormParams() {
        // Collect non-null reference images
        const refs = referenceImageData.filter(img => img !== null);
        return {
            prompt: document.getElementById('prompt').value.trim(),
            width: parseInt(widthSelect.value),
            height: parseInt(heightSelect.value),
            steps: parseInt(stepsInput.value),
            seed: document.getElementById('seed').value.trim() || null,
            referenceImages: refs.length > 0 ? refs : null
        };
    }

    function addToQueue(params) {
        // Deep copy the params to preserve reference images even if UI is cleared
        const queueItem = {
            id: Date.now(),
            prompt: params.prompt,
            width: params.width,
            height: params.height,
            steps: params.steps,
            seed: params.seed,
            // Deep copy reference images array (data URLs are strings, already immutable)
            referenceImages: params.referenceImages ? [...params.referenceImages] : null
        };
        generationQueue.push(queueItem);
        renderQueue();
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

        // Add remove handlers
        queueList.querySelectorAll('.queue-item-remove').forEach(btn => {
            btn.addEventListener('click', () => {
                removeFromQueue(parseInt(btn.dataset.id));
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

    // Update steps display
    stepsInput.addEventListener('input', () => {
        stepsValue.textContent = stepsInput.value;
    });

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

        // Re-render all slots
        for (let i = 0; i < 4; i++) {
            const preview = refPreviews[i];
            const clearBtn = refClearBtns[i];
            const slotEl = refSlots[i];
            const input = refInputs[i];

            if (referenceImageData[i]) {
                preview.innerHTML = `<img src="${referenceImageData[i]}" alt="Reference image ${i + 1}">`;
                clearBtn.style.display = 'block';
                slotEl.classList.add('has-image');
            } else {
                preview.innerHTML = `<span>Image ${i + 1}</span>`;
                clearBtn.style.display = 'none';
                slotEl.classList.remove('has-image');
            }
            input.value = '';
        }
        updateClearAllButton();
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

    // Drag and drop support for each slot
    refSlots.forEach((slotEl, slot) => {
        const preview = refPreviews[slot];

        slotEl.addEventListener('dragover', (e) => {
            e.preventDefault();
            preview.style.borderColor = 'var(--accent)';
            preview.style.backgroundColor = 'rgba(99, 102, 241, 0.05)';
        });

        slotEl.addEventListener('dragleave', () => {
            preview.style.borderColor = '';
            preview.style.backgroundColor = '';
        });

        slotEl.addEventListener('drop', (e) => {
            e.preventDefault();
            preview.style.borderColor = '';
            preview.style.backgroundColor = '';

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
        // Clear all slots directly without compacting
        referenceImageData = [null, null, null, null];
        for (let i = 0; i < 4; i++) {
            const preview = refPreviews[i];
            const clearBtn = refClearBtns[i];
            const slotEl = refSlots[i];
            const input = refInputs[i];

            preview.innerHTML = `<span>Image ${i + 1}</span>`;
            clearBtn.style.display = 'none';
            slotEl.classList.remove('has-image');
            input.value = '';
        }
        updateClearAllButton();
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

            const { prompt, width, height, steps, seed, referenceImages } = params;

            // Store current generation params for remix
            currentGeneration = { prompt, width, height, steps };

            // Disable form and show progress
            setGenerating(true);
            progressPrompt.textContent = prompt;
            progressPrompt.title = prompt;
            showProgress('Starting generation...', 0);

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

        if (isGenerating) {
            // Add to queue if already generating
            addToQueue(params);
        } else {
            // Generate immediately
            try {
                await generateImage(params);
            } catch (error) {
                // Error already shown
            }
            // Process queue if any items were added while generating
            if (generationQueue.length > 0) {
                processNextInQueue();
            }
        }
    });

    function connectToProgress(jobId, totalSteps, onComplete, onError) {
        currentEventSource = new EventSource(`/progress/${jobId}`);

        currentEventSource.addEventListener('progress', (e) => {
            const data = JSON.parse(e.data);
            const percent = (data.progress / data.total_steps) * 100;
            const stepTime = data.step_time ? ` (${data.step_time.toFixed(1)}s)` : '';
            const elapsed = data.elapsed ? ` - ${data.elapsed.toFixed(1)}s elapsed` : '';
            showProgress(`${data.phase} - Step ${data.progress}/${data.total_steps}${stepTime}${elapsed}`, percent);
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
            }
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

    function setGenerating(generating) {
        isGenerating = generating;
        generateBtn.textContent = generating ? 'Queue' : 'Generate';
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
        const history = cachedHistory;
        historyGrid.innerHTML = '';

        const visible = history.slice(0, historyVisible);
        for (let i = 0; i < visible.length; i++) {
            const item = visible[i];
            const div = document.createElement('div');
            div.className = 'history-item';
            const imgUrl = `${item.image_url}?t=${item.created_at}`;
            div.setAttribute('draggable', 'true');
            div.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('text/plain', item.image_url);
                e.dataTransfer.effectAllowed = 'copy';
            });
            div.innerHTML = `
                <img src="${imgUrl}" alt="${item.prompt}" data-fullsize="${imgUrl}" draggable="false">
                <button class="history-item-delete" title="Delete">&times;</button>
                <div class="history-item-overlay">
                    <div class="history-item-prompt">${escapeHtml(item.prompt)}</div>
                </div>
            `;
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
                selectHistoryItem(item);
            });
            div.addEventListener('dblclick', (e) => {
                if (e.target.closest('.history-item-delete')) return;
                openLightbox(
                    `${item.image_url}?t=${item.created_at}`,
                    cachedHistory,
                    itemIndex
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

        // Fill form with the item's parameters
        document.getElementById('prompt').value = item.prompt;
        setDimensions(item.width, item.height);
        document.getElementById('steps').value = item.steps;
        stepsValue.textContent = item.steps;
        document.getElementById('seed').value = item.seed || '';

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
});
