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
    const inputImage = document.getElementById('input-image');
    const inputPreview = document.getElementById('input-preview');
    const clearImageBtn = document.getElementById('clear-image');
    const remixBtn = document.getElementById('remix-btn');
    const historySection = document.getElementById('history-section');
    const historyGrid = document.getElementById('history-grid');
    const aspectPreset = document.getElementById('aspect-preset');
    const widthSelect = document.getElementById('width');
    const heightSelect = document.getElementById('height');
    const queueSection = document.getElementById('queue-section');
    const queueList = document.getElementById('queue-list');
    const queueCount = document.getElementById('queue-count');

    const lightboxModal = document.getElementById('lightbox-modal');
    const lightboxImage = document.getElementById('lightbox-image');
    const lightboxClose = document.getElementById('lightbox-close');
    const cancelBtn = document.getElementById('cancel-btn');
    const useAsInputBtn = document.getElementById('use-as-input-btn');
    const slideshowBtn = document.getElementById('slideshow-btn');
    const lightboxPrev = document.getElementById('lightbox-prev');
    const lightboxNext = document.getElementById('lightbox-next');

    let currentEventSource = null;
    let inputImageData = null;
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
    }

    function lightboxNavigate(direction) {
        if (lightboxItems.length === 0) return;
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
            closeLightbox();
        } else if (e.key === 'ArrowLeft') {
            lightboxNavigate(-1);
        } else if (e.key === 'ArrowRight') {
            lightboxNavigate(1);
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
        return {
            prompt: document.getElementById('prompt').value.trim(),
            width: parseInt(widthSelect.value),
            height: parseInt(heightSelect.value),
            steps: parseInt(stepsInput.value),
            seed: document.getElementById('seed').value.trim() || null,
            inputImage: inputImageData
        };
    }

    function addToQueue(params) {
        generationQueue.push({
            id: Date.now(),
            ...params
        });
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

    // Handle image upload
    inputImage.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                inputImageData = event.target.result;
                inputPreview.innerHTML = `<img src="${inputImageData}" alt="Input image">`;
                clearImageBtn.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });

    // Drag and drop support for input area
    const fileInputWrapper = document.querySelector('.file-input-wrapper');
    fileInputWrapper.addEventListener('dragover', (e) => {
        e.preventDefault();
        inputPreview.style.borderColor = 'var(--accent)';
        inputPreview.style.backgroundColor = 'rgba(99, 102, 241, 0.05)';
    });
    fileInputWrapper.addEventListener('dragleave', () => {
        inputPreview.style.borderColor = '';
        inputPreview.style.backgroundColor = '';
    });
    fileInputWrapper.addEventListener('drop', (e) => {
        e.preventDefault();
        inputPreview.style.borderColor = '';
        inputPreview.style.backgroundColor = '';

        // Check for image URL from drag
        const imageUrl = e.dataTransfer.getData('text/plain');
        if (imageUrl && imageUrl.startsWith('/image/')) {
            loadImageAsInput(imageUrl);
            return;
        }

        // Fall back to file drop
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (event) => {
                inputImageData = event.target.result;
                inputPreview.innerHTML = `<img src="${inputImageData}" alt="Input image">`;
                clearImageBtn.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });

    // Clear input image
    clearImageBtn.addEventListener('click', () => {
        inputImage.value = '';
        inputImageData = null;
        inputPreview.innerHTML = '<span>Drop image here or click to select</span>';
        clearImageBtn.style.display = 'none';
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
        loadImageAsInput(img.src);
    });

    // Fetch an image URL and set it as img2img input
    async function loadImageAsInput(url) {
        try {
            const response = await fetch(url);
            const blob = await response.blob();
            const reader = new FileReader();
            reader.onload = () => {
                inputImageData = reader.result;
                inputPreview.innerHTML = `<img src="${inputImageData}" alt="Input image">`;
                clearImageBtn.style.display = 'block';
            };
            reader.readAsDataURL(blob);
        } catch (err) {
            console.error('Failed to load image as input:', err);
        }
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

            const { prompt, width, height, steps, seed, inputImage } = params;

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
                    input_image: inputImage,
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
