
const $ = (id) => document.getElementById(id);

let cameraStream = null;
let scanTimer = null;
let isScanning = false;

// -------------------------------
// Load model status
// -------------------------------

async function loadHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();

        const status = $('modelStatus');

        if (status) {
            status.textContent = data.classifier_ready
                ? 'CNN Model Loaded'
                : 'Model Not Available';
        }
    } catch (error) {
        if ($('modelStatus')) {
            $('modelStatus').textContent = 'Offline';
        }
    }
}

// -------------------------------
// Escape HTML
// -------------------------------

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, char => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }[char]));
}

// -------------------------------
// Prediction display
// -------------------------------

function predictionHTML(data) {
    const label = data.classification || 'NOT AVAILABLE';

    const confidence = data.confidence == null
        ? 'N/A'
        : (data.confidence * 100).toFixed(2) + '%';

    const real = data.real_probability == null
        ? 'N/A'
        : (data.real_probability * 100).toFixed(2) + '%';

    const fake = data.fake_probability == null
        ? 'N/A'
        : (data.fake_probability * 100).toFixed(2) + '%';

    const color = label === 'FAKE'
        ? '#dc2626'
        : label === 'REAL'
            ? '#16a34a'
            : '#64748b';

    return `
        <div class="prediction-result">
            <h3>Deepfake Classification</h3>

            <h2 style="color:${color}">
                ${escapeHtml(label)}
            </h2>

            <p>Confidence: <strong>${confidence}</strong></p>
            <p>Real Probability: <strong>${real}</strong></p>
            <p>Fake Probability: <strong>${fake}</strong></p>

            <p style="font-size:13px;color:#d97706">
                Experimental CNN prediction.
                Results require further validation.
            </p>

            <p>${escapeHtml(data.message || '')}</p>
        </div>
    `;
}

// -------------------------------
// VIDEO UPLOAD
// -------------------------------

if ($('videoInput')) {
    $('videoInput').addEventListener('change', event => {
        $('fileName').textContent =
            event.target.files[0]?.name ||
            'Select a file from your computer (max 300 MB)';
    });
}

if ($('uploadForm')) {
    $('uploadForm').addEventListener('submit', async event => {
        event.preventDefault();

        const file = $('videoInput')?.files[0];

        if (!file) {
            alert('Please select a video first.');
            return;
        }

        const button = $('analyzeButton');

        button.disabled = true;
        button.textContent = 'Inspecting video...';

        $('uploadResult').classList.remove('hidden');
        $('uploadResult').textContent =
            'Reading video and analyzing frames. Please wait...';

        try {
            const form = new FormData();
            form.append('video', file);

            const response = await fetch('/api/analyze-video', {
                method: 'POST',
                body: form
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Analysis failed');
            }

            $('uploadResult').innerHTML = `
                <strong>Video inspection completed</strong><br>

                File: ${escapeHtml(data.filename)}<br>

                Duration: ${data.duration_seconds}s ·
                FPS: ${data.fps}<br>

                Resolution: ${escapeHtml(data.resolution)}<br>

                Total frames: ${data.total_frames}<br>

                Sampled frames: ${data.sampled_frames}<br>

                Faces detected:
                ${data.faces_detected_across_samples}<br>

                Frames classified:
                ${data.predicted_frames ?? 0}

                ${predictionHTML(data)}
            `;

            loadHistory();

        } catch (error) {
            $('uploadResult').textContent =
                'Error: ' + error.message;

        } finally {
            button.disabled = false;
            button.textContent = 'Inspect video →';
        }
    });
}

// -------------------------------
// SINGLE PERSON LIVE CAMERA
// -------------------------------

async function startCamera() {
    try {
        if (!navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia) {
            throw new Error(
                'Camera access is unavailable. Open the app using localhost.'
            );
        }

        if (cameraStream) {
            stopCamera();
        }

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });

        const video = $('camera');

        if (!video) {
            throw new Error('Camera video element not found in index.html');
        }

        video.srcObject = cameraStream;
        video.autoplay = true;
        video.playsInline = true;
        video.muted = true;

        await video.play();

        $('cameraPlaceholder').style.display = 'none';
        video.style.display = 'block';

        $('startCamera').disabled = true;
        $('stopCamera').disabled = false;

        $('cameraStatus').textContent =
            'Camera connected. Analyzing frames...';

        if (scanTimer) {
            clearInterval(scanTimer);
        }

        scanTimer = setInterval(scanFrame, 1500);

    } catch (error) {
        $('cameraStatus').textContent =
            'Camera error: ' + error.message +
            '. Check browser camera permission.';

        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
    }
}

// -------------------------------
// STOP CAMERA
// -------------------------------

function stopCamera() {
    if (scanTimer) {
        clearInterval(scanTimer);
        scanTimer = null;
    }

    isScanning = false;

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    const video = $('camera');

    if (video) {
        video.pause();
        video.srcObject = null;
        video.style.display = 'none';
    }

    if ($('cameraPlaceholder')) {
        $('cameraPlaceholder').style.display = 'block';
    }

    if ($('startCamera')) {
        $('startCamera').disabled = false;
    }

    if ($('stopCamera')) {
        $('stopCamera').disabled = true;
    }

    if ($('cameraStatus')) {
        $('cameraStatus').textContent = 'Camera is stopped.';
    }

    if ($('faceCount')) {
        $('faceCount').textContent = 'Faces detected: —';
    }
}

// -------------------------------
// ANALYZE CAMERA FRAME
// -------------------------------

async function scanFrame() {
    const video = $('camera');

    if (!video || !cameraStream || !video.videoWidth || isScanning) {
        return;
    }

    isScanning = true;

    try {
        const canvas = document.createElement('canvas');

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const context = canvas.getContext('2d');

        context.drawImage(video, 0, 0);

        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg', 0.75);
        });

        if (!blob) {
            throw new Error('Could not capture camera frame.');
        }

        const form = new FormData();
        form.append('frame', blob, 'webcam.jpg');

        const response = await fetch('/api/analyze-frame', {
            method: 'POST',
            body: form
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Frame analysis failed.');
        }

        $('faceCount').textContent =
            `Faces detected: ${data.faces_detected}`;

        $('cameraStatus').textContent =
            `Classification: ${data.classification || 'N/A'} · ` +
            `Confidence: ${
                data.confidence == null
                    ? 'N/A'
                    : (data.confidence * 100).toFixed(2) + '%'
            } · ` +
            `${data.message || 'Frame analyzed.'}`;

    } catch (error) {
        if ($('cameraStatus')) {
            $('cameraStatus').textContent =
                'Analysis error: ' + error.message;
        }
    } finally {
        isScanning = false;
    }
}

// -------------------------------
// CAMERA BUTTONS
// -------------------------------

if ($('startCamera')) {
    $('startCamera').addEventListener('click', startCamera);
}

if ($('stopCamera')) {
    $('stopCamera').addEventListener('click', stopCamera);
}

// -------------------------------
// ANALYSIS HISTORY
// -------------------------------

async function loadHistory() {
    try {
        const response = await fetch('/api/history');

        if (!response.ok) {
            throw new Error('History request failed.');
        }

        const rows = await response.json();

        if (!$('historyList')) return;

        $('historyList').innerHTML = rows.length
            ? rows.map(item => `
                <div class="history-row">
                    <div>
                        <b>${escapeHtml(item.filename)}</b><br>
                        <small>${escapeHtml(item.created_at)}</small>
                    </div>

                    <div>
                        ${item.faces_detected} face hits<br>
                        <small>${escapeHtml(item.status)}</small>
                    </div>
                </div>
            `).join('')
            : 'No analyses yet.';

    } catch (error) {
        if ($('historyList')) {
            $('historyList').textContent =
                'Could not load history.';
        }
    }
}

// -------------------------------
// REFRESH HISTORY
// -------------------------------

if ($('refreshHistory')) {
    $('refreshHistory').addEventListener('click', loadHistory);
}

// -------------------------------
// INITIALIZE APPLICATION
// -------------------------------

loadHealth();
loadHistory();

// Stop camera when leaving the page
window.addEventListener('beforeunload', () => {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
    }
});
