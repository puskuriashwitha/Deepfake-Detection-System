(function () {
const $ = (id) => document.getElementById(id);

let localStream = null;
let peerConnection = null;
let roomId = null;
let role = null;

let signalCursor = 0;
let signalTimer = null;
let statusTimer = null;
let detectionTimer = null;

let offerCreated = false;
let processingFrame = false;
let processingSignals = false;

let pendingCandidates = [];

const ICE_SERVERS = {
    iceServers: [
        {
            urls: "stun:stun.l.google.com:19302"
        }
    ]
};

// ------------------------------------
// API HELPER
// ------------------------------------

async function api(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || "Request failed");
    }

    return data;
}

// ------------------------------------
// CAMERA AND MICROPHONE
// ------------------------------------

async function startLocalMedia() {
    if (localStream) return;

    localStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
    });

    $("localVideo").srcObject = localStream;

    $("cameraButton").disabled = true;
    $("micButton").disabled = false;
    $("endCall").disabled = false;

    $("cameraButton").textContent = "Camera Started";
    $("connectionStatus").textContent = "Camera ready";
}

// ------------------------------------
// CREATE PEER CONNECTION
// ------------------------------------

function createPeerConnection() {
    if (peerConnection) return;

    peerConnection = new RTCPeerConnection(ICE_SERVERS);

    // Add local camera and microphone tracks
    localStream.getTracks().forEach(track => {
        peerConnection.addTrack(track, localStream);
    });

    // Receive remote video and audio
    peerConnection.ontrack = event => {
        $("remoteVideo").srcObject = event.streams[0];

        $("remoteStatus").textContent = "Participant connected";
        $("connectionStatus").textContent = "Video call connected";

        startDetection();
    };

    // Send ICE candidates through Flask
    peerConnection.onicecandidate = event => {
        if (event.candidate) {
            sendSignal("candidate", event.candidate.toJSON())
                .catch(error => console.error(error));
        }
    };

    peerConnection.onconnectionstatechange = () => {
        const state = peerConnection.connectionState;

        $("connectionStatus").textContent =
            "Connection: " + state;

        if (state === "connected") {
            $("remoteStatus").textContent = "Connected";
        }

        if (state === "failed" || state === "disconnected") {
            $("remoteStatus").textContent = state;
        }
    };
}

// ------------------------------------
// SIGNALING
// ------------------------------------

async function sendSignal(type, data) {
    await api(`/api/rooms/${roomId}/signal`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            role,
            type,
            data
        })
    });
}

// ------------------------------------
// CREATE ROOM
// ------------------------------------

async function createRoom() {
    try {
        $("connectionStatus").textContent = "Creating room...";

        const result = await api("/api/rooms/create", {
            method: "POST"
        });

        roomId = result.room_id;
        role = "host";

        $("roomInput").value = roomId;
        $("roomIdDisplay").textContent = roomId;
        $("roomInfo").classList.remove("hidden");

        await startLocalMedia();

        createPeerConnection();

        $("connectionStatus").textContent =
            "Room created. Waiting for participant...";

        startSignaling();

    } catch (error) {
        $("connectionStatus").textContent = error.message;
    }
}

// ------------------------------------
// JOIN ROOM
// ------------------------------------

async function joinRoom() {
    try {
        const enteredRoom = $("roomInput").value.trim();

        if (!enteredRoom) {
            alert("Please enter a Room ID.");
            return;
        }

        $("connectionStatus").textContent = "Joining room...";

        await api(`/api/rooms/${encodeURIComponent(enteredRoom)}/join`, {
            method: "POST"
        });

        roomId = enteredRoom;
        role = "guest";

        await startLocalMedia();

        createPeerConnection();

        $("roomIdDisplay").textContent = roomId;
        $("roomInfo").classList.remove("hidden");

        $("connectionStatus").textContent =
            "Joined room. Connecting...";

        startSignaling();

    } catch (error) {
        $("connectionStatus").textContent = error.message;
    }
}

// ------------------------------------
// START SIGNAL POLLING
// ------------------------------------

function startSignaling() {
    if (signalTimer) clearInterval(signalTimer);
    if (statusTimer) clearInterval(statusTimer);

    signalTimer = setInterval(pollSignals, 700);

    if (role === "host") {
        statusTimer = setInterval(checkGuest, 1000);
    }
}

// ------------------------------------
// HOST CHECKS GUEST JOIN
// ------------------------------------

async function checkGuest() {
    if (role !== "host" || offerCreated) return;

    try {
        const result = await api(
            `/api/rooms/${roomId}/status`
        );

        if (result.guest_joined) {
            await createOffer();
        }

    } catch (error) {
        console.error("Room status error:", error);
    }
}

// ------------------------------------
// CREATE WEBRTC OFFER
// ------------------------------------

async function createOffer() {
    if (offerCreated || !peerConnection) return;

    offerCreated = true;

    try {
        $("connectionStatus").textContent =
            "Connecting to participant...";

        const offer = await peerConnection.createOffer();

        await peerConnection.setLocalDescription(offer);

        await sendSignal("offer", {
            type: offer.type,
            sdp: offer.sdp
        });

    } catch (error) {
        offerCreated = false;
        $("connectionStatus").textContent =
            "Offer error: " + error.message;
    }
}

// ------------------------------------
// HANDLE INCOMING SIGNALS
// ------------------------------------

async function pollSignals() {
    if (!roomId || !role || processingSignals) return;

    processingSignals = true;

    try {
        const result = await api(
            `/api/rooms/${roomId}/signals?role=${role}&since=${signalCursor}`
        );

        signalCursor = result.next;

        for (const signal of result.messages) {
            await handleSignal(signal);
        }

    } catch (error) {
        console.error("Signal polling error:", error);

    } finally {
        processingSignals = false;
    }
}

// ------------------------------------
// HANDLE OFFER, ANSWER, CANDIDATE
// ------------------------------------

async function handleSignal(signal) {
    if (!peerConnection) return;

    if (signal.type === "offer" && role === "guest") {

        await peerConnection.setRemoteDescription(
            new RTCSessionDescription(signal.data)
        );

        await flushCandidates();

        const answer = await peerConnection.createAnswer();

        await peerConnection.setLocalDescription(answer);

        await sendSignal("answer", {
            type: answer.type,
            sdp: answer.sdp
        });

    } else if (signal.type === "answer" && role === "host") {

        await peerConnection.setRemoteDescription(
            new RTCSessionDescription(signal.data)
        );

        await flushCandidates();

    } else if (signal.type === "candidate") {

        const candidate = new RTCIceCandidate(signal.data);

        if (peerConnection.remoteDescription) {
            await peerConnection.addIceCandidate(candidate);
        } else {
            pendingCandidates.push(candidate);
        }
    }
}

// ------------------------------------
// PROCESS QUEUED ICE CANDIDATES
// ------------------------------------

async function flushCandidates() {
    while (pendingCandidates.length > 0) {
        const candidate = pendingCandidates.shift();

        await peerConnection.addIceCandidate(candidate);
    }
}

// ------------------------------------
// LIVE DEEPFAKE DETECTION
// ------------------------------------

function startDetection() {
    if (detectionTimer) return;

    detectionTimer = setInterval(
        analyzeRemoteFrame,
        3000
    );

    $("detectionMessage").textContent =
        "Analyzing sampled frames from remote participant...";
}

async function analyzeRemoteFrame() {
    const video = $("remoteVideo");

    if (
        !video.videoWidth ||
        video.readyState < 2 ||
        processingFrame
    ) return;

    processingFrame = true;

    try {
        const canvas = document.createElement("canvas");

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        canvas.getContext("2d").drawImage(
            video,
            0,
            0,
            canvas.width,
            canvas.height
        );

        const blob = await new Promise(resolve =>
            canvas.toBlob(resolve, "image/jpeg", 0.75)
        );

        if (!blob) return;

        const form = new FormData();

        form.append("frame", blob, "remote-frame.jpg");

        const result = await api("/api/analyze-frame", {
            method: "POST",
            body: form
        });

        const confidence = result.confidence;

        let label = result.classification || "NOT AVAILABLE";

        // Display low-confidence predictions as uncertain.
        if (
            confidence !== null &&
            confidence !== undefined &&
            confidence < 0.60
        ) {
            label = "UNCERTAIN";
        }

        $("detectionLabel").textContent = label;

        $("realProbability").textContent =
            "REAL: " +
            (result.real_probability * 100).toFixed(2) + "%";

        $("fakeProbability").textContent =
            "FAKE: " +
            (result.fake_probability * 100).toFixed(2) + "%";

        $("detectionMessage").textContent =
            label === "FAKE"
                ? "Warning: Experimental model flagged this frame."
                : label === "UNCERTAIN"
                ? "Low-confidence prediction. Authenticity is undetermined."
                : "Experimental frame analysis completed.";

    } catch (error) {
        $("detectionMessage").textContent =
            "Frame analysis error: " + error.message;

    } finally {
        processingFrame = false;
    }
}

// ------------------------------------
// MUTE MICROPHONE
// ------------------------------------

function toggleMicrophone() {
    if (!localStream) return;

    const audioTrack = localStream.getAudioTracks()[0];

    if (!audioTrack) return;

    audioTrack.enabled = !audioTrack.enabled;

    $("micButton").textContent =
        audioTrack.enabled
            ? "Mute Microphone"
            : "Unmute Microphone";
}

// ------------------------------------
// END CALL
// ------------------------------------

async function endCall() {
    if (signalTimer) clearInterval(signalTimer);
    if (statusTimer) clearInterval(statusTimer);
    if (detectionTimer) clearInterval(detectionTimer);

    signalTimer = null;
    statusTimer = null;
    detectionTimer = null;

    if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
    }

    if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
        localStream = null;
    }

    $("localVideo").srcObject = null;
    $("remoteVideo").srcObject = null;

    if (roomId && role) {
        try {
            await api(`/api/rooms/${roomId}/leave`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ role })
            });
        } catch (error) {
            console.error(error);
        }
    }

    roomId = null;
    role = null;
    signalCursor = 0;
    offerCreated = false;
    pendingCandidates = [];

    $("cameraButton").disabled = false;
    $("cameraButton").textContent = "Start Camera";

    $("micButton").disabled = true;
    $("micButton").textContent = "Mute Microphone";

    $("endCall").disabled = true;

    $("remoteStatus").textContent = "Waiting for participant";
    $("connectionStatus").textContent = "Call ended";

    $("detectionLabel").textContent = "Waiting for video...";
    $("realProbability").textContent = "REAL: —";
    $("fakeProbability").textContent = "FAKE: —";

    $("detectionMessage").textContent =
        "Detection will begin after the call connects.";

    $("roomInfo").classList.add("hidden");
}

// ------------------------------------
// COPY ROOM ID
// ------------------------------------

async function copyRoomId() {
    if (!roomId) return;

    try {
        await navigator.clipboard.writeText(roomId);
        alert("Room ID copied!");
    } catch {
        alert("Please copy the Room ID manually.");
    }
}

// ------------------------------------
// BUTTON EVENTS
// ------------------------------------

$("createRoom").addEventListener("click", createRoom);
$("joinRoom").addEventListener("click", joinRoom);

$("cameraButton").addEventListener("click", async () => {
    try {
        await startLocalMedia();

        if (!roomId) {
            $("connectionStatus").textContent =
                "Camera ready. Create or join a room.";
        }
    } catch (error) {
        $("connectionStatus").textContent =
            "Camera error: " + error.message;
    }
});

$("micButton").addEventListener("click", toggleMicrophone);
$("endCall").addEventListener("click", endCall);
$("copyRoom").addEventListener("click", copyRoomId);
})();