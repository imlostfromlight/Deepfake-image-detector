import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Shield, Monitor, ArrowLeft, CheckCircle2, AlertTriangle, Square, Play, Mic, MicOff } from 'lucide-react';

const CAPTURE_INTERVAL_MS = 3000;
const AUDIO_INTERVAL_MS   = 4000;   // voice analysis slightly offset from video
const MAX_HISTORY = 8;

const API = import.meta.env.VITE_API_URL;

const VideoCallDetection = ({ onBack }) => {
    const videoRef        = useRef(null);
    const streamRef       = useRef(null);
    const intervalRef     = useRef(null);
    const audioIntervalRef = useRef(null);
    const analyzeRef      = useRef(false);
    const mediaRecRef     = useRef(null);
    const audioChunksRef  = useRef([]);

    const [isCapturing, setIsCapturing]     = useState(false);
    const [result, setResult]               = useState(null);
    const [voiceResult, setVoiceResult]     = useState(null);
    const [history, setHistory]             = useState([]);
    const [error, setError]                 = useState(null);
    const [isAnalyzing, setIsAnalyzing]     = useState(false);
    const [hasAudio, setHasAudio]           = useState(false);

    // ── Video frame analysis ────────────────────────────────────────────────
    const captureAndAnalyze = useCallback(() => {
        const video = videoRef.current;
        if (!video || !video.videoWidth || analyzeRef.current) return;

        analyzeRef.current = true;
        setIsAnalyzing(true);

        const canvas = document.createElement('canvas');
        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);

        canvas.toBlob(async (blob) => {
            if (!blob) { analyzeRef.current = false; setIsAnalyzing(false); return; }
            try {
                const fd = new FormData();
                fd.append('image', new File([blob], 'frame.png', { type: 'image/png' }));
                fd.append('source', 'videocall');   // skips challenge token + heatmap
                const res = await fetch(`${API}/api/detect/`, { method: 'POST', body: fd });
                if (!res.ok) throw new Error('Server error');
                const data = await res.json();
                setResult(data);
                setHistory(prev => [
                    { ...data, time: new Date().toLocaleTimeString(), kind: 'video' },
                    ...prev,
                ].slice(0, MAX_HISTORY));
                setError(null);
            } catch {
                setError('Video analysis failed — check backend is running');
            } finally {
                analyzeRef.current = false;
                setIsAnalyzing(false);
            }
        }, 'image/png');
    }, []);

    // ── Audio analysis ──────────────────────────────────────────────────────
    const analyzeAudio = useCallback(async () => {
        const chunks = audioChunksRef.current.splice(0);
        if (!chunks.length) return;
        const blob = new Blob(chunks, { type: 'audio/webm' });
        if (blob.size < 1000) return;   // skip near-empty chunks
        try {
            const fd = new FormData();
            fd.append('audio', new File([blob], 'clip.webm', { type: 'audio/webm' }));
            const res = await fetch(`${API}/api/detect/voice/`, { method: 'POST', body: fd });
            if (!res.ok) return;
            const data = await res.json();
            if (data.fake_prob !== null) {
                setVoiceResult(data);
                setHistory(prev => [
                    { overall: data.prediction, deepfake: { confidence: data.confidence }, time: new Date().toLocaleTimeString(), kind: 'voice' },
                    ...prev,
                ].slice(0, MAX_HISTORY));
            }
        } catch { /* silent — audio is best-effort */ }
    }, []);

    // ── Start / stop ────────────────────────────────────────────────────────
    const stopCapture = useCallback(() => {
        clearInterval(intervalRef.current);
        clearInterval(audioIntervalRef.current);
        intervalRef.current = null;
        audioIntervalRef.current = null;
        if (mediaRecRef.current && mediaRecRef.current.state !== 'inactive') {
            mediaRecRef.current.stop();
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(t => t.stop());
            streamRef.current = null;
        }
        if (videoRef.current) videoRef.current.srcObject = null;
        setIsCapturing(false);
    }, []);

    const startCapture = useCallback(async () => {
        setError(null);
        try {
            const stream = await navigator.mediaDevices.getDisplayMedia({
                video: true,
                audio: true,     // capture tab/system audio if browser allows
            });
            streamRef.current = stream;
            if (videoRef.current) videoRef.current.srcObject = stream;
            stream.getVideoTracks()[0].addEventListener('ended', stopCapture);

            // Check if we got audio tracks
            const audioTracks = stream.getAudioTracks();
            const gotAudio = audioTracks.length > 0;
            setHasAudio(gotAudio);

            if (gotAudio) {
                // Set up MediaRecorder on the audio stream
                const audioStream = new MediaStream(audioTracks);
                const rec = new MediaRecorder(audioStream, { mimeType: 'audio/webm' });
                rec.ondataavailable = e => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
                rec.start(500);   // collect chunks every 500ms
                mediaRecRef.current = rec;
                audioIntervalRef.current = setInterval(analyzeAudio, AUDIO_INTERVAL_MS);
            }

            setIsCapturing(true);
            intervalRef.current = setInterval(captureAndAnalyze, CAPTURE_INTERVAL_MS);
        } catch (err) {
            if (err.name !== 'NotAllowedError') {
                setError('Failed to start screen capture. Try a different browser or check permissions.');
            }
        }
    }, [captureAndAnalyze, analyzeAudio, stopCapture]);

    useEffect(() => () => stopCapture(), [stopCapture]);

    // ── Derived state ───────────────────────────────────────────────────────
    const videoFake  = result?.overall === 'deepfake';
    const voiceFake  = voiceResult?.prediction === 'fake';
    const anyFake    = videoFake || voiceFake;

    const alertLevel = !isCapturing ? 'idle'
        : anyFake               ? 'danger'
        : result?.overall === 'real' ? 'safe'
        : 'scanning';

    const alertStyles = {
        idle:     'border-slate-700 bg-slate-800/50',
        scanning: 'border-blue-700 bg-blue-900/20',
        safe:     'border-green-700 bg-green-900/20',
        danger:   'border-red-600 bg-red-900/30 animate-pulse',
    };

    const deepfakeCount = history.filter(h => h.overall === 'deepfake' || h.overall === 'fake').length;

    return (
        <div className="min-h-screen bg-slate-900 text-white flex flex-col">
            <nav className="p-4 border-b border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Shield className="w-6 h-6 text-blue-500" />
                    <span className="font-bold text-lg">DeepVerify — Video Call Scan</span>
                </div>
                <button onClick={onBack} className="flex items-center gap-2 px-4 py-2 hover:bg-slate-800 rounded-lg transition text-sm">
                    <ArrowLeft className="w-4 h-4" />Back
                </button>
            </nav>

            <main className="flex-1 p-6 max-w-5xl mx-auto w-full flex flex-col gap-5">
                {/* How-to banner */}
                {!isCapturing && (
                    <div className="p-5 bg-slate-800 rounded-2xl border border-slate-700">
                        <div className="flex items-center gap-3 mb-3">
                            <Monitor className="w-5 h-5 text-blue-400" />
                            <h2 className="font-bold">How to use</h2>
                        </div>
                        <ol className="space-y-1.5 text-slate-300 text-sm list-decimal list-inside">
                            <li>Click <strong>Start Screen Capture</strong> and select the tab or window with your video call</li>
                            <li>Allow audio sharing when prompted — enables voice deepfake detection</li>
                            <li>Frames are analysed every 3s (video) and 4s (voice) automatically</li>
                        </ol>
                        <p className="text-xs text-slate-500 mt-3">
                            Works with Google Meet, Zoom (web), Teams, and any browser-based call
                        </p>
                    </div>
                )}

                {/* Status bar */}
                {isCapturing && (
                    <div className={`flex items-center justify-between p-4 rounded-xl border transition-all duration-500 ${alertStyles[alertLevel]}`}>
                        <div className="flex items-center gap-3">
                            <span className={`w-3 h-3 rounded-full animate-pulse ${alertLevel === 'danger' ? 'bg-red-500' : alertLevel === 'safe' ? 'bg-green-500' : 'bg-blue-500'}`} />
                            <span className="font-semibold text-sm">
                                {alertLevel === 'danger' && '⚠ DEEPFAKE DETECTED'}
                                {alertLevel === 'safe'   && '✓ Call appears authentic'}
                                {alertLevel === 'scanning' && 'Scanning...'}
                            </span>
                        </div>
                        <div className="flex items-center gap-3 text-xs text-slate-400">
                            {hasAudio
                                ? <span className="flex items-center gap-1 text-green-400"><Mic className="w-3 h-3" />Voice active</span>
                                : <span className="flex items-center gap-1 text-slate-500"><MicOff className="w-3 h-3" />No audio</span>
                            }
                            <span>{history.length} frames · {deepfakeCount} flagged</span>
                        </div>
                    </div>
                )}

                {/* Video preview */}
                <div className="relative bg-black rounded-2xl overflow-hidden border border-slate-700" style={{ aspectRatio: '16/9' }}>
                    <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-contain" />
                    {!isCapturing && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900 text-slate-500">
                            <Monitor className="w-20 h-20 mb-4 opacity-20" />
                            <p className="text-sm">Screen preview will appear here</p>
                        </div>
                    )}
                    {isCapturing && (
                        <div className="absolute top-3 left-3 flex items-center gap-2 bg-black/70 px-3 py-1 rounded-full">
                            <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                            <span className="text-xs font-bold tracking-wider">LIVE SCAN</span>
                        </div>
                    )}
                    {isAnalyzing && (
                        <div className="absolute top-3 right-3 bg-purple-900/80 border border-purple-700 px-3 py-1 rounded-full text-xs text-purple-200">
                            Analyzing...
                        </div>
                    )}
                    {/* Live scores overlay */}
                    {isCapturing && (result || voiceResult) && (
                        <div className="absolute bottom-3 right-3 flex flex-col gap-1.5 items-end">
                            {result && (
                                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-bold backdrop-blur-sm border ${
                                    result.overall === 'real'
                                        ? 'bg-green-900/90 text-green-300 border-green-700'
                                        : 'bg-red-900/90 text-red-300 border-red-600'
                                }`}>
                                    {result.overall === 'real' ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
                                    Face: {result.overall === 'real' ? 'Real' : 'Deepfake'} · {result.deepfake?.confidence}%
                                </div>
                            )}
                            {voiceResult && voiceResult.fake_prob !== null && (
                                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-bold backdrop-blur-sm border ${
                                    voiceResult.prediction === 'real'
                                        ? 'bg-green-900/90 text-green-300 border-green-700'
                                        : 'bg-red-900/90 text-red-300 border-red-600'
                                }`}>
                                    <Mic className="w-3.5 h-3.5" />
                                    Voice: {voiceResult.prediction === 'real' ? 'Real' : 'Deepfake'} · {voiceResult.confidence}%
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Controls */}
                <div className="flex gap-3 items-center">
                    {!isCapturing ? (
                        <button onClick={startCapture} className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-full font-bold transition shadow-lg">
                            <Play className="w-4 h-4" />Start Screen Capture
                        </button>
                    ) : (
                        <button onClick={stopCapture} className="flex items-center gap-2 bg-red-600 hover:bg-red-700 px-6 py-3 rounded-full font-bold transition shadow-lg">
                            <Square className="w-4 h-4" />Stop Capture
                        </button>
                    )}
                    <span className="text-xs text-slate-500">Video every {CAPTURE_INTERVAL_MS / 1000}s · Voice every {AUDIO_INTERVAL_MS / 1000}s</span>
                </div>

                {error && (
                    <div className="p-4 bg-red-900/40 border border-red-700 rounded-xl text-red-300 text-sm">{error}</div>
                )}

                {/* History */}
                {history.length > 0 && (
                    <div>
                        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Analysis History</h3>
                        <div className="space-y-2">
                            {history.map((item, i) => {
                                const fake = item.overall === 'deepfake' || item.overall === 'fake';
                                return (
                                    <div key={i} className={`flex items-center justify-between px-4 py-3 rounded-xl text-sm border ${
                                        fake ? 'bg-red-900/20 border-red-800/60' : 'bg-green-900/10 border-green-900/50'
                                    }`}>
                                        <div className="flex items-center gap-2">
                                            {fake
                                                ? <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                                                : <CheckCircle2  className="w-4 h-4 text-green-400 shrink-0" />
                                            }
                                            <span className={`font-semibold capitalize ${fake ? 'text-red-300' : 'text-green-300'}`}>
                                                {item.overall}
                                            </span>
                                            <span className="text-slate-400">· {item.deepfake?.confidence}%</span>
                                            {item.kind === 'voice' && (
                                                <span className="text-xs text-purple-400 flex items-center gap-1">
                                                    <Mic className="w-3 h-3" />voice
                                                </span>
                                            )}
                                        </div>
                                        <span className="text-slate-500 text-xs tabular-nums">{item.time}</span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
};

export default VideoCallDetection;
