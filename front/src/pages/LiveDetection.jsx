import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import { Shield, ArrowLeft, CheckCircle2, AlertTriangle, RefreshCw } from 'lucide-react';

const MOTION_THRESHOLD    = 6;    // avg pixel diff per channel to count as motion
const MOTION_FRAMES_NEEDED = 3;   // motion frames needed to pass a turn
const TURN_WINDOW_MS      = 5000; // ms to complete each turn
const POSITION_WAIT_MS    = 2000; // ms to position face before challenge starts

const STEP_LABELS = ['Position', 'Turn Left', 'Turn Right', 'Verify'];
const STEP_PHASES = ['position', 'turn_left', 'turn_right', 'verifying'];

const LiveDetection = ({ onBack }) => {
    const webcamRef = useRef(null);
    const [phase, setPhase] = useState('idle');
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [motionOk, setMotionOk] = useState(false);
    const [motionCount, setMotionCount] = useState(0);

    // Motion detection state
    const prevPixelsRef   = useRef(null);
    const motionCountRef  = useRef(0);
    const motionMetRef    = useRef(false);
    const timerRef        = useRef(null);
    const diffIntervalRef = useRef(null);

    // Injection-defense state
    const challengeTokenRef = useRef(null);  // server-issued HMAC token
    const frame0Ref         = useRef(null);  // screenshot after position phase
    const frame1Ref         = useRef(null);  // screenshot after left-turn phase

    const captureScreenshot = useCallback(() => (
        webcamRef.current?.getScreenshot() ?? null
    ), []);

    const sampleFrame = useCallback(() => {
        const video = webcamRef.current?.video;
        if (!video || video.readyState < 2) return null;
        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 480;
        const cx = vw * 0.2, cy = vh * 0.1;
        const cw = vw * 0.6, ch = vh * 0.8;
        const c = document.createElement('canvas');
        c.width = 120; c.height = 120;
        c.getContext('2d').drawImage(video, cx, cy, cw, ch, 0, 0, 120, 120);
        return c.getContext('2d').getImageData(0, 0, 120, 120).data;
    }, []);

    const computeDiff = useCallback(() => {
        const curr = sampleFrame();
        if (!curr) return 0;
        let diff = 0;
        if (prevPixelsRef.current) {
            for (let i = 0; i < curr.length; i += 4)
                diff += Math.abs(curr[i] - prevPixelsRef.current[i]);
            diff /= curr.length / 4;
        }
        prevPixelsRef.current = curr;
        return diff;
    }, [sampleFrame]);

    const stopAll = useCallback(() => {
        clearInterval(diffIntervalRef.current);
        clearTimeout(timerRef.current);
        diffIntervalRef.current = null;
        timerRef.current = null;
    }, []);

    const runTurn = useCallback((turnPhase, onPass, onFail) => {
        setPhase(turnPhase);
        setMotionOk(false);
        setMotionCount(0);
        motionCountRef.current = 0;
        motionMetRef.current = false;
        prevPixelsRef.current = null;

        diffIntervalRef.current = setInterval(() => {
            const diff = computeDiff();
            if (diff > MOTION_THRESHOLD) {
                motionCountRef.current++;
                setMotionCount(motionCountRef.current);
                if (motionCountRef.current >= MOTION_FRAMES_NEEDED && !motionMetRef.current) {
                    motionMetRef.current = true;
                    setMotionOk(true);
                }
            }
        }, 150);

        timerRef.current = setTimeout(() => {
            clearInterval(diffIntervalRef.current);
            motionMetRef.current ? onPass() : onFail();
        }, TURN_WINDOW_MS);
    }, [computeDiff]);

    // ── Verification — submits 3 frames + signed token to backend ────────────
    const verify = useCallback(async () => {
        setPhase('verifying');

        const frame2Src = captureScreenshot();
        const frame0Src = frame0Ref.current;
        const frame1Src = frame1Ref.current;
        const token     = challengeTokenRef.current;

        if (!frame0Src || !frame1Src || !frame2Src || !token) {
            setError('Frame capture failed — try again.');
            setPhase('failed');
            return;
        }

        try {
            const toBlob = async (src) => (await fetch(src)).blob();
            const [blob0, blob1, blob2] = await Promise.all([
                toBlob(frame0Src), toBlob(frame1Src), toBlob(frame2Src),
            ]);

            const formData = new FormData();
            formData.append('frame_0', new File([blob0], 'frame0.png', { type: 'image/png' }));
            formData.append('frame_1', new File([blob1], 'frame1.png', { type: 'image/png' }));
            formData.append('frame_2', new File([blob2], 'frame2.png', { type: 'image/png' }));
            formData.append('challenge_token', token);
            formData.append('source', 'live');

            const response = await fetch('http://localhost:8000/api/detect/', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Server error');
            }

            setResult(await response.json());
            setPhase('complete');
        } catch (e) {
            setError(e.message || 'Connection to server failed.');
            setPhase('failed');
        }
    }, [captureScreenshot]);

    // ── Challenge start — fetches token then runs the 3-phase challenge ───────
    const startChallenge = useCallback(async () => {
        stopAll();
        setResult(null);
        setError(null);
        setMotionOk(false);
        setMotionCount(0);
        prevPixelsRef.current  = null;
        challengeTokenRef.current = null;
        frame0Ref.current      = null;
        frame1Ref.current      = null;

        // Step 1: obtain server-issued challenge token (injection defense)
        try {
            const res = await fetch('http://localhost:8000/api/live/challenge/');
            if (!res.ok) throw new Error('Server error');
            const data = await res.json();
            challengeTokenRef.current = data.token;
        } catch {
            setError('Could not start challenge — check server connection.');
            return;
        }

        setPhase('position');

        timerRef.current = setTimeout(() => {
            // Capture frame_0: neutral frontal position
            frame0Ref.current = captureScreenshot();

            const fail = () => { stopAll(); setPhase('failed'); };

            runTurn('turn_left',
                () => {
                    // Capture frame_1: after completing left turn
                    frame1Ref.current = captureScreenshot();
                    runTurn('turn_right', () => verify(), fail);
                },
                fail,
            );
        }, POSITION_WAIT_MS);
    }, [stopAll, runTurn, verify, captureScreenshot]);

    const reset = useCallback(() => {
        stopAll();
        setPhase('idle');
        setResult(null);
        setError(null);
        setMotionOk(false);
        setMotionCount(0);
        prevPixelsRef.current     = null;
        challengeTokenRef.current = null;
        frame0Ref.current         = null;
        frame1Ref.current         = null;
    }, [stopAll]);

    useEffect(() => () => stopAll(), [stopAll]);

    const ovalColor = (() => {
        if (phase === 'complete') return '#22c55e';
        if (phase === 'failed')   return '#ef4444';
        if (motionOk)             return '#22c55e';
        if (phase === 'idle')     return '#475569';
        if (phase === 'verifying') return '#a855f7';
        return '#3b82f6';
    })();

    const instruction = {
        idle:       'Press Start to begin verification',
        position:   'Center your face in the oval',
        turn_left:  '← Slowly turn your head LEFT',
        turn_right: 'Slowly turn your head RIGHT →',
        verifying:  'Analyzing...',
        complete:   result?.overall === 'real' ? 'Verification passed' : `Detected: ${result?.overall?.toUpperCase()}`,
        failed:     error || 'Motion not detected — please try again',
    }[phase] ?? '';

    const activeStep = STEP_PHASES.indexOf(phase);
    const videoConstraints = { width: 720, height: 720, facingMode: 'user' };

    return (
        <div className="min-h-screen bg-slate-900 text-white flex flex-col">
            <nav className="p-4 border-b border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Shield className="w-6 h-6 text-blue-500" />
                    <span className="font-bold text-lg">DeepVerify Live</span>
                </div>
                <button onClick={onBack} className="flex items-center gap-2 px-4 py-2 hover:bg-slate-800 rounded-lg transition">
                    <ArrowLeft className="w-4 h-4" />
                    Back to Home
                </button>
            </nav>

            <main className="flex-1 flex flex-col items-center justify-center p-4 gap-6">

                {/* Step indicators */}
                {phase !== 'idle' && (
                    <div className="flex items-center gap-1">
                        {STEP_LABELS.map((label, i) => {
                            const done   = activeStep > i || phase === 'complete';
                            const active = activeStep === i;
                            return (
                                <React.Fragment key={label}>
                                    <div className="flex flex-col items-center gap-1">
                                        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-all ${
                                            done   ? 'bg-green-500 text-white' :
                                            active ? 'bg-blue-500 text-white ring-4 ring-blue-500/30' :
                                                     'bg-slate-700 text-slate-400'
                                        }`}>
                                            {done ? <CheckCircle2 className="w-4 h-4" /> : i + 1}
                                        </div>
                                        <span className={`text-xs ${active ? 'text-blue-400' : done ? 'text-green-400' : 'text-slate-500'}`}>
                                            {label}
                                        </span>
                                    </div>
                                    {i < STEP_LABELS.length - 1 && (
                                        <div className={`w-10 h-0.5 mb-5 transition-all ${done ? 'bg-green-500' : 'bg-slate-700'}`} />
                                    )}
                                </React.Fragment>
                            );
                        })}
                    </div>
                )}

                {/* Webcam + oval overlay */}
                <div className="relative w-72 h-72 sm:w-96 sm:h-96">
                    <Webcam
                        audio={false}
                        ref={webcamRef}
                        screenshotFormat="image/png"
                        videoConstraints={videoConstraints}
                        className="w-full h-full object-cover rounded-2xl"
                        mirrored={true}
                    />

                    <svg
                        className="absolute inset-0 w-full h-full"
                        viewBox="0 0 100 100"
                        preserveAspectRatio="none"
                    >
                        <defs>
                            <mask id="oval-cut">
                                <rect width="100" height="100" fill="white" />
                                <ellipse cx="50" cy="47" rx="35" ry="43" fill="black" />
                            </mask>
                        </defs>
                        <rect width="100" height="100" fill="rgba(0,0,0,0.55)" mask="url(#oval-cut)" />
                        <ellipse
                            cx="50" cy="47" rx="35" ry="43"
                            fill="none"
                            stroke={ovalColor}
                            strokeWidth="1.8"
                            style={{ transition: 'stroke 0.4s' }}
                        />
                        {phase === 'position' && ['M15,22 L15,12 L25,12', 'M85,22 L85,12 L75,12', 'M15,72 L15,82 L25,82', 'M85,72 L85,82 L75,82'].map((d, i) => (
                            <path key={i} d={d} fill="none" stroke={ovalColor} strokeWidth="1.5" />
                        ))}
                    </svg>

                    {(phase === 'turn_left' || phase === 'turn_right') && (
                        <div className="absolute inset-0 flex items-center pointer-events-none">
                            {phase === 'turn_left' && (
                                <span className={`text-5xl font-black ml-1 transition-colors ${motionOk ? 'text-green-400' : 'text-blue-400'} animate-pulse`}>←</span>
                            )}
                            {phase === 'turn_right' && (
                                <span className={`text-5xl font-black ml-auto mr-1 transition-colors ${motionOk ? 'text-green-400' : 'text-blue-400'} animate-pulse`}>→</span>
                            )}
                        </div>
                    )}

                    {phase === 'verifying' && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-2xl">
                            <div className="text-center">
                                <div className="w-12 h-12 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                                <p className="text-purple-300 font-semibold">Analyzing...</p>
                            </div>
                        </div>
                    )}

                    {phase === 'complete' && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-2xl">
                            <div className={`text-center p-5 rounded-2xl ${result?.overall === 'real' ? 'bg-green-900/90' : 'bg-red-900/90'}`}>
                                {result?.overall === 'real'
                                    ? <CheckCircle2 className="w-14 h-14 text-green-400 mx-auto mb-2" />
                                    : <AlertTriangle className="w-14 h-14 text-red-400 mx-auto mb-2" />
                                }
                                <div className={`text-2xl font-black ${result?.overall === 'real' ? 'text-green-300' : 'text-red-300'}`}>
                                    {result?.overall?.toUpperCase()}
                                </div>
                                <div className="text-xs text-slate-300 mt-1">
                                    {result?.deepfake?.confidence}% confidence
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Instruction */}
                <div className="text-center space-y-3">
                    <p className={`text-lg font-semibold transition-colors ${
                        phase === 'complete' && result?.overall === 'real' ? 'text-green-400' :
                        phase === 'complete'  ? 'text-red-400'    :
                        phase === 'failed'    ? 'text-red-400'    :
                        motionOk             ? 'text-green-400'  :
                                               'text-slate-200'
                    }`}>
                        {instruction}
                    </p>

                    {(phase === 'turn_left' || phase === 'turn_right') && (
                        <div className="flex items-center gap-2 justify-center">
                            <span className="text-xs text-slate-400">Motion</span>
                            <div className="w-28 h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-green-500 rounded-full transition-all duration-200"
                                    style={{ width: `${Math.min(100, (motionCount / MOTION_FRAMES_NEEDED) * 100)}%` }}
                                />
                            </div>
                            {motionOk && <CheckCircle2 className="w-4 h-4 text-green-400" />}
                        </div>
                    )}

                    {phase === 'complete' && (
                        <div className="flex gap-2 justify-center flex-wrap">
                            <span className={`text-xs px-3 py-1 rounded-full ${result?.liveness?.prediction === 'spoof' ? 'bg-red-900/60 text-red-300' : 'bg-green-900/60 text-green-300'}`}>
                                Anti-spoof: {result?.liveness?.prediction ?? 'live'} {result?.liveness?.confidence}%
                            </span>
                            <span className={`text-xs px-3 py-1 rounded-full ${result?.deepfake?.prediction === 'real' ? 'bg-green-900/60 text-green-300' : 'bg-red-900/60 text-red-300'}`}>
                                Deepfake: {result?.deepfake?.prediction} {result?.deepfake?.confidence}%
                            </span>
                        </div>
                    )}
                </div>

                {(phase === 'idle' || phase === 'failed' || phase === 'complete') && (
                    <button
                        onClick={startChallenge}
                        className="px-8 py-3 bg-blue-600 hover:bg-blue-700 rounded-full font-bold transition flex items-center gap-2 shadow-lg"
                    >
                        {phase === 'idle'
                            ? 'Start Verification'
                            : <><RefreshCw className="w-4 h-4" /> Try Again</>
                        }
                    </button>
                )}

                <p className="text-xs text-slate-600 text-center max-w-xs">
                    Ensure good lighting and face the camera. Remove sunglasses for best results.
                </p>
            </main>
        </div>
    );
};

export default LiveDetection;
