import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import { Shield, ArrowLeft, Camera, AlertTriangle, CheckCircle2 } from 'lucide-react';

const LiveDetection = ({ onBack }) => {
    const webcamRef = useRef(null);
    const [isDetecting, setIsDetecting] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const intervalRef = useRef(null);

    const captureAndDetect = useCallback(async () => {
        if (!webcamRef.current) return;

        const imageSrc = webcamRef.current.getScreenshot();
        if (!imageSrc) return;

        // Convert base64 to blob
        const res = await fetch(imageSrc);
        const blob = await res.blob();
        const file = new File([blob], "webcam-frame.jpg", { type: "image/jpeg" });

        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('http://localhost:8000/api/detect/', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error('Network response was not ok');

            const data = await response.json();
            setResult(data);
            setError(null);
        } catch (err) {
            console.error("Detection Error:", err);
            setError("Connection to server failed.");
            setIsDetecting(false); // Stop on error to prevent spamming
            if (intervalRef.current) clearInterval(intervalRef.current);
        }
    }, [webcamRef]);

    const toggleDetection = () => {
        if (isDetecting) {
            setIsDetecting(false);
            if (intervalRef.current) clearInterval(intervalRef.current);
            setResult(null);
        } else {
            setIsDetecting(true);
            intervalRef.current = setInterval(captureAndDetect, 1000); // 1 fps
        }
    };

    useEffect(() => {
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, []);

    const videoConstraints = {
        width: 720,
        height: 720,
        facingMode: "user"
    };

    return (
        <div className="min-h-screen bg-slate-900 text-white flex flex-col">
            <nav className="p-4 border-b border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Shield className="w-6 h-6 text-blue-500" />
                    <span className="font-bold text-lg">DeepVerify Live</span>
                </div>
                <button
                    onClick={onBack}
                    className="flex items-center gap-2 px-4 py-2 hover:bg-slate-800 rounded-lg transition"
                >
                    <ArrowLeft className="w-4 h-4" />
                    Back to Home
                </button>
            </nav>

            <main className="flex-1 flex flex-col items-center justify-center p-4">
                <div className="relative max-w-2xl w-full bg-black rounded-3xl overflow-hidden shadow-2xl border-4 border-slate-800">
                    <Webcam
                        audio={false}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        videoConstraints={videoConstraints}
                        className="w-full h-auto"
                        mirrored={true}
                    />

                    {/* Overlay UI */}
                    <div className="absolute inset-x-0 bottom-0 p-6 bg-gradient-to-t from-black/80 to-transparent">
                        <div className="flex items-center justify-between">
                            <div>
                                {result ? (
                                    <div className="flex items-center gap-3">
                                        {result.prediction === 'fake' ? (
                                            <div className="flex items-center gap-2 text-red-500">
                                                <AlertTriangle className="w-8 h-8" />
                                                <span className="text-3xl font-bold uppercase tracking-wider">FAKE</span>
                                            </div>
                                        ) : (
                                            <div className="flex items-center gap-2 text-green-500">
                                                <CheckCircle2 className="w-8 h-8" />
                                                <span className="text-3xl font-bold uppercase tracking-wider">REAL</span>
                                            </div>
                                        )}
                                        <div className="bg-slate-800 px-3 py-1 rounded text-sm text-slate-300">
                                            {result.confidence}% Conf.
                                        </div>
                                    </div>
                                ) : (
                                    <div className="text-slate-400 font-medium">
                                        {isDetecting ? "Analyzing video stream..." : "Ready to start detection"}
                                    </div>
                                )}
                            </div>

                            <button
                                onClick={toggleDetection}
                                className={`px-6 py-3 rounded-full font-bold transition flex items-center gap-2 ${
                                    isDetecting
                                        ? "bg-red-600 hover:bg-red-700 text-white"
                                        : "bg-blue-600 hover:bg-blue-700 text-white"
                                }`}
                            >
                                <Camera className="w-5 h-5" />
                                {isDetecting ? "Stop" : "Start Live Check"}
                            </button>
                        </div>
                        {error && <p className="text-red-400 mt-2 text-sm">{error}</p>}
                    </div>
                </div>

                <div className="mt-8 text-center text-slate-500 max-w-md">
                    <p>Ensure good lighting and face the camera directly.</p>
                    <p className="text-xs mt-2 opacity-50">Note: This system analyzes frames for digital manipulation artifacts. It does not replace biometric liveness verification.</p>
                </div>
            </main>
        </div>
    );
};

export default LiveDetection;
