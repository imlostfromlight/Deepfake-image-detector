import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Search, Eye, Zap } from 'lucide-react';
import { Shield, Upload, Loader2, AlertTriangle, CheckCircle2, FileImage } from 'lucide-react';

const LandingPage = ({ onStartLive }) => {
    const [imagePreview, setImagePreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    // Backend Connection Logic
    const onDrop = useCallback(async (acceptedFiles) => {
        const file = acceptedFiles[0];
        if (!file) return;

        // Set Preview
        setImagePreview(URL.createObjectURL(file));
        setLoading(true);
        setResult(null);

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
        } catch (error) {
            console.error("Detection Error:", error);
            alert("Failed to connect to Django backend. Ensure server is running at port 8000.");
        } finally {
            setLoading(false);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.webp'] },
        multiple: false
    });

    return (
        <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
            <nav className="fixed w-full z-50 bg-white/80 backdrop-blur-md border-b border-slate-200">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between h-16 items-center">
                        <div className="flex items-center gap-2">
                            <Shield className="w-8 h-8 text-blue-600" />
                            <span className="font-bold text-xl tracking-tight">DeepVerify AI</span>
                        </div>
                        <div className="hidden md:flex space-x-8 font-medium">
                            <a href="#features" className="hover:text-blue-600 transition">Features</a>
                            <a href="#technology" className="hover:text-blue-600 transition">Technology</a>
                            <a href="#workflow" className="hover:text-blue-600 transition">How it Works</a>
                        </div>
                    </div>
                </div>
            </nav>

            <header className="pt-32 pb-16 px-4">
                <div className="max-w-4xl mx-auto text-center">
                    <h1 className="text-5xl font-extrabold mb-6 tracking-tight">
                        Instant <span className="text-blue-600">Deepfake</span> Analysis
                    </h1>

                    {/* Drag and Drop Zone */}
                    <div
                        {...getRootProps()}
                        className={`relative mt-10 p-12 border-2 border-dashed rounded-3xl transition-all cursor-pointer
                        ${isDragActive ? 'border-blue-500 bg-blue-50 scale-[1.02]' : 'border-slate-300 bg-white hover:border-blue-400'}`}
                    >
                        <input {...getInputProps()} />

                        <div className="flex flex-col items-center gap-4">
                            {loading ? (
                                <Loader2 className="w-12 h-12 text-blue-600 animate-spin" />
                            ) : (
                                <Upload className={`w-12 h-12 ${isDragActive ? 'text-blue-600' : 'text-slate-400'}`} />
                            )}

                            <div>
                                <p className="text-xl font-semibold">
                                    {isDragActive ? "Drop the image here" : "Drag & drop an image here"}
                                </p>
                                <p className="text-slate-500 mt-1 text-sm">Supports JPEG, PNG, WebP</p>
                            </div>
                        </div>
                    </div>

                    <div className="mt-8 flex justify-center">
                        <button
                            onClick={onStartLive}
                            className="flex items-center gap-2 bg-slate-900 text-white px-6 py-3 rounded-full hover:bg-slate-800 transition shadow-lg hover:shadow-xl transform hover:-translate-y-1"
                        >
                            <span className="relative flex h-3 w-3">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                            </span>
                            <span className="font-semibold">Switch to Live Camera Check</span>
                        </button>
                    </div>

                    {/* Analysis & Preview Section */}
                    <div className="mt-12 grid md:grid-cols-2 gap-8 items-start">
                        {/* Image Preview */}
                        <div className="bg-white p-3 rounded-2xl shadow-xl border border-slate-200">
                            {imagePreview ? (
                                <img src={imagePreview} className="w-full h-64 object-cover rounded-xl" alt="To be analyzed" />
                            ) : (
                                <div className="h-64 flex flex-col items-center justify-center text-slate-400 bg-slate-50 rounded-xl">
                                    <FileImage className="w-10 h-10 mb-2" />
                                    <p>No image uploaded</p>
                                </div>
                            )}
                        </div>

                        {/* Results Card */}
                        <div className="h-full">
                            {loading ? (
                                <div className="bg-white p-8 rounded-2xl border border-slate-200 h-64 flex flex-col items-center justify-center">
                                    <Loader2 className="w-8 h-8 text-blue-600 animate-spin mb-4" />
                                    <p className="font-medium animate-pulse">Running ViT-v2 Inference...</p>
                                </div>
                            ) : result ? (
                                <div className={`p-8 rounded-2xl border-2 h-64 flex flex-col justify-center transition-all ${result.prediction === 'fake' ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'
                                    }`}>
                                    <div className="flex items-center gap-3 mb-4">
                                        {result.prediction === 'fake' ?
                                            <AlertTriangle className="text-red-600 w-8 h-8" /> :
                                            <CheckCircle2 className="text-green-600 w-8 h-8" />
                                        }
                                        <h3 className="text-3xl font-bold capitalize">{result.prediction}</h3>
                                    </div>
                                    <div className="space-y-2">
                                        <div className="flex justify-between text-sm font-bold uppercase text-slate-500">
                                            <span>Confidence Score</span>
                                            <span>{result.confidence}%</span>
                                        </div>
                                        <div className="w-full bg-slate-200 rounded-full h-3">
                                            <div
                                                className={`h-3 rounded-full transition-all duration-1000 ${result.prediction === 'fake' ? 'bg-red-500' : 'bg-green-500'}`}
                                                style={{ width: `${result.confidence}%` }}
                                            />
                                        </div>
                                        <p className="text-xs text-slate-400 mt-4 italic">Analysis powered by PrithivML Deep-Fake-Detector-v2</p>
                                    </div>
                                </div>
                            ) : (
                                <div className="bg-slate-100 p-8 rounded-2xl border border-slate-200 h-64 flex items-center justify-center text-slate-400">
                                    <p>Results will appear here</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </header>
            {/* Stats/Social Proof */}
            <section className="py-12 bg-white border-y border-slate-200">
                <div className="max-w-7xl mx-auto px-4 flex flex-wrap justify-around gap-8 text-center">
                    <div>
                        <div className="text-3xl font-bold text-slate-900">90%+</div>
                        <div className="text-sm text-slate-500 uppercase tracking-widest font-semibold">Model Accuracy</div>
                    </div>
                    <div>
                        <div className="text-3xl font-bold text-slate-900">CNN-Based</div>
                        <div className="text-sm text-slate-500 uppercase tracking-widest font-semibold">Architecture </div>
                    </div>
                    <div>
                        <div className="text-3xl font-bold text-slate-900">Real-Time</div>
                        <div className="text-sm text-slate-500 uppercase tracking-widest font-semibold">Inference</div>
                    </div>
                </div>
            </section>

            {/* Problem Section */}
            <section id="features" className="py-24 px-4 bg-slate-50">
                <div className="max-w-7xl mx-auto">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl md:text-4xl font-bold mb-4 text-slate-900">Why Automated Detection?</h2>
                        <p className="text-slate-600 max-w-xl mx-auto text-lg">
                            Human eyes struggle to identify subtle deepfake artifacts, highlighting the urgent need for automated ML solutions.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-3 gap-8">
                        <FeatureCard
                            icon={<Search className="w-6 h-6 text-blue-600" />}
                            title="Xception & EfficientNet"
                            desc="Leveraging state-of-the-art CNNs with depthwise separable convolutions for robust feature extraction."
                        />
                        <FeatureCard
                            icon={<Eye className="w-6 h-6 text-blue-600" />}
                            title="Explainable AI (XAI)"
                            desc="Integration of Grad-CAM to visualize decision-making heatmaps, increasing user trust."
                        />
                        <FeatureCard
                            icon={<Zap className="w-6 h-6 text-blue-600" />}
                            title="Scalable Architecture"
                            desc="React-Django integration ensuring high performance and accessible web-based verification."
                        />
                    </div>
                </div>
            </section>

            {/* Workflow Section */}
            <section id="workflow" className="py-24 bg-white px-4">
                <div className="max-w-5xl mx-auto">
                    <div className="flex flex-col md:flex-row items-center gap-12">
                        <div className="flex-1">
                            <h2 className="text-3xl font-bold mb-6">Simple 3-Step Verification</h2>
                            <div className="space-y-6">
                                <Step num="01" title="Upload Image" text="Upload any suspicious image through our intuitive web interface." />
                                <Step num="02" title="ML Analysis" text="Our CNN models analyze the image for subtle GAN-generated patterns." />
                                <Step num="03" title="Review Results" text="Receive a authenticity score along with visual heatmaps explaining the detection." />
                            </div>
                        </div>
                        <div className="flex-1 bg-slate-900 rounded-3xl p-8 text-white relative overflow-hidden">
                            <div className="absolute top-0 right-0 p-4 opacity-10"><Shield className="w-32 h-32" /></div>
                            <h3 className="text-xl font-bold mb-4">Security First</h3>
                            <p className="text-slate-400 mb-6">Implementing API security mechanisms including input validation and encryption to protect data integrity.</p>
                            <div className="flex gap-2">
                                <span className="px-3 py-1 bg-slate-800 rounded-full text-xs">Auth</span>
                                <span className="px-3 py-1 bg-slate-800 rounded-full text-xs">S3 Storage</span>
                                <span className="px-3 py-1 bg-slate-800 rounded-full text-xs">Django API</span>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="bg-slate-900 text-white py-12 border-t border-slate-800">
                <div className="max-w-7xl mx-auto px-4 flex flex-col md:flex-row justify-between items-center gap-6">
                    <div>
                        <div className="flex items-center gap-2 mb-2">
                            <Shield className="w-6 h-6 text-blue-400" />
                            <span className="font-bold text-lg tracking-tight">DeepVerify AI</span>
                        </div>
                        <p className="text-slate-400 text-sm">© 2025 Astana IT University. Diploma Project.</p>
                    </div>
                    <div className="text-right">
                        <p className="text-slate-400 text-sm">Student: Nadyrkhan Shyntemir Nurlanuly </p>
                        <p className="text-slate-400 text-sm">Cybersecurity Program, 6B06301 </p>
                    </div>
                </div>
            </footer>
        </div>
    );
};
const FeatureCard = ({ icon, title, desc }) => (
    <div className="p-8 bg-white rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition group">
        <div className="w-12 h-12 bg-blue-50 rounded-xl flex items-center justify-center mb-6 group-hover:bg-blue-600 group-hover:text-white transition-colors">
            {icon}
        </div>
        <h3 className="text-xl font-bold mb-3 text-slate-900">{title}</h3>
        <p className="text-slate-600 leading-relaxed text-sm">{desc}</p>
    </div>
);

const Step = ({ num, title, text }) => (
    <div className="flex gap-4">
        <div className="text-2xl font-black text-blue-100">{num}</div>
        <div>
            <h4 className="font-bold text-slate-900">{title}</h4>
            <p className="text-slate-600 text-sm">{text}</p>
        </div>
    </div>
);

export default LandingPage;