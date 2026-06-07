import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Shield, ArrowLeft, FileText, Loader2, CheckCircle2, AlertTriangle, FileImage, Eye, Layers } from 'lucide-react';

/* ── Visualization tab component reused per image ──────────────────────────── */
const VizTabs = ({ item }) => {
    const [tab, setTab] = useState('face');
    const tabs = [
        { key: 'face',    label: 'Face crop' },
        { key: 'heatmap', label: 'GradCAM heatmap' },
        { key: 'ela',     label: 'ELA map' },
    ];

    const src =
        tab === 'face'    ? item.deepfake?.face_crop :
        tab === 'heatmap' ? item.deepfake?.heatmap   :
                            item.deepfake?.ela_map;

    const hint =
        tab === 'face'    ? 'Aligned face extracted from the document' :
        tab === 'heatmap' ? 'Red = regions the model focused on for this verdict' :
                            'Bright areas show inconsistent JPEG compression — editing hotspots';

    return (
        <div className="mt-3 rounded-xl overflow-hidden border border-slate-200 bg-white">
            {/* Tab bar */}
            <div className="flex border-b border-slate-100">
                {tabs.map(t => (
                    <button
                        key={t.key}
                        onClick={() => setTab(t.key)}
                        className={`flex-1 py-1.5 text-xs font-semibold uppercase tracking-wider transition
                            ${tab === t.key
                                ? 'bg-blue-600 text-white'
                                : 'text-slate-400 hover:text-slate-600 hover:bg-slate-50'}`}
                    >
                        {t.label}
                    </button>
                ))}
            </div>

            {/* Hint text */}
            <p className="text-xs text-slate-400 px-3 pt-1.5">{hint}</p>

            {/* Image */}
            <div className="p-2">
                {src ? (
                    <img src={src} alt={tab} className="w-full max-h-56 object-contain rounded-lg bg-slate-50" />
                ) : (
                    <div className="h-32 flex items-center justify-center text-slate-400 text-xs bg-slate-50 rounded-lg">
                        {tab === 'face'    && <><FileImage className="w-5 h-5 mr-1.5 opacity-40" />No face detected</>}
                        {tab === 'heatmap' && <><Eye     className="w-5 h-5 mr-1.5 opacity-40" />Heatmap unavailable</>}
                        {tab === 'ela'     && <><Layers  className="w-5 h-5 mr-1.5 opacity-40" />ELA unavailable</>}
                    </div>
                )}
            </div>

            {/* ELA score bar */}
            {tab === 'ela' && item.deepfake?.ela_score !== undefined && (
                <div className="px-3 pb-2">
                    <div className="flex justify-between text-xs text-slate-500 mb-1">
                        <span>ELA score: <span className={`font-bold ${
                            item.deepfake.ela_score > 8 ? 'text-red-600' :
                            item.deepfake.ela_score > 4 ? 'text-amber-600' : 'text-green-600'
                        }`}>{item.deepfake.ela_score}</span></span>
                        <span>{item.deepfake.ela_score > 8 ? 'High — likely edited' : item.deepfake.ela_score > 4 ? 'Moderate' : 'Low'}</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-1.5">
                        <div
                            className={`h-1.5 rounded-full ${
                                item.deepfake.ela_score > 8 ? 'bg-red-500' :
                                item.deepfake.ela_score > 4 ? 'bg-amber-500' : 'bg-green-500'
                            }`}
                            style={{ width: `${Math.min(100, item.deepfake.ela_score * 6)}%`, transition: 'width 0.8s ease' }}
                        />
                    </div>
                </div>
            )}
        </div>
    );
};

/* ── Main page ─────────────────────────────────────────────────────────────── */
const DocumentDetection = ({ onBack }) => {
    const [result, setResult]   = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError]     = useState(null);
    const [fileName, setFileName] = useState(null);

    const onDrop = useCallback(async (acceptedFiles) => {
        const file = acceptedFiles[0];
        if (!file) return;

        setFileName(file.name);
        setLoading(true);
        setResult(null);
        setError(null);

        const formData = new FormData();
        formData.append('document', file);

        try {
            const response = await fetch(`${import.meta.env.VITE_API_URL}/api/detect/document/`, {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            if (!response.ok || (data.error && !data.results?.length)) {
                throw new Error(data.error || 'Server error');
            }
            setResult(data);
        } catch (err) {
            setError(err.message || 'Failed to connect to backend. Ensure server is running at port 8000.');
        } finally {
            setLoading(false);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'application/pdf': ['.pdf'],
            'image/jpeg': ['.jpg', '.jpeg'],
            'image/png': ['.png'],
            'image/webp': ['.webp'],
        },
        multiple: false,
    });

    const reset = () => { setResult(null); setError(null); setFileName(null); };

    return (
        <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
            <nav className="fixed w-full z-50 bg-white/80 backdrop-blur-md border-b border-slate-200">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between h-16 items-center">
                        <div className="flex items-center gap-2">
                            <Shield className="w-8 h-8 text-blue-600" />
                            <span className="font-bold text-xl tracking-tight">DeepVerify — Document Scan</span>
                        </div>
                        <button onClick={onBack} className="flex items-center gap-2 px-4 py-2 hover:bg-slate-100 rounded-lg transition text-sm font-medium">
                            <ArrowLeft className="w-4 h-4" />
                            Back to Home
                        </button>
                    </div>
                </div>
            </nav>

            <main className="pt-28 pb-16 px-4 max-w-3xl mx-auto">
                <div className="text-center mb-10">
                    <h1 className="text-4xl font-extrabold mb-3 tracking-tight">
                        Document <span className="text-blue-600">Deepfake</span> Scan
                    </h1>
                    <p className="text-slate-500 text-lg max-w-xl mx-auto">
                        Upload a PDF or image — passports, ID cards, reports — to detect AI-generated or manipulated faces inside
                    </p>
                </div>

                {/* Upload zone */}
                <div
                    {...getRootProps()}
                    className={`p-12 border-2 border-dashed rounded-3xl transition-all cursor-pointer mb-6
                    ${isDragActive ? 'border-blue-500 bg-blue-50 scale-[1.02]' : 'border-slate-300 bg-white hover:border-blue-400'}`}
                >
                    <input {...getInputProps()} />
                    <div className="flex flex-col items-center gap-4">
                        {loading
                            ? <Loader2 className="w-12 h-12 text-blue-600 animate-spin" />
                            : <FileText className={`w-12 h-12 ${isDragActive ? 'text-blue-600' : 'text-slate-400'}`} />
                        }
                        <div className="text-center">
                            <p className="text-xl font-semibold">
                                {isDragActive ? 'Drop document here' : 'Drag & drop a document'}
                            </p>
                            <p className="text-slate-500 mt-1 text-sm">Supports PDF, JPEG, PNG, WebP</p>
                        </div>
                    </div>
                </div>

                {/* Loading */}
                {loading && (
                    <div className="p-6 bg-white rounded-2xl border border-slate-200 text-center shadow-sm">
                        <Loader2 className="w-8 h-8 text-blue-600 animate-spin mx-auto mb-3" />
                        <p className="font-semibold animate-pulse">Extracting faces and running forensic analysis…</p>
                        <p className="text-xs text-slate-400 mt-1">Generating heatmap + ELA for each face found</p>
                    </div>
                )}

                {/* Error */}
                {error && !loading && (
                    <div className="p-4 bg-red-50 border border-red-200 rounded-2xl text-red-700 text-sm mb-4">
                        {error}
                    </div>
                )}

                {/* Results */}
                {result && !loading && (
                    <div className="space-y-4">
                        {/* Summary card */}
                        <div className={`p-6 rounded-2xl border-2 ${
                            result.deepfakes_found > 0 ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'
                        }`}>
                            <div className="flex items-start gap-4">
                                {result.deepfakes_found > 0
                                    ? <AlertTriangle className="w-8 h-8 text-red-600 shrink-0 mt-0.5" />
                                    : <CheckCircle2  className="w-8 h-8 text-green-600 shrink-0 mt-0.5" />
                                }
                                <div className="flex-1">
                                    <h2 className={`text-xl font-bold mb-1 ${result.deepfakes_found > 0 ? 'text-red-700' : 'text-green-700'}`}>
                                        {result.deepfakes_found > 0
                                            ? `${result.deepfakes_found} Deepfake${result.deepfakes_found > 1 ? 's' : ''} Detected`
                                            : 'No Deepfakes Found'}
                                    </h2>
                                    <div className="flex flex-wrap gap-3 text-sm text-slate-500">
                                        <span className="flex items-center gap-1">
                                            <FileImage className="w-4 h-4" />{result.filename}
                                        </span>
                                        <span>{result.total_analyzed} image{result.total_analyzed !== 1 ? 's' : ''} analyzed</span>
                                    </div>
                                </div>
                                <button onClick={reset} className="text-xs text-slate-400 hover:text-slate-600 underline shrink-0">
                                    Scan another
                                </button>
                            </div>
                        </div>

                        {/* Per-image breakdown */}
                        {result.results.length > 0 && (
                            <div className="space-y-3">
                                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                                    Image-by-image breakdown
                                </h3>
                                {result.results.map((item, i) => {
                                    const isFake = item.overall === 'deepfake';
                                    return (
                                        <div
                                            key={i}
                                            className={`p-4 rounded-2xl border ${isFake ? 'bg-red-50 border-red-200' : 'bg-white border-slate-200'}`}
                                        >
                                            {/* Header row */}
                                            <div className="flex items-center justify-between mb-2">
                                                <div className="flex items-center gap-2">
                                                    {isFake
                                                        ? <AlertTriangle className="w-4 h-4 text-red-500 shrink-0" />
                                                        : <CheckCircle2  className="w-4 h-4 text-green-500 shrink-0" />
                                                    }
                                                    <span className="text-sm font-medium text-slate-700">
                                                        Page {item.page}, Image {item.index}
                                                    </span>
                                                    <span className={`text-xs font-bold uppercase px-2 py-0.5 rounded-full ${
                                                        isFake ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                                                    }`}>
                                                        {item.overall}
                                                    </span>
                                                </div>
                                                <span className="text-sm font-bold text-slate-600">
                                                    {item.deepfake?.confidence}%
                                                </span>
                                            </div>

                                            {/* Confidence bar */}
                                            <div className="w-full bg-slate-100 rounded-full h-1.5 mb-3">
                                                <div
                                                    className={`h-1.5 rounded-full transition-all duration-700 ${isFake ? 'bg-red-500' : 'bg-green-500'}`}
                                                    style={{ width: `${item.deepfake?.confidence}%` }}
                                                />
                                            </div>

                                            {/* Face / Heatmap / ELA tabs */}
                                            <VizTabs item={item} />
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                )}

                {/* Empty state */}
                {!result && !loading && !error && (
                    <div className="mt-4 text-center text-slate-400 text-sm">
                        <p>Upload a document to begin analysis. All processing is done locally — nothing is stored.</p>
                    </div>
                )}
            </main>
        </div>
    );
};

export default DocumentDetection;
