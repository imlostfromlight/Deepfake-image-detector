import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
    Shield, Upload, Loader2, AlertTriangle, CheckCircle2, FileImage,
    Monitor, FileText, Info, ChevronDown, ChevronUp, Users, Layers,
    Fingerprint, Eye, Search, Zap, Activity, Cpu, ScanLine, ShieldAlert,
    ShieldCheck, BarChart2, Mic2, Waves
} from 'lucide-react';

const API = import.meta.env.VITE_API_URL;

/* ─────────────────────────────────────────────────────────────────────────── */
/* Sub-components                                                              */
/* ─────────────────────────────────────────────────────────────────────────── */

const EmptySlot = ({ icon, text }) => (
    <div className="h-72 flex flex-col items-center justify-center text-slate-400 bg-slate-50 rounded-xl gap-2">
        {icon}<p className="text-sm">{text}</p>
    </div>
);

const FeatureCard = ({ icon, title, desc }) => (
    <div className="p-8 bg-white rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition group">
        <div className="w-12 h-12 bg-blue-50 rounded-xl flex items-center justify-center mb-6 group-hover:bg-blue-600 group-hover:text-white transition-colors">
            {icon}
        </div>
        <h3 className="text-xl font-bold mb-3">{title}</h3>
        <p className="text-slate-600 leading-relaxed text-sm">{desc}</p>
    </div>
);

const Step = ({ num, title, text }) => (
    <div className="flex gap-4">
        <div className="text-2xl font-black text-blue-100">{num}</div>
        <div>
            <h4 className="font-bold">{title}</h4>
            <p className="text-slate-600 text-sm">{text}</p>
        </div>
    </div>
);

/* Verdict header ─────────────────────────────────────────────────────────── */
const VerdictHeader = ({ result }) => {
    const isFake = result.overall !== 'real';
    const conf   = result.deepfake?.confidence;
    const thresh = result.analysis?.threshold;
    const overridden = result.analysis?.provenance_override;

    return (
        <div className={`relative overflow-hidden rounded-2xl ${isFake
            ? 'bg-gradient-to-br from-red-700 via-red-600 to-red-800'
            : 'bg-gradient-to-br from-green-700 via-emerald-600 to-green-800'
        }`}>
            {/* Background icon */}
            <div className="absolute right-4 top-1/2 -translate-y-1/2 opacity-10">
                {isFake
                    ? <ShieldAlert className="w-32 h-32 text-white" />
                    : <ShieldCheck  className="w-32 h-32 text-white" />
                }
            </div>

            <div className="relative px-6 py-5 flex items-center gap-5">
                <div className={`w-14 h-14 rounded-2xl flex items-center justify-center shrink-0 ${isFake ? 'bg-red-900/50' : 'bg-green-900/50'}`}>
                    {isFake
                        ? <AlertTriangle className="w-7 h-7 text-red-200" />
                        : <CheckCircle2  className="w-7 h-7 text-green-200" />
                    }
                </div>
                <div className="flex-1 text-white">
                    <p className="text-xs font-bold uppercase tracking-widest opacity-70 mb-0.5">
                        {overridden ? 'Metadata Override' : 'Verdict'}
                    </p>
                    <h2 className="text-3xl font-black tracking-tight uppercase">
                        {isFake ? 'Deepfake' : 'Authentic'}
                    </h2>
                    <p className="text-sm opacity-80 mt-0.5">
                        {overridden
                            ? 'IPTC / C2PA declares AI-generated — ML score ignored'
                            : `${conf}% confidence · threshold ${thresh}%`}
                        {result.faces?.length > 1 && (
                            <span className="ml-2 inline-flex items-center gap-1 bg-white/20 px-2 py-0.5 rounded-full text-xs font-semibold">
                                <Users className="w-3 h-3" /> {result.faces.length} faces
                            </span>
                        )}
                    </p>
                </div>
                <div className="text-right text-white">
                    <p className="text-4xl font-black">{conf}%</p>
                    <p className="text-xs opacity-60">confidence</p>
                </div>
            </div>

            {/* Confidence bar */}
            <div className="px-6 pb-4">
                <div className="w-full bg-black/20 rounded-full h-1.5 relative">
                    <div
                        className="h-1.5 rounded-full bg-white/80 transition-all duration-1000"
                        style={{ width: `${conf}%` }}
                    />
                    {thresh && (
                        <div className="absolute top-0 h-1.5 w-0.5 bg-yellow-300"
                            style={{ left: `${thresh}%` }}
                            title={`Threshold: ${thresh}%`} />
                    )}
                </div>
                <div className="flex justify-between text-xs text-white/50 mt-1">
                    <span>0%</span>
                    <span className="text-yellow-300/80">{thresh}% threshold</span>
                    <span>100%</span>
                </div>
            </div>
        </div>
    );
};

/* Generator card ─────────────────────────────────────────────────────────── */
const CONFIDENCE_STYLE = {
    definitive: { pill: 'bg-blue-500 text-white',           dot: 'bg-blue-400',   label: 'Definitive' },
    high:       { pill: 'bg-emerald-900 text-emerald-300',  dot: 'bg-emerald-400', label: 'High confidence' },
    medium:     { pill: 'bg-amber-900 text-amber-300',      dot: 'bg-amber-400',  label: 'Medium confidence' },
    none:       { pill: 'bg-slate-700 text-slate-400',      dot: 'bg-slate-500',  label: 'Unknown' },
};

const GeneratorCard = ({ provenance }) => {
    if (!provenance) return null;
    const hasInfo = provenance.generator || provenance.has_c2pa || provenance.has_iptc_ai || provenance.signals?.length > 0;
    if (!hasInfo) return (
        <div className="px-6 py-3 border-t border-slate-100 flex items-center gap-3 text-sm text-slate-400 bg-white">
            <Fingerprint className="w-4 h-4 shrink-0" />
            <span>No provenance metadata found — source unknown</span>
        </div>
    );

    const style = CONFIDENCE_STYLE[provenance.confidence] ?? CONFIDENCE_STYLE.none;
    return (
        <div className="px-6 py-4 border-t border-slate-800 bg-slate-900 flex items-start gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-900/50 flex items-center justify-center shrink-0 mt-0.5">
                <Fingerprint className="w-4 h-4 text-blue-400" />
            </div>
            <div className="flex-1 min-w-0">
                <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">AI Source Attribution</p>
                <p className="text-white font-bold truncate">{provenance.generator ?? 'Generator not identified'}</p>
                <div className="flex flex-wrap gap-2 mt-2">
                    {provenance.confidence !== 'none' && (
                        <span className={`inline-flex items-center gap-1.5 text-xs px-2 py-0.5 rounded-full font-semibold ${style.pill}`}>
                            <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />{style.label}
                        </span>
                    )}
                    {provenance.has_c2pa && (
                        <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-blue-900 text-blue-300 font-semibold">
                            🔏 C2PA verified
                        </span>
                    )}
                    {provenance.has_iptc_ai && (
                        <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-red-900 text-red-300 font-semibold">
                            <AlertTriangle className="w-3 h-3" /> IPTC: AI-generated
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
};

/* Score pipeline row ─────────────────────────────────────────────────────── */
const SCORE_COLOR = {
    blue:   { accent: 'border-blue-500',   num: 'text-blue-400',   bar: 'bg-blue-500'   },
    purple: { accent: 'border-purple-500', num: 'text-purple-400', bar: 'bg-purple-500' },
    green:  { accent: 'border-green-500',  num: 'text-green-400',  bar: 'bg-green-500'  },
    orange: { accent: 'border-amber-500',  num: 'text-amber-400',  bar: 'bg-amber-500'  },
    red:    { accent: 'border-red-500',    num: 'text-red-400',    bar: 'bg-red-500'    },
    gray:   { accent: 'border-slate-600',  num: 'text-slate-500',  bar: 'bg-slate-600'  },
};

const ScoreRow = ({ label, desc, value, max, color, prefix = '', unavailable }) => {
    const c   = SCORE_COLOR[color] ?? SCORE_COLOR.blue;
    const pct = Math.min(100, Math.max(0, ((value ?? 0) / max) * 100));
    return (
        <div className={`pl-3 border-l-2 ${c.accent} py-2`}>
            <div className="flex items-start justify-between gap-3 mb-1.5">
                <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-slate-200 leading-tight">{label}</p>
                    <p className="text-xs text-slate-500 mt-0.5 leading-snug">{desc}</p>
                </div>
                <span className={`text-lg font-black shrink-0 ${unavailable ? 'text-slate-600' : c.num}`}>
                    {unavailable ? '—' : `${prefix}${value?.toFixed?.(1) ?? '0.0'}%`}
                </span>
            </div>
            <div className="w-full bg-slate-800 rounded-full h-1">
                <div className={`h-1 rounded-full transition-all duration-1000 ${c.bar}`}
                    style={{ width: unavailable ? '0%' : `${pct}%` }} />
            </div>
        </div>
    );
};

/* Stat tile ──────────────────────────────────────────────────────────────── */
const TILE_COLOR = {
    green:  { bg: 'bg-emerald-900/40', border: 'border-emerald-700/50', val: 'text-emerald-300', dot: 'bg-emerald-500' },
    amber:  { bg: 'bg-amber-900/40',   border: 'border-amber-700/50',   val: 'text-amber-300',   dot: 'bg-amber-500'  },
    red:    { bg: 'bg-red-900/40',     border: 'border-red-700/50',     val: 'text-red-300',     dot: 'bg-red-500'    },
    blue:   { bg: 'bg-blue-900/40',    border: 'border-blue-700/50',    val: 'text-blue-300',    dot: 'bg-blue-500'   },
};

const StatTile = ({ label, value, note, color = 'green' }) => {
    const c = TILE_COLOR[color] ?? TILE_COLOR.green;
    return (
        <div className={`p-3 rounded-xl border ${c.bg} ${c.border}`}>
            <div className="flex items-center gap-1.5 mb-2">
                <span className={`w-2 h-2 rounded-full shrink-0 ${c.dot}`} />
                <p className="text-xs text-slate-400 font-medium leading-tight">{label}</p>
            </div>
            <p className={`text-xl font-black ${c.val}`}>{value}</p>
            <p className="text-xs text-slate-600 mt-0.5">{note}</p>
        </div>
    );
};

/* Signal badges ──────────────────────────────────────────────────────────── */
const SignalBadge = ({ text, ok }) => (
    <span className={`inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium ${
        ok ? 'bg-emerald-900/50 text-emerald-400 border border-emerald-800'
           : 'bg-amber-900/50 text-amber-400 border border-amber-800'
    }`}>
        {ok ? <CheckCircle2 className="w-3 h-3" /> : <AlertTriangle className="w-3 h-3" />}
        {text}
    </span>
);

const SignalBadges = ({ signals, okMsg }) => {
    if (!signals || signals.length === 0) return <SignalBadge text={okMsg} ok />;
    return (
        <div className="flex flex-wrap gap-2">
            {signals.map((s, i) => <SignalBadge key={i} text={s} ok={false} />)}
        </div>
    );
};

/* Section card ───────────────────────────────────────────────────────────── */
const ForensicSection = ({ title, icon, accent = 'border-slate-700', children }) => (
    <div className={`rounded-xl border border-slate-800 overflow-hidden`}>
        <div className={`flex items-center gap-2 px-4 py-2.5 bg-slate-800/60 border-b border-slate-700`}>
            <span className="text-slate-400">{icon}</span>
            <p className="text-xs font-bold uppercase tracking-widest text-slate-400">{title}</p>
        </div>
        <div className="p-4 space-y-3">{children}</div>
    </div>
);

/* Multi-face grid ────────────────────────────────────────────────────────── */
const FaceGrid = ({ faces }) => (
    <div className="px-6 py-4 bg-white border-t border-slate-100">
        <p className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-2">
            <Users className="w-4 h-4" /> {faces.length} faces detected in image
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {faces.map(f => (
                <div key={f.face_index} className={`p-2.5 rounded-xl border text-center ${
                    f.prediction === 'fake'
                        ? 'border-red-200 bg-red-50'
                        : 'border-green-200 bg-green-50'
                }`}>
                    <p className="text-xs text-slate-500 mb-1">Face #{f.face_index}</p>
                    <p className={`text-sm font-black uppercase ${f.prediction === 'fake' ? 'text-red-600' : 'text-green-600'}`}>
                        {f.prediction === 'fake' ? 'Deepfake' : 'Real'}
                    </p>
                    <p className="text-xs text-slate-400 mt-0.5">{f.fake_prob}%</p>
                </div>
            ))}
        </div>
    </div>
);

/* ─────────────────────────────────────────────────────────────────────────── */
/* Main Page                                                                   */
/* ─────────────────────────────────────────────────────────────────────────── */
const LandingPage = ({ onStartLive, onStartVideoCall, onStartDocument, onStartVideoLink }) => {
    const [imagePreview, setImagePreview] = useState(null);
    const [result, setResult]             = useState(null);
    const [loading, setLoading]           = useState(false);
    const [showAnalysis, setShowAnalysis] = useState(false);
    const [activeViz, setActiveViz]       = useState('original');

    const onDrop = useCallback(async (acceptedFiles) => {
        const file = acceptedFiles[0];
        if (!file) return;
        setImagePreview(URL.createObjectURL(file));
        setLoading(true);
        setResult(null);
        setShowAnalysis(false);
        setActiveViz('original');

        const fd = new FormData();
        fd.append('image', file);
        fd.append('source', 'upload');

        try {
            const res = await fetch(`${API}/api/detect/`, { method: 'POST', body: fd });
            if (!res.ok) throw new Error('Network response was not ok');
            setResult(await res.json());
        } catch {
            alert('Failed to connect to Django backend. Ensure server is running at port 8000.');
        } finally {
            setLoading(false);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.webp'] },
        multiple: false,
    });

    const isFake = result?.overall !== 'real';

    return (
        <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
            {/* Nav */}
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

                    {/* Drop zone */}
                    <div
                        {...getRootProps()}
                        className={`relative mt-10 p-12 border-2 border-dashed rounded-3xl transition-all cursor-pointer
                        ${isDragActive ? 'border-blue-500 bg-blue-50 scale-[1.02]' : 'border-slate-300 bg-white hover:border-blue-400'}`}
                    >
                        <input {...getInputProps()} />
                        <div className="flex flex-col items-center gap-4">
                            {loading
                                ? <Loader2 className="w-12 h-12 text-blue-600 animate-spin" />
                                : <Upload className={`w-12 h-12 ${isDragActive ? 'text-blue-600' : 'text-slate-400'}`} />
                            }
                            <div>
                                <p className="text-xl font-semibold">
                                    {isDragActive ? 'Drop the image here' : 'Drag & drop an image here'}
                                </p>
                                <p className="text-slate-500 mt-1 text-sm">Supports JPEG, PNG, WebP</p>
                            </div>
                        </div>
                    </div>

                    {/* Mode buttons */}
                    <div className="mt-8 flex flex-wrap justify-center gap-3">
                        <button onClick={onStartLive} className="flex items-center gap-2 bg-slate-900 text-white px-6 py-3 rounded-full hover:bg-slate-800 transition shadow-lg hover:shadow-xl transform hover:-translate-y-1">
                            <span className="relative flex h-3 w-3">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                                <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500" />
                            </span>
                            <span className="font-semibold">Live Camera Check</span>
                        </button>
                        <button onClick={onStartVideoCall} className="flex items-center gap-2 bg-blue-600 text-white px-6 py-3 rounded-full hover:bg-blue-700 transition shadow-lg hover:shadow-xl transform hover:-translate-y-1">
                            <Monitor className="w-4 h-4" /><span className="font-semibold">Video Call Scan</span>
                        </button>
                        <button onClick={onStartDocument} className="flex items-center gap-2 bg-violet-600 text-white px-6 py-3 rounded-full hover:bg-violet-700 transition shadow-lg hover:shadow-xl transform hover:-translate-y-1">
                            <FileText className="w-4 h-4" /><span className="font-semibold">Document Scan</span>
                        </button>
                        <button onClick={onStartVideoLink} className="flex items-center gap-2 bg-pink-600 text-white px-6 py-3 rounded-full hover:bg-pink-700 transition shadow-lg hover:shadow-xl transform hover:-translate-y-1">
                            <span className="font-semibold">🎵 TikTok / Reels / Shorts</span>
                        </button>
                    </div>

                    {/* ── Results section ── */}
                    <div className="mt-12 space-y-4 text-left">

                        {/* Visualization tabs */}
                        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                            <div className="flex border-b border-slate-100">
                                {[
                                    { key: 'original', label: 'Original' },
                                    { key: 'heatmap',  label: 'GradCAM++ Heatmap' },
                                    { key: 'ela',      label: 'ELA Map' },
                                ].map(tab => (
                                    <button key={tab.key} onClick={() => setActiveViz(tab.key)}
                                        className={`flex-1 py-2 text-xs font-semibold uppercase tracking-wider transition
                                        ${activeViz === tab.key ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-600 hover:bg-slate-50'}`}>
                                        {tab.label}
                                    </button>
                                ))}
                            </div>
                            <p className="text-xs text-slate-400 px-4 pt-2">
                                {activeViz === 'original' && 'Uploaded image'}
                                {activeViz === 'heatmap'  && 'Red = regions CLIP focused on when making the decision'}
                                {activeViz === 'ela'      && 'Error Level Analysis — bright areas reveal inconsistent JPEG compression'}
                            </p>
                            <div className="p-3">
                                {loading ? (
                                    <div className="h-72 flex flex-col items-center justify-center bg-slate-50 rounded-xl text-slate-400">
                                        <Loader2 className="w-8 h-8 animate-spin mb-2 text-blue-500" />
                                        <p className="text-sm animate-pulse">Analysing…</p>
                                    </div>
                                ) : activeViz === 'original' ? (
                                    imagePreview
                                        ? <img src={imagePreview} className="w-full h-72 object-cover rounded-xl" alt="Uploaded" />
                                        : <EmptySlot icon={<FileImage className="w-10 h-10 mb-2 opacity-30" />} text="No image uploaded" />
                                ) : activeViz === 'heatmap' ? (
                                    result?.heatmap
                                        ? <img src={result.heatmap} className="w-full h-72 object-cover rounded-xl" alt="GradCAM heatmap" />
                                        : <EmptySlot icon={<Eye className="w-10 h-10 mb-2 opacity-30" />} text="Heatmap appears after upload" />
                                ) : (
                                    result?.ela_map
                                        ? <img src={result.ela_map} className="w-full h-72 object-cover rounded-xl" alt="ELA map" />
                                        : <EmptySlot icon={<Layers className="w-10 h-10 mb-2 opacity-30" />} text="ELA map appears after upload" />
                                )}
                            </div>
                            {result?.ela_score !== undefined && activeViz === 'ela' && (
                                <div className="px-4 pb-3">
                                    <div className="flex justify-between text-xs text-slate-500 mb-1">
                                        <span>ELA mean score: <span className={`font-bold ${result.ela_score > 8 ? 'text-red-600' : result.ela_score > 4 ? 'text-amber-600' : 'text-green-600'}`}>{result.ela_score}</span></span>
                                        <span>{result.ela_score > 8 ? 'High — likely edited' : result.ela_score > 4 ? 'Moderate' : 'Low — consistent'}</span>
                                    </div>
                                    <div className="w-full bg-slate-100 rounded-full h-1.5">
                                        <div className={`h-1.5 rounded-full ${result.ela_score > 8 ? 'bg-red-500' : result.ela_score > 4 ? 'bg-amber-500' : 'bg-green-500'}`}
                                            style={{ width: `${Math.min(100, result.ela_score * 6)}%`, transition: 'width 0.8s ease' }} />
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Loading state */}
                        {loading ? (
                            <div className="bg-white p-8 rounded-2xl border border-slate-200 flex flex-col items-center justify-center">
                                <Loader2 className="w-8 h-8 text-blue-600 animate-spin mb-4" />
                                <p className="font-semibold animate-pulse">Running full forensic pipeline…</p>
                                <p className="text-xs text-slate-400 mt-1">CLIP · ELA · geometry · provenance · multi-face</p>
                            </div>
                        ) : result ? (

                            /* ── Result card ── */
                            <div className="rounded-2xl overflow-hidden border border-slate-200 shadow-sm">

                                <VerdictHeader result={result} />

                                {result.analysis?.provenance_override && (
                                    <div className="px-5 py-3 bg-red-950 flex items-center gap-2 text-xs text-red-300 font-semibold border-t border-red-900">
                                        <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                                        Metadata override active — IPTC / C2PA declares AI-generated. ML score is ignored.
                                    </div>
                                )}

                                <GeneratorCard provenance={result.provenance} />

                                {result.faces?.length > 1 && <FaceGrid faces={result.faces} />}

                                {/* Breakdown toggle */}
                                <button
                                    onClick={() => setShowAnalysis(v => !v)}
                                    className="w-full px-6 py-3.5 bg-slate-900 flex items-center justify-between text-sm font-semibold text-slate-300 hover:bg-slate-800 transition border-t border-slate-800"
                                >
                                    <span className="flex items-center gap-2">
                                        <BarChart2 className="w-4 h-4 text-blue-400" />
                                        Full forensic breakdown
                                    </span>
                                    {showAnalysis
                                        ? <ChevronUp className="w-4 h-4 text-slate-500" />
                                        : <ChevronDown className="w-4 h-4 text-slate-500" />}
                                </button>

                                {/* ── Forensic breakdown panel ── */}
                                {showAnalysis && (
                                    <div className="bg-slate-950 border-t border-slate-800 p-5 space-y-4">

                                        {/* Score pipeline */}
                                        <ForensicSection title="Score Pipeline" icon={<Activity className="w-4 h-4" />}>
                                            <ScoreRow
                                                label="CLIP ViT-B/16 + LoRA  (primary ML)"
                                                desc="Fine-tuned on face-swap deepfake datasets. This is the primary signal."
                                                value={result.analysis?.ml_score} max={100} color="blue" />
                                            <ScoreRow
                                                label="UniversalFakeDetect  (CLIP ViT-L/14)"
                                                desc="Linear probe trained on GAN + diffusion generators. Ensembled via max()."
                                                value={result.analysis?.ufd_score} max={100} color="purple"
                                                unavailable={result.analysis?.ufd_score == null} />
                                            <ScoreRow
                                                label="FFT frequency analysis"
                                                desc="GAN upsampling leaves periodic checkerboard artifacts in frequency spectrum."
                                                value={result.analysis?.fft_score} max={100} color="purple"
                                                unavailable={result.analysis?.fft_score == null} />
                                            <div className="pt-1 border-t border-slate-800 mt-1">
                                                <p className="text-xs text-slate-500 mb-2 font-semibold uppercase tracking-wider">Adjustments</p>
                                                <div className="space-y-2">
                                                    <ScoreRow
                                                        label="EXIF metadata"
                                                        desc={result.analysis?.meta_adjustment < 0 ? "Camera EXIF present → reduces fake probability" : "No camera EXIF → slight suspicion increase"}
                                                        value={Math.abs(result.analysis?.meta_adjustment ?? 0)} max={15}
                                                        color={result.analysis?.meta_adjustment < 0 ? "green" : "orange"}
                                                        prefix={result.analysis?.meta_adjustment < 0 ? "−" : "+"} />
                                                    <ScoreRow
                                                        label="Geometry & shadow"
                                                        desc="Shadow direction variance and face/background lighting asymmetry."
                                                        value={Math.abs(result.analysis?.geo_adjustment ?? 0)} max={12}
                                                        color={(result.analysis?.geo_adjustment ?? 0) > 0 ? "orange" : "green"}
                                                        prefix={(result.analysis?.geo_adjustment ?? 0) > 0 ? "+" : ""} />
                                                    <ScoreRow
                                                        label="Forensic signals  (CA · noise · eyes)"
                                                        desc="Chromatic aberration pattern, spatial noise CV, eye highlight symmetry."
                                                        value={Math.abs(result.analysis?.forensic_adjustment ?? 0)} max={10}
                                                        color={(result.analysis?.forensic_adjustment ?? 0) > 0 ? "orange" : "green"}
                                                        prefix={(result.analysis?.forensic_adjustment ?? 0) > 0 ? "+" : ""} />
                                                    <ScoreRow
                                                        label="Partial manipulation"
                                                        desc="Score discrepancy between upper and lower face halves."
                                                        value={Math.abs(result.analysis?.partial_adjustment ?? 0)} max={8}
                                                        color={(result.analysis?.partial_adjustment ?? 0) > 0 ? "orange" : "green"}
                                                        prefix={(result.analysis?.partial_adjustment ?? 0) > 0 ? "+" : ""} />
                                                    <ScoreRow
                                                        label="Provenance / metadata"
                                                        desc={result.analysis?.provenance_override
                                                            ? "IPTC or C2PA definitively declares AI-generated — ML score overridden."
                                                            : "AI generator identified in metadata or authenticated camera source."}
                                                        value={Math.abs(result.analysis?.provenance_adjustment ?? 0)} max={25}
                                                        color={result.analysis?.provenance_override ? "red"
                                                            : (result.analysis?.provenance_adjustment ?? 0) < 0 ? "green" : "orange"}
                                                        prefix={result.analysis?.provenance_override ? "OVERRIDE"
                                                            : (result.analysis?.provenance_adjustment ?? 0) < 0 ? "−"
                                                            : (result.analysis?.provenance_adjustment ?? 0) > 0 ? "+" : ""} />
                                                </div>
                                            </div>

                                            {/* Final verdict line */}
                                            <div className="mt-2 px-4 py-3 bg-slate-800 rounded-xl flex justify-between items-center">
                                                <span className="text-sm font-bold text-slate-300">Final score vs threshold</span>
                                                <span className={`text-lg font-black ${isFake ? 'text-red-400' : 'text-emerald-400'}`}>
                                                    {result.analysis?.provenance_override
                                                        ? '100% → DEEPFAKE'
                                                        : `${result.analysis?.final_score}% ${result.analysis?.final_score >= result.analysis?.threshold ? '> ' : '< '}${result.analysis?.threshold}% → ${isFake ? 'DEEPFAKE' : 'REAL'}`}
                                                </span>
                                            </div>
                                        </ForensicSection>

                                        {/* Forensic metrics */}
                                        {(result.forensic || result.geometry || result.partial) && (
                                            <ForensicSection title="Forensic Metrics" icon={<ScanLine className="w-4 h-4" />}>
                                                {result.forensic && (
                                                    <div>
                                                        <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-2">Hany Farid signals</p>
                                                        <div className="grid grid-cols-3 gap-2 mb-2">
                                                            {result.forensic.ca_score != null && (
                                                                <StatTile label="Chrom. aberration" value={`${result.forensic.ca_score}×`}
                                                                    note="periph/centre" color={result.forensic.ca_score < 0.8 ? 'amber' : 'green'} />
                                                            )}
                                                            {result.forensic.noise_cv != null && (
                                                                <StatTile label="Noise CV" value={result.forensic.noise_cv}
                                                                    note="coeff. of variation" color={result.forensic.noise_cv > 1.2 ? 'red' : result.forensic.noise_cv > 0.9 ? 'amber' : 'green'} />
                                                            )}
                                                            {result.forensic.eye_highlight_delta != null && (
                                                                <StatTile label="Eye highlight Δ" value={result.forensic.eye_highlight_delta}
                                                                    note="0 = perfect match" color={result.forensic.eye_highlight_delta > 0.55 ? 'red' : result.forensic.eye_highlight_delta > 0.35 ? 'amber' : 'green'} />
                                                            )}
                                                        </div>
                                                        <SignalBadges signals={result.forensic.signals} okMsg="CA, noise, and eye highlights consistent" />
                                                    </div>
                                                )}

                                                {result.geometry && (
                                                    <div className="pt-3 border-t border-slate-800">
                                                        <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-2">Geometry & shadow</p>
                                                        <div className="grid grid-cols-2 gap-2 mb-2">
                                                            {result.geometry.shadow_variance != null && (
                                                                <StatTile label="Shadow variance" value={result.geometry.shadow_variance}
                                                                    note="0=uniform · 1=chaotic" color={result.geometry.shadow_variance > 0.92 ? 'red' : result.geometry.shadow_variance > 0.82 ? 'amber' : 'green'} />
                                                            )}
                                                            {result.geometry.face_bg_lighting_delta != null && (
                                                                <StatTile label="Face/scene light Δ"
                                                                    value={result.geometry.face_bg_lighting_delta > 0 ? `+${result.geometry.face_bg_lighting_delta}` : '✓'}
                                                                    note="brightness asymmetry" color={result.geometry.face_bg_lighting_delta > 0 ? 'amber' : 'green'} />
                                                            )}
                                                        </div>
                                                        <SignalBadges signals={result.geometry.signals} okMsg="Shadow and lighting direction consistent" />
                                                    </div>
                                                )}

                                                {result.partial && (
                                                    <div className="pt-3 border-t border-slate-800">
                                                        <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-2">Partial face manipulation</p>
                                                        <div className="grid grid-cols-3 gap-2 mb-2">
                                                            {result.partial.upper_score != null && (
                                                                <StatTile label="Upper face" value={`${result.partial.upper_score}%`}
                                                                    note="forehead · eyes" color={result.partial.upper_score > 50 ? 'red' : 'green'} />
                                                            )}
                                                            {result.partial.lower_score != null && (
                                                                <StatTile label="Lower face" value={`${result.partial.lower_score}%`}
                                                                    note="nose · mouth" color={result.partial.lower_score > 50 ? 'red' : 'green'} />
                                                            )}
                                                            {result.partial.discrepancy != null && (
                                                                <StatTile label="Discrepancy" value={`${result.partial.discrepancy}pt`}
                                                                    note="<15 = consistent" color={result.partial.discrepancy > 25 ? 'red' : result.partial.discrepancy > 15 ? 'amber' : 'green'} />
                                                            )}
                                                        </div>
                                                        <SignalBadges signals={result.partial.signals} okMsg="Upper and lower face are consistent" />
                                                    </div>
                                                )}
                                            </ForensicSection>
                                        )}

                                        {/* Provenance */}
                                        <ForensicSection title="Provenance & Content Credentials" icon={<Fingerprint className="w-4 h-4" />}>
                                            <div className="flex flex-wrap gap-2">
                                                {result.provenance?.has_c2pa && (
                                                    <SignalBadge text="C2PA cryptographic manifest present" ok={false} />
                                                )}
                                                {result.provenance?.has_iptc_ai && (
                                                    <SignalBadge text="IPTC declares AI-generated" ok={false} />
                                                )}
                                                {result.provenance?.signals?.map((s, i) => (
                                                    <SignalBadge key={i} text={s} ok={false} />
                                                ))}
                                                {result.metadata?.authenticity_signals?.map((s, i) => (
                                                    <SignalBadge key={i} text={s} ok />
                                                ))}
                                                {result.metadata?.suspicion_signals?.map((s, i) => (
                                                    <SignalBadge key={i} text={s} ok={false} />
                                                ))}
                                                {!result.provenance?.has_c2pa && !result.provenance?.has_iptc_ai &&
                                                    !result.provenance?.signals?.length &&
                                                    !result.metadata?.authenticity_signals?.length &&
                                                    !result.metadata?.suspicion_signals?.length && (
                                                    <SignalBadge text="No provenance signals detected" ok />
                                                )}
                                            </div>
                                        </ForensicSection>

                                        {/* Visual tips */}
                                        <ForensicSection title="Visual tells — how to spot deepfakes" icon={<Eye className="w-4 h-4" />}>
                                            <div className="grid grid-cols-2 gap-2">
                                                {[
                                                    ['Face edges',      'Blurring or colour mismatch where face meets hair/neck'],
                                                    ['Teeth & mouth',   'Often smeared, wrong count, or unnaturally bright'],
                                                    ['Eye reflections', 'Light source inconsistent with scene'],
                                                    ['Skin texture',    'Overly smooth — no pores, uniform pigmentation'],
                                                    ['Ear detail',      'Frequently missing, misshapen, or wrong angle'],
                                                    ['Background halo', 'Slight glow or blur ring around face outline'],
                                                ].map(([t, d]) => (
                                                    <div key={t} className="p-2.5 rounded-lg bg-slate-800/60 border border-slate-700/50">
                                                        <p className="text-xs font-bold text-slate-300 mb-0.5">{t}</p>
                                                        <p className="text-xs text-slate-500">{d}</p>
                                                    </div>
                                                ))}
                                            </div>
                                        </ForensicSection>

                                    </div>
                                )}
                            </div>

                        ) : (
                            <div className="bg-slate-100 p-8 rounded-2xl border border-slate-200 flex items-center justify-center text-slate-400">
                                <p>Results will appear here after upload</p>
                            </div>
                        )}
                    </div>
                </div>
            </header>

            {/* Stats bar */}
            <section className="py-12 bg-white border-y border-slate-200">
                <div className="max-w-7xl mx-auto px-4 flex flex-wrap justify-around gap-8 text-center">
                    <div><div className="text-3xl font-bold">90%+</div><div className="text-sm text-slate-500 uppercase tracking-widest font-semibold">Model Accuracy</div></div>
                    <div><div className="text-3xl font-bold">CNN-Based</div><div className="text-sm text-slate-500 uppercase tracking-widest font-semibold">Architecture</div></div>
                    <div><div className="text-3xl font-bold">Real-Time</div><div className="text-sm text-slate-500 uppercase tracking-widest font-semibold">Inference</div></div>
                </div>
            </section>

            <section id="features" className="py-24 px-4 bg-slate-50">
                <div className="max-w-7xl mx-auto">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl md:text-4xl font-bold mb-4">Why Automated Detection?</h2>
                        <p className="text-slate-600 max-w-xl mx-auto text-lg">Human eyes struggle with subtle deepfake artifacts — automated ML is essential.</p>
                    </div>
                    <div className="grid md:grid-cols-3 gap-8">
                        <FeatureCard icon={<Search className="w-6 h-6 text-blue-600" />} title="CLIP + LoRA Detector" desc="Fine-tuned CLIP ViT-B/16 with LoRA adapters for robust deepfake feature extraction." />
                        <FeatureCard icon={<Eye className="w-6 h-6 text-blue-600" />} title="Explainable AI (XAI)" desc="GradCAM++ heatmaps visualise where the model focuses when making its decision." />
                        <FeatureCard icon={<Zap className="w-6 h-6 text-blue-600" />} title="Multi-Signal Pipeline" desc="6-layer forensic pipeline: ML · ELA · geometry · forensic signals · provenance · C2PA." />
                    </div>
                </div>
            </section>

            <section id="workflow" className="py-24 bg-white px-4">
                <div className="max-w-5xl mx-auto">
                    <div className="flex flex-col md:flex-row items-center gap-12">
                        <div className="flex-1">
                            <h2 className="text-3xl font-bold mb-6">Simple 3-Step Verification</h2>
                            <div className="space-y-6">
                                <Step num="01" title="Upload Image" text="Upload any suspicious image through our intuitive web interface." />
                                <Step num="02" title="ML Analysis" text="CLIP + forensic pipeline analyses the image for deepfake artifacts." />
                                <Step num="03" title="Review Results" text="Receive a verdict with score breakdown, heatmaps, and ELA visualisation." />
                            </div>
                        </div>
                        <div className="flex-1 bg-slate-900 rounded-3xl p-8 text-white relative overflow-hidden">
                            <div className="absolute top-0 right-0 p-4 opacity-10"><Shield className="w-32 h-32" /></div>
                            <h3 className="text-xl font-bold mb-4">Security First</h3>
                            <p className="text-slate-400 mb-6">Challenge-response liveness check, HMAC-signed tokens, and C2PA provenance verification.</p>
                            <div className="flex gap-2 flex-wrap">
                                {['HMAC tokens', 'C2PA', 'IPTC', 'Challenge-response', 'Django API'].map(t => (
                                    <span key={t} className="px-3 py-1 bg-slate-800 rounded-full text-xs">{t}</span>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </section>

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
                        <p className="text-slate-400 text-sm">Student: Nadyrkhan Shyntemir Nurlanuly</p>
                        <p className="text-slate-400 text-sm">Cybersecurity Program, 6B06301</p>
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default LandingPage;
