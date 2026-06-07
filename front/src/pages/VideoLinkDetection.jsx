import { useState, useRef } from 'react';
import { ArrowLeft, Link, Play, CheckCircle2, AlertTriangle, Loader2, Film } from 'lucide-react';

const PLATFORM_PATTERNS = [
    { name: 'TikTok',    color: '#010101', pattern: /tiktok\.com|vm\.tiktok/i },
    { name: 'Instagram', color: '#E1306C', pattern: /instagram\.com/i },
    { name: 'YouTube',   color: '#FF0000', pattern: /youtube\.com\/shorts|youtu\.be/i },
];

function detectPlatform(url) {
    return PLATFORM_PATTERNS.find(p => p.pattern.test(url)) ?? null;
}

function FrameTimeline({ frames, threshold }) {
    if (!frames?.length) return null;
    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-400">
                <span>Frame-by-frame analysis ({frames.length} frames with faces)</span>
                <span>threshold {threshold}%</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
                {frames.map((f, i) => (
                    <div
                        key={i}
                        title={`${f.timestamp}s — ${f.prediction} (${f.fake_prob}%)`}
                        className={`w-4 h-4 rounded-sm cursor-default transition-transform hover:scale-125 ${
                            f.prediction === 'fake' ? 'bg-red-500' : 'bg-green-500'
                        }`}
                        style={{ opacity: 0.4 + (f.fake_prob / 100) * 0.6 }}
                    />
                ))}
            </div>
            <div className="flex gap-4 text-xs text-slate-500">
                <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-green-500 inline-block" /> Real</span>
                <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-red-500 inline-block" /> Deepfake</span>
                <span className="text-slate-600">Opacity = confidence</span>
            </div>
        </div>
    );
}

const VideoLinkDetection = ({ onBack }) => {
    const [url, setUrl]         = useState('');
    const [status, setStatus]   = useState('idle');   // idle | loading | done | error
    const [result, setResult]   = useState(null);
    const [errorMsg, setErrorMsg] = useState('');
    const abortRef = useRef(null);

    const platform = detectPlatform(url);

    const analyze = async () => {
        if (!url.trim()) return;
        setStatus('loading');
        setResult(null);
        setErrorMsg('');

        const controller = new AbortController();
        abortRef.current = controller;

        try {
            const res = await fetch(`${import.meta.env.VITE_API_URL}/api/detect/video/`, {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ url: url.trim() }),
                signal:  controller.signal,
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Server error');
            setResult(data);
            setStatus('done');
        } catch (e) {
            if (e.name === 'AbortError') return;
            setErrorMsg(e.message || 'Something went wrong');
            setStatus('error');
        }
    };

    const reset = () => {
        abortRef.current?.abort();
        setUrl('');
        setStatus('idle');
        setResult(null);
        setErrorMsg('');
    };

    const isFake   = result?.overall === 'deepfake';
    const summary  = result?.summary;

    return (
        <div className="min-h-screen bg-slate-900 text-white flex flex-col">
            {/* Nav */}
            <nav className="p-4 border-b border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Film className="w-6 h-6 text-pink-500" />
                    <span className="font-bold text-lg">DeepVerify — Social Media</span>
                </div>
                <button onClick={onBack} className="flex items-center gap-2 px-4 py-2 hover:bg-slate-800 rounded-lg transition">
                    <ArrowLeft className="w-4 h-4" /> Back
                </button>
            </nav>

            <main className="flex-1 flex flex-col items-center p-6 gap-6 max-w-2xl mx-auto w-full">

                {/* Platform badges */}
                <div className="flex gap-2 text-xs">
                    {PLATFORM_PATTERNS.map(p => (
                        <span
                            key={p.name}
                            className="px-3 py-1 rounded-full border text-slate-300"
                            style={{ borderColor: p.color + '66', background: p.color + '22' }}
                        >
                            {p.name}
                        </span>
                    ))}
                </div>

                {/* URL input */}
                <div className="w-full space-y-3">
                    <div className="relative">
                        <Link className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                        <input
                            type="url"
                            value={url}
                            onChange={e => { setUrl(e.target.value); setStatus('idle'); setErrorMsg(''); }}
                            onKeyDown={e => e.key === 'Enter' && status !== 'loading' && analyze()}
                            placeholder="Paste TikTok, Instagram Reel, or YouTube Shorts link…"
                            disabled={status === 'loading'}
                            className="w-full bg-slate-800 border border-slate-700 rounded-xl pl-10 pr-4 py-3 text-sm focus:outline-none focus:border-pink-500 focus:ring-1 focus:ring-pink-500 placeholder-slate-500 disabled:opacity-50"
                        />
                        {/* Detected platform label */}
                        {platform && (
                            <span
                                className="absolute right-3 top-1/2 -translate-y-1/2 text-xs px-2 py-0.5 rounded-full font-medium"
                                style={{ background: platform.color + '33', color: platform.color }}
                            >
                                {platform.name}
                            </span>
                        )}
                    </div>

                    <button
                        onClick={analyze}
                        disabled={!url.trim() || status === 'loading'}
                        className="w-full py-3 rounded-xl font-bold bg-pink-600 hover:bg-pink-700 disabled:opacity-40 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
                    >
                        {status === 'loading'
                            ? <><Loader2 className="w-4 h-4 animate-spin" /> Downloading &amp; analyzing…</>
                            : <><Play className="w-4 h-4" /> Analyze Video</>
                        }
                    </button>
                </div>

                {/* Loading hint */}
                {status === 'loading' && (
                    <div className="w-full bg-slate-800/60 rounded-xl p-4 text-sm text-slate-400 space-y-1 border border-slate-700">
                        <p>⬇ Downloading video at lowest quality…</p>
                        <p>🔍 Extracting frames and detecting faces…</p>
                        <p>🧠 Running CLIP deepfake model on each face…</p>
                        <p className="text-slate-600">This takes 20–60 seconds depending on video length.</p>
                    </div>
                )}

                {/* Error */}
                {status === 'error' && (
                    <div className="w-full bg-red-900/30 border border-red-700/50 rounded-xl p-4 text-sm text-red-300">
                        <p className="font-semibold mb-1">Analysis failed</p>
                        <p>{errorMsg}</p>
                        <button onClick={reset} className="mt-3 text-xs text-red-400 underline">Try another link</button>
                    </div>
                )}

                {/* Results */}
                {status === 'done' && result && (
                    <div className="w-full space-y-4">

                        {/* Verdict card */}
                        <div className={`rounded-2xl p-5 border ${
                            isFake
                                ? 'bg-red-900/30 border-red-700/50'
                                : 'bg-green-900/30 border-green-700/50'
                        }`}>
                            <div className="flex items-start gap-4">
                                {isFake
                                    ? <AlertTriangle className="w-10 h-10 text-red-400 shrink-0 mt-0.5" />
                                    : <CheckCircle2 className="w-10 h-10 text-green-400 shrink-0 mt-0.5" />
                                }
                                <div className="flex-1 min-w-0">
                                    <div className={`text-2xl font-black ${isFake ? 'text-red-300' : 'text-green-300'}`}>
                                        {isFake ? 'DEEPFAKE DETECTED' : 'APPEARS REAL'}
                                    </div>
                                    <p className="text-sm text-slate-400 mt-0.5 truncate">{result.title}</p>
                                    <p className="text-xs text-slate-500 mt-0.5">
                                        {result.platform} · {result.duration}s
                                    </p>
                                </div>
                            </div>
                        </div>

                        {/* Stats */}
                        {summary && (
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                                {[
                                    { label: 'Avg fake prob', value: `${summary.avg_fake_prob}%`,  color: isFake ? 'text-red-400' : 'text-green-400' },
                                    { label: 'Peak fake prob', value: `${summary.max_fake_prob}%`, color: summary.max_fake_prob >= summary.threshold ? 'text-red-400' : 'text-slate-300' },
                                    { label: 'Fake frames',    value: `${summary.fake_frames}/${summary.total_frames}`, color: 'text-slate-300' },
                                    { label: 'Threshold',      value: `${summary.threshold}%`,     color: 'text-slate-500' },
                                ].map(s => (
                                    <div key={s.label} className="bg-slate-800/60 rounded-xl p-3 border border-slate-700">
                                        <div className={`text-lg font-bold ${s.color}`}>{s.value}</div>
                                        <div className="text-xs text-slate-500 mt-0.5">{s.label}</div>
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Frame timeline */}
                        <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700">
                            <FrameTimeline frames={result.frames} threshold={summary?.threshold} />
                        </div>

                        <button onClick={reset} className="w-full py-2.5 rounded-xl text-sm text-slate-400 hover:text-white border border-slate-700 hover:border-slate-500 transition">
                            Analyze another video
                        </button>
                    </div>
                )}

                {/* Info footer */}
                {status === 'idle' && (
                    <div className="w-full bg-slate-800/40 rounded-xl p-4 border border-slate-700/50 text-xs text-slate-500 space-y-1">
                        <p>• Only <strong className="text-slate-400">public</strong> videos are supported</p>
                        <p>• Maximum video length: <strong className="text-slate-400">5 minutes</strong></p>
                        <p>• Instagram Reels may require a direct post link (not a story)</p>
                        <p>• Videos are downloaded at lowest quality and discarded after analysis</p>
                    </div>
                )}
            </main>
        </div>
    );
};

export default VideoLinkDetection;
