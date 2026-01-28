import { useState, useEffect, useRef } from 'react';
import { FileText, Sparkles, ChevronDown, ChevronUp, Download, Cpu, Settings, Cloud, Key } from 'lucide-react';
import sampleText from './sample-text.txt?raw';

const webllmWorkerCode = `
import * as webllm from 'https://esm.run/@mlc-ai/web-llm';
let engine = null;
self.onmessage = async (e) => {
  const { type, payload } = e.data;
  if (type === 'load') {
    try {
      engine = await webllm.CreateMLCEngine(payload.modelId, {
        initProgressCallback: (p) => self.postMessage({ type: 'progress', payload: p.progress || 0 })
      });
      self.postMessage({ type: 'ready' });
    } catch (err) { self.postMessage({ type: 'error', payload: err.message }); }
  }
  if (type === 'generate') {
    try {
      const r = await engine.chat.completions.create({ messages: [{ role: 'user', content: payload.prompt }], max_tokens: 1000, temperature: 0.3 });
      self.postMessage({ type: 'result', payload: r.choices[0]?.message?.content });
    } catch (err) { self.postMessage({ type: 'error', payload: err.message }); }
  }
};
`;

const transformersWorkerCode = `
import { AutoTokenizer, AutoModelForCausalLM } from 'https://esm.run/@huggingface/transformers';
let tokenizer = null, model = null;
self.onmessage = async (e) => {
  const { type, payload } = e.data;
  if (type === 'load') {
    try {
      self.postMessage({ type: 'progress', payload: 0.1 });
      tokenizer = await AutoTokenizer.from_pretrained(payload.modelId, { progress_callback: (p) => { if (p.progress) self.postMessage({ type: 'progress', payload: p.progress / 200 }); } });
      self.postMessage({ type: 'progress', payload: 0.5 });
      model = await AutoModelForCausalLM.from_pretrained(payload.modelId, { dtype: 'q4', progress_callback: (p) => { if (p.progress) self.postMessage({ type: 'progress', payload: 0.5 + p.progress / 200 }); } });
      self.postMessage({ type: 'ready' });
    } catch (err) { self.postMessage({ type: 'error', payload: err.message }); }
  }
  if (type === 'generate') {
    try {
      // Format as chat message for instruction-tuned models
      const messages = [{ role: 'user', content: payload.prompt }];
      const inputs = tokenizer.apply_chat_template(messages, { add_generation_prompt: true, return_dict: true });
      const output = await model.generate({ ...inputs, max_new_tokens: 800, temperature: 0.5, do_sample: true });
      const promptLength = inputs.input_ids.dims[1];
      const newTokens = output.slice(null, [promptLength, null]);
      const result = tokenizer.batch_decode(newTokens, { skip_special_tokens: true })[0].trim();
      self.postMessage({ type: 'result', payload: result });
    } catch (err) { self.postMessage({ type: 'error', payload: err.message }); }
  }
};
`;

export default function TextSummarizer() {
  const [text, setText] = useState(sampleText);
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const [error, setError] = useState('');
  const [modelStatus, setModelStatus] = useState('idle');
  const [downloadProgress, setDownloadProgress] = useState(0);
  const workerRef = useRef(null);

  const [generationTime, setGenerationTime] = useState(0);
  const timerRef = useRef(null);

  const [detailLevel, setDetailLevel] = useState(3);
  const [bulletPoints, setBulletPoints] = useState(true);
  const [tone, setTone] = useState('neutral');
  const [focus, setFocus] = useState('general');
  const [audience, setAudience] = useState('general');
  const [customPrompt, setCustomPrompt] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [isLocalEnv, setIsLocalEnv] = useState(true);

  useEffect(() => {
    // Detect if running in Claude artifact preview (restricted environment)
    try {
      const test = new Worker(URL.createObjectURL(new Blob([''], { type: 'application/javascript' })), { type: 'module' });
      test.terminate();
    } catch { setIsLocalEnv(false); }
  }, []);

  const [selectedModel, setSelectedModel] = useState(() => localStorage.getItem('selectedModel') || 'phi-3.5');
  const shouldAutoLoad = useRef(localStorage.getItem('modelLoaded') === 'true');
  const [downloadedModels, setDownloadedModels] = useState(() => {
    try { return JSON.parse(localStorage.getItem('downloadedModels') || '[]'); }
    catch { return []; }
  });
  const isSelectedModelCached = downloadedModels.includes(selectedModel);
  const [hasWebGPU, setHasWebGPU] = useState(null);
  const [forceGPU, setForceGPU] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const detailLabels = ['Brief', 'Concise', 'Moderate', 'Detailed', 'Comprehensive'];

  const webgpuModels = [
    { id: 'phi-3.5', name: 'Phi-3.5 Mini', mlcId: 'Phi-3.5-mini-instruct-q4f16_1-MLC', size: '~2.0 GB', desc: 'Best quality' },
    { id: 'llama-3.2-3b', name: 'Llama 3.2 3B', mlcId: 'Llama-3.2-3B-Instruct-q4f16_1-MLC', size: '~1.8 GB', desc: 'Fast & capable' },
    { id: 'llama-3.2-1b', name: 'Llama 3.2 1B', mlcId: 'Llama-3.2-1B-Instruct-q4f16_1-MLC', size: '~0.9 GB', desc: 'Lightweight' },
    { id: 'qwen-1.5b', name: 'Qwen 2.5 1.5B', mlcId: 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC', size: '~1.1 GB', desc: 'Multilingual' }
  ];

  const cpuModels = [
    { id: 'qwen-0.5b-cpu', name: 'Qwen2.5 0.5B', hfId: 'onnx-community/Qwen2.5-0.5B-Instruct', size: '~350 MB', desc: 'Best for CPU' }
  ];

  const cloudModels = [
    { id: 'groq-llama', name: 'Llama 3.3 70B', provider: 'groq', model: 'llama-3.3-70b-versatile', desc: 'Groq — free tier, very fast' },
    { id: 'openai-gpt4o-mini', name: 'GPT-4o Mini', provider: 'openai', model: 'gpt-4o-mini', desc: 'Fast & affordable' },
    { id: 'openai-gpt4o', name: 'GPT-4o', provider: 'openai', model: 'gpt-4o', desc: 'Most capable' },
  ];

  const [apiKeys, setApiKeys] = useState(() => {
    try { return JSON.parse(localStorage.getItem('apiKeys') || '{}'); }
    catch { return {}; }
  });

  const isCloudModel = cloudModels.some(m => m.id === selectedModel);
  const allLocalModels = [...webgpuModels, ...cpuModels];
  const models = allLocalModels; // Keep for backward compat
  const selectedCloudModel = cloudModels.find(m => m.id === selectedModel);
  const currentProvider = selectedCloudModel?.provider;
  const hasApiKey = currentProvider && apiKeys[currentProvider];

  const saveApiKey = (provider, key) => {
    const updated = { ...apiKeys, [provider]: key };
    setApiKeys(updated);
    localStorage.setItem('apiKeys', JSON.stringify(updated));
  };

  const callCloudApi = async (prompt) => {
    const model = selectedCloudModel;
    const key = apiKeys[model.provider];

    if (model.provider === 'openai') {
      const res = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${key}` },
        body: JSON.stringify({ model: model.model, messages: [{ role: 'user', content: prompt }], max_tokens: 1000 })
      });
      if (!res.ok) throw new Error((await res.json()).error?.message || res.statusText);
      const data = await res.json();
      return data.choices[0].message.content;
    }

    if (model.provider === 'gemini') {
      const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model.model}:generateContent?key=${key}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] })
      });
      if (!res.ok) throw new Error((await res.json()).error?.message || res.statusText);
      const data = await res.json();
      return data.candidates[0].content.parts[0].text;
    }

    if (model.provider === 'groq') {
      const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${key}` },
        body: JSON.stringify({ model: model.model, messages: [{ role: 'user', content: prompt }], max_tokens: 1000 })
      });
      if (!res.ok) throw new Error((await res.json()).error?.message || res.statusText);
      const data = await res.json();
      return data.choices[0].message.content;
    }

    throw new Error('Unknown provider');
  };

  const toneOptions = [
    { value: 'neutral', label: 'Neutral' },
    { value: 'formal', label: 'Formal' },
    { value: 'casual', label: 'Casual' },
    { value: 'technical', label: 'Technical' }
  ];

  const focusOptions = [
    { value: 'general', label: 'General' },
    { value: 'key-facts', label: 'Key Facts' },
    { value: 'action-items', label: 'Action Items' },
    { value: 'themes', label: 'Main Themes' },
    { value: 'arguments', label: 'Arguments' }
  ];

  const audienceOptions = [
    { value: 'general', label: 'General' },
    { value: 'expert', label: 'Expert' },
    { value: 'beginner', label: 'Beginner' },
    { value: 'executive', label: 'Executive' }
  ];

  useEffect(() => {
    let mounted = true;

    // Detect mobile
    const mobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    setIsMobile(mobile);

    // On mobile, default to Groq cloud model
    if (mobile && !localStorage.getItem('selectedModel')) {
      setSelectedModel('groq-llama');
    }

    const checkWebGPU = async () => {
      try {
        if (navigator.gpu) {
          const adapter = await navigator.gpu.requestAdapter();
          if (!mounted) return;
          setHasWebGPU(!!adapter);
          if (!adapter && !localStorage.getItem('selectedModel') && !mobile) setSelectedModel('qwen-0.5b-cpu');
        } else {
          if (!mounted) return;
          setHasWebGPU(false);
          if (!localStorage.getItem('selectedModel') && !mobile) setSelectedModel('qwen-0.5b-cpu');
        }
      } catch {
        if (!mounted) return;
        setHasWebGPU(false);
        if (!localStorage.getItem('selectedModel') && !mobile) setSelectedModel('qwen-0.5b-cpu');
      }
    };
    checkWebGPU();
    return () => { mounted = false; workerRef.current?.terminate(); clearInterval(timerRef.current); };
  }, []);

  const loadModel = () => {
    if (modelStatus === 'ready' || modelStatus === 'loading') return;
    setModelStatus('loading');
    setError('');

    // Determine if selected model is GPU or CPU based on which array it's in
    const isGpuModel = webgpuModels.some(m => m.id === selectedModel);
    const workerCode = isGpuModel ? webllmWorkerCode : transformersWorkerCode;
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const worker = new Worker(URL.createObjectURL(blob), { type: 'module' });
    workerRef.current = worker;

    worker.onmessage = (e) => {
      const { type, payload } = e.data;
      if (type === 'progress') setDownloadProgress(Math.round(payload * 100));
      if (type === 'ready') {
        setModelStatus('ready');
        localStorage.setItem('modelLoaded', 'true');
        localStorage.setItem('selectedModel', selectedModel);
        // Track this model as downloaded
        setDownloadedModels(prev => {
          if (!prev.includes(selectedModel)) {
            const updated = [...prev, selectedModel];
            localStorage.setItem('downloadedModels', JSON.stringify(updated));
            return updated;
          }
          return prev;
        });
      }
      if (type === 'result') { setSummary(payload); setLoading(false); clearInterval(timerRef.current); }
      if (type === 'error') { setError(payload); setLoading(false); clearInterval(timerRef.current); setModelStatus(prev => prev === 'loading' ? 'idle' : prev); }
    };

    const gpuModel = webgpuModels.find(m => m.id === selectedModel);
    const cpuModel = cpuModels.find(m => m.id === selectedModel);
    const modelId = gpuModel ? gpuModel.mlcId : cpuModel?.hfId;
    worker.postMessage({ type: 'load', payload: { modelId } });
  };

  // Auto-load model if previously loaded
  useEffect(() => {
    if (shouldAutoLoad.current && hasWebGPU !== null && modelStatus === 'idle') {
      const timer = setTimeout(() => loadModel(), 0);
      return () => clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasWebGPU]);

  const buildPrompt = () => {
    if (customPrompt.trim()) {
      return customPrompt.replace('{text}', text);
    }

    const detailInstructions = {
      0: 'Provide a very brief 1-2 sentence summary capturing only the core message.',
      1: 'Provide a concise summary in 3-4 sentences covering the main points.',
      2: 'Provide a moderate summary covering key points and some supporting details.',
      3: 'Provide a detailed summary with main points, supporting details, and examples.',
      4: 'Provide a comprehensive summary capturing all significant points, details, nuances, and examples.'
    };

    const toneInstructions = {
      neutral: '',
      formal: 'Use formal, professional language.',
      casual: 'Use casual, conversational language.',
      technical: 'Use precise technical terminology.'
    };

    const focusInstructions = {
      general: '',
      'key-facts': 'Focus on extracting key facts and data points.',
      'action-items': 'Focus on identifying action items and next steps.',
      themes: 'Focus on identifying main themes and patterns.',
      arguments: 'Focus on the main arguments and supporting evidence.'
    };

    const audienceInstructions = {
      general: '',
      expert: 'Write for an expert audience familiar with the subject.',
      beginner: 'Write for beginners, explaining any complex concepts simply.',
      executive: 'Write for busy executives who need the bottom line quickly.'
    };

    const format = bulletPoints ? 'Use bullet points to organize the information.' : 'Write in flowing paragraphs.';

    const parts = [
      'Summarize the following text.',
      detailInstructions[detailLevel],
      toneInstructions[tone],
      focusInstructions[focus],
      audienceInstructions[audience],
      format
    ].filter(Boolean);

    return `${parts.join(' ')}\n\nText:\n${text}\n\nSummary:`;
  };

  const summarize = async () => {
    if (!text.trim()) return;
    if (isCloudModel) {
      if (!hasApiKey) return;
      setLoading(true);
      setError('');
      setSummary('');
      setGenerationTime(0);
      timerRef.current = setInterval(() => setGenerationTime(t => t + 0.1), 100);
      try {
        const result = await callCloudApi(buildPrompt());
        setSummary(result);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
        clearInterval(timerRef.current);
      }
    } else {
      if (!workerRef.current || modelStatus !== 'ready') return;
      setLoading(true);
      setError('');
      setSummary('');
      setGenerationTime(0);
      timerRef.current = setInterval(() => setGenerationTime(t => t + 0.1), 100);
      workerRef.current.postMessage({ type: 'generate', payload: { prompt: buildPrompt() } });
    }
  };

  const SelectGroup = ({ label, value, onChange, options }) => (
    <div>
      <label className="text-purple-200 text-sm mb-2 block">{label}</label>
      <div className="flex flex-wrap gap-2">
        {options.map(opt => (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${value === opt.value ? 'bg-purple-500 text-white' : 'bg-white/10 text-purple-200 hover:bg-white/20'}`}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-4">
      <div className="max-w-2xl mx-auto">
        <div className="text-center mb-6">
          <div className="inline-flex items-center gap-2 bg-white/10 backdrop-blur px-4 py-2 rounded-full mb-3">
            <Sparkles className="w-5 h-5 text-purple-300" />
            <span className="text-white font-medium">On-Device Summarizer</span>
          </div>
          <p className="text-purple-200 text-sm">Runs locally — no data leaves your device</p>
          {hasWebGPU === false && !forceGPU && (
            <p className="text-yellow-300/80 text-xs mt-1">
              WebGPU not available — using CPU mode.{' '}
              <button onClick={() => { setForceGPU(true); setSelectedModel('phi-3.5'); }} className="underline hover:text-yellow-200">Try GPU anyway</button>
            </p>
          )}
          {forceGPU && (
            <p className="text-orange-300/80 text-xs mt-1">
              Forcing GPU mode (experimental).{' '}
              <button onClick={() => { setForceGPU(false); setSelectedModel('qwen-0.5b-cpu'); }} className="underline hover:text-orange-200">Switch to CPU</button>
            </p>
          )}
        </div>

        {!isLocalEnv && (
          <div className="bg-yellow-500/20 border border-yellow-500/30 rounded-2xl p-4 mb-4">
            <p className="text-yellow-200 text-sm font-medium mb-2">Preview Mode</p>
            <p className="text-yellow-200/80 text-xs">This app requires Web Workers and external ML libraries that can't run in this preview. Copy the code and run it locally with the setup instructions to use it.</p>
          </div>
        )}

        {hasWebGPU !== null && modelStatus !== 'ready' && !isCloudModel && (
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 mb-4">
            {modelStatus === 'idle' && (
              <div className="text-center">
                <Cpu className="w-10 h-10 text-purple-300 mx-auto mb-3" />
                <p className="text-purple-200 text-sm mb-4">Select a model</p>

                {isMobile && (
                  <div className="bg-blue-500/20 border border-blue-500/30 rounded-xl p-3 mb-4 text-left">
                    <p className="text-blue-200 text-sm">On mobile? We recommend <strong>Groq</strong> — it's free and fast. Local models may not work reliably on mobile browsers.</p>
                  </div>
                )}

                <p className="text-purple-400 text-xs mb-2 text-left">Cloud APIs (bring your own key)</p>
                <div className="space-y-2 mb-4">
                  {cloudModels.map(m => (
                    <button
                      key={m.id}
                      onClick={() => setSelectedModel(m.id)}
                      className={`w-full flex items-center justify-between p-3 rounded-xl transition-colors ${selectedModel === m.id ? 'bg-blue-500/30 border border-blue-500' : 'bg-white/5 hover:bg-white/10 border border-transparent'}`}
                    >
                      <div className="text-left">
                        <p className={`font-medium ${selectedModel === m.id ? 'text-white' : 'text-purple-200'}`}>
                          {m.name}
                          {apiKeys[m.provider] && <span className="ml-2 text-xs text-green-400">✓ key saved</span>}
                        </p>
                        <p className="text-purple-400 text-xs">{m.desc}</p>
                      </div>
                      <Cloud className="w-4 h-4 text-purple-300" />
                    </button>
                  ))}
                </div>

                {(hasWebGPU || forceGPU) && (
                  <>
                    <p className="text-purple-400 text-xs mb-2 text-left">On-device GPU (private, requires WebGPU)</p>
                    <div className="space-y-2 mb-4">
                      {webgpuModels.map(m => (
                        <button
                          key={m.id}
                          onClick={() => setSelectedModel(m.id)}
                          className={`w-full flex items-center justify-between p-3 rounded-xl transition-colors ${selectedModel === m.id ? 'bg-purple-500/30 border border-purple-500' : 'bg-white/5 hover:bg-white/10 border border-transparent'}`}
                        >
                          <div className="text-left">
                            <p className={`font-medium ${selectedModel === m.id ? 'text-white' : 'text-purple-200'}`}>
                              {m.name}
                              {downloadedModels.includes(m.id) && <span className="ml-2 text-xs text-green-400">✓ cached</span>}
                            </p>
                            <p className="text-purple-400 text-xs">{m.desc}</p>
                          </div>
                          <span className="text-purple-300 text-sm">{m.size}</span>
                        </button>
                      ))}
                    </div>
                  </>
                )}

                <p className="text-purple-400 text-xs mb-2 text-left">On-device CPU (private, works everywhere)</p>
                <div className="space-y-2 mb-4">
                  {cpuModels.map(m => (
                    <button
                      key={m.id}
                      onClick={() => setSelectedModel(m.id)}
                      className={`w-full flex items-center justify-between p-3 rounded-xl transition-colors ${selectedModel === m.id ? 'bg-purple-500/30 border border-purple-500' : 'bg-white/5 hover:bg-white/10 border border-transparent'}`}
                    >
                      <div className="text-left">
                        <p className={`font-medium ${selectedModel === m.id ? 'text-white' : 'text-purple-200'}`}>
                          {m.name}
                          {downloadedModels.includes(m.id) && <span className="ml-2 text-xs text-green-400">✓ cached</span>}
                        </p>
                        <p className="text-purple-400 text-xs">{m.desc}</p>
                      </div>
                      <span className="text-purple-300 text-sm">{m.size}</span>
                    </button>
                  ))}
                </div>

                <button onClick={loadModel} className="bg-purple-500 hover:bg-purple-600 text-white font-medium px-6 py-3 rounded-xl transition-colors inline-flex items-center gap-2">
                  {isSelectedModelCached ? <Cpu className="w-4 h-4" /> : <Download className="w-4 h-4" />}
                  {isSelectedModelCached ? 'Load' : 'Download'} {models.find(m => m.id === selectedModel)?.name}
                </button>
              </div>
            )}
            {modelStatus === 'loading' && (
              <div className="text-center">
                <div className="w-12 h-12 border-4 border-purple-500/30 border-t-purple-500 rounded-full animate-spin mx-auto mb-3" />
                <p className="text-white font-medium mb-2">{isSelectedModelCached ? 'Loading from cache...' : 'Downloading model...'}</p>
                <div className="w-full bg-white/10 rounded-full h-2 mb-2">
                  <div className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all" style={{ width: `${downloadProgress}%` }} />
                </div>
                <p className="text-purple-300 text-sm">{downloadProgress}%</p>
              </div>
            )}
          </div>
        )}

        {isCloudModel && (
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 mb-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Cloud className="w-5 h-5 text-blue-300" />
                <span className="text-white font-medium">{selectedCloudModel?.name}</span>
              </div>
              <button
                onClick={() => setSelectedModel(cpuModels[0].id)}
                className="text-purple-300/70 text-xs hover:text-purple-300"
              >
                Change model
              </button>
            </div>
            <div className="space-y-3">
              <div>
                <label className="text-purple-200 text-xs mb-1 block flex items-center gap-1">
                  <Key className="w-3 h-3" />
                  {currentProvider === 'openai' ? 'OpenAI' : currentProvider === 'gemini' ? 'Google AI' : 'Groq'} API Key
                </label>
                <input
                  type="password"
                  value={apiKeys[currentProvider] || ''}
                  onChange={(e) => saveApiKey(currentProvider, e.target.value)}
                  placeholder={`Enter your ${currentProvider} API key`}
                  className="w-full bg-white/10 rounded-lg px-3 py-2 text-white placeholder-purple-300/50 text-sm focus:outline-none focus:ring-1 focus:ring-purple-500"
                />
              </div>
              {hasApiKey ? (
                <div className="flex items-center gap-2 text-green-300 text-sm">
                  <div className="w-2 h-2 bg-green-400 rounded-full" />
                  Ready to summarize
                </div>
              ) : (
                <p className="text-purple-300/70 text-xs">
                  {currentProvider === 'openai' && 'Get your key at platform.openai.com'}
                  {currentProvider === 'gemini' && 'Get your key at aistudio.google.com'}
                  {currentProvider === 'groq' && 'Get your key at console.groq.com'}
                </p>
              )}
            </div>
          </div>
        )}

        {modelStatus === 'ready' && !isCloudModel && (
          <div className="flex items-center justify-between bg-green-500/20 border border-green-500/30 rounded-xl px-4 py-2 mb-4">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full" />
              <span className="text-green-300 text-sm">{models.find(m => m.id === selectedModel)?.name} ready</span>
            </div>
            <button
              onClick={() => { workerRef.current?.terminate(); workerRef.current = null; setModelStatus('idle'); setSummary(''); localStorage.removeItem('modelLoaded'); shouldAutoLoad.current = false; }}
              className="text-green-300/70 text-xs hover:text-green-300"
            >
              Change model
            </button>
          </div>
        )}

        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 mb-4">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste your text here..."
            className="w-full h-40 bg-transparent text-white placeholder-purple-300/50 resize-none focus:outline-none text-base"
          />
          <div className="flex justify-between items-center text-purple-300 text-xs mt-2">
            <span>{text.length} characters</span>
            {text && <button onClick={() => setText('')} className="hover:text-white">Clear</button>}
          </div>
        </div>

        <button
          onClick={() => setShowOptions(!showOptions)}
          className="w-full flex items-center justify-between bg-white/5 hover:bg-white/10 rounded-xl px-4 py-3 mb-4 transition-colors"
        >
          <span className="text-purple-200 text-sm">Options</span>
          <div className="flex items-center gap-2">
            <span className="text-purple-300 text-xs">{detailLabels[detailLevel]} • {tone} • {focus}</span>
            {showOptions ? <ChevronUp className="w-4 h-4 text-purple-300" /> : <ChevronDown className="w-4 h-4 text-purple-300" />}
          </div>
        </button>

        {showOptions && (
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 mb-4 space-y-5">
            <div>
              <label className="text-purple-200 text-sm mb-2 block">Detail: {detailLabels[detailLevel]}</label>
              <input type="range" min="0" max="4" value={detailLevel} onChange={(e) => setDetailLevel(parseInt(e.target.value))} className="w-full accent-purple-500" />
            </div>

            <div className="flex items-center justify-between">
              <span className="text-purple-200 text-sm">Bullet points</span>
              <button onClick={() => setBulletPoints(!bulletPoints)} className={`w-12 h-6 rounded-full transition-colors ${bulletPoints ? 'bg-purple-500' : 'bg-white/20'}`}>
                <div className={`w-5 h-5 bg-white rounded-full transition-transform ${bulletPoints ? 'translate-x-6' : 'translate-x-0.5'}`} />
              </button>
            </div>

            <SelectGroup label="Tone" value={tone} onChange={setTone} options={toneOptions} />
            <SelectGroup label="Focus" value={focus} onChange={setFocus} options={focusOptions} />
            <SelectGroup label="Audience" value={audience} onChange={setAudience} options={audienceOptions} />

            <button onClick={() => setShowAdvanced(!showAdvanced)} className="flex items-center gap-2 text-purple-300 text-sm hover:text-white">
              <Settings className="w-4 h-4" />
              {showAdvanced ? 'Hide' : 'Show'} custom prompt
            </button>

            {showAdvanced && (
              <div>
                <label className="text-purple-200 text-sm mb-2 block">Custom prompt (use {'{text}'} as placeholder)</label>
                <textarea
                  value={customPrompt}
                  onChange={(e) => setCustomPrompt(e.target.value)}
                  placeholder="Leave empty to use options above. Example: Summarize this in Spanish: {text}"
                  className="w-full h-24 bg-white/10 rounded-xl p-3 text-white placeholder-purple-300/50 text-sm resize-none focus:outline-none"
                >arstarst</textarea>
              </div>
            )}
          </div>
        )}

        <button
          onClick={summarize}
          disabled={!text.trim() || loading || (isCloudModel ? !hasApiKey : modelStatus !== 'ready')}
          className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 text-white font-semibold py-4 rounded-xl transition-all flex items-center justify-center gap-2"
        >
          {loading ? (
            <div className="flex flex-col items-center gap-1">
              <div className="flex items-center gap-2">
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Generating...</span>
              </div>
              <span className="text-white/60 text-sm">{generationTime.toFixed(1)}s</span>
            </div>
          ) : (
            <>
              <FileText className="w-5 h-5" />
              Summarize
            </>
          )}
        </button>

        {error && (
          <div className="mt-4 bg-red-500/20 border border-red-500/50 rounded-xl p-4">
            <p className="text-red-200 text-sm">{error}</p>
          </div>
        )}

        {summary && (
          <div className="mt-4 bg-white/10 backdrop-blur-lg rounded-2xl p-4">
            <div className="flex justify-between items-center mb-3">
              <h3 className="text-purple-200 text-sm font-medium">Summary</h3>
              <span className="text-purple-300 text-xs">{generationTime.toFixed(1)}s</span>
            </div>
            <div className="text-white whitespace-pre-wrap text-base leading-relaxed">{summary}</div>
          </div>
        )}
      </div>
    </div>
  );
}