import React, { useState, useEffect } from 'react'
import {
  Layers,
  Cpu,
  GitFork,
  Sparkles,
  Upload,
  Image as ImageIcon,
  Zap,
  Activity,
  Download,
  Camera,
  RefreshCw,
  CheckCircle2,
  Sliders
} from 'lucide-react'

const API_BASE = 'http://127.0.0.1:8000'

export default function App() {
  const [activeTab, setActiveTab] = useState('task1') // 'task1' | 'task2' | 'task3' | 'task4'
  const [systemHealth, setSystemHealth] = useState(null)

  // Task 1 States
  const [sampleList, setSampleList] = useState([])
  const [selectedSample, setSelectedSample] = useState(null)
  const [uploadedFile, setUploadedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [corruption, setCorruption] = useState('salt_and_pepper')
  const [severity, setSeverity] = useState('medium')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  // Check backend health & fetch sample images
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((res) => res.json())
      .then((data) => setSystemHealth(data))
      .catch(() => setSystemHealth({ status: 'offline' }))

    fetch(`${API_BASE}/api/sample-images`)
      .then((res) => res.json())
      .then((data) => {
        if (data.samples && data.samples.length > 0) {
          setSampleList(data.samples)
          setSelectedSample(data.samples[0])
        }
      })
      .catch(() => {})
  }, [])

  // Update preview when sample selection changes
  useEffect(() => {
    if (selectedSample && !uploadedFile) {
      fetch(`${API_BASE}/api/sample-images/${selectedSample}`)
        .then((res) => res.json())
        .then((data) => setPreviewUrl(data.image_base64))
        .catch(() => {})
    }
  }, [selectedSample, uploadedFile])

  const handleFileUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      setUploadedFile(file)
      setSelectedSample(null)
      const reader = new FileReader()
      reader.onload = () => setPreviewUrl(reader.result)
      reader.readAsDataURL(file)
    }
  }

  const runTask1Restoration = async () => {
    setLoading(true)
    setResult(null)
    const formData = new FormData()

    if (uploadedFile) {
      formData.append('file', uploadedFile)
    } else if (selectedSample) {
      formData.append('sample_name', selectedSample)
    }

    formData.append('corruption', corruption)
    formData.append('severity', severity)

    try {
      const res = await fetch(`${API_BASE}/api/universal-restoration`, {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (res.ok) {
        setResult(data)
      } else {
        alert(data.detail || 'Inference failed')
      }
    } catch (err) {
      alert('Could not connect to FastAPI backend: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Header */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur px-6 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center text-indigo-400 font-bold">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-white tracking-tight flex items-center gap-2">
              OmniRestorer <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 font-normal border border-slate-700">IEEE Paper Suite</span>
            </h1>
            <p className="text-xs text-slate-400">Generative AI Image Restoration & Synthesis</p>
          </div>
        </div>

        {/* Backend Status Pill */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800">
            <span className={`w-2 h-2 rounded-full ${systemHealth?.status === 'healthy' ? 'bg-emerald-400' : 'bg-rose-400'}`} />
            <span className="text-slate-300">
              {systemHealth?.status === 'healthy' ? 'ONNX Backend Active' : 'Backend Offline'}
            </span>
          </div>
        </div>
      </header>

      {/* Workspace Tabs */}
      <div className="border-b border-slate-800/80 bg-slate-900/30 px-6">
        <div className="flex items-center gap-1 max-w-6xl mx-auto overflow-x-auto">
          <button
            onClick={() => setActiveTab('task1')}
            className={`px-4 py-3 text-sm font-medium border-b-2 flex items-center gap-2 transition-colors ${
              activeTab === 'task1'
                ? 'border-indigo-500 text-indigo-400 bg-indigo-500/5'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-4 h-4" />
            1. Universal Autoencoder
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Ready</span>
          </button>

          <button
            onClick={() => setActiveTab('task2')}
            className={`px-4 py-3 text-sm font-medium border-b-2 flex items-center gap-2 transition-colors ${
              activeTab === 'task2'
                ? 'border-indigo-500 text-indigo-400 bg-indigo-500/5'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <GitFork className="w-4 h-4" />
            2. Hard-Routed Specialists
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">Task 2</span>
          </button>

          <button
            onClick={() => setActiveTab('task3')}
            className={`px-4 py-3 text-sm font-medium border-b-2 flex items-center gap-2 transition-colors ${
              activeTab === 'task3'
                ? 'border-indigo-500 text-indigo-400 bg-indigo-500/5'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Activity className="w-4 h-4" />
            3. Soft Mixture-of-Experts
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">Task 3</span>
          </button>

          <button
            onClick={() => setActiveTab('task4')}
            className={`px-4 py-3 text-sm font-medium border-b-2 flex items-center gap-2 transition-colors ${
              activeTab === 'task4'
                ? 'border-indigo-500 text-indigo-400 bg-indigo-500/5'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Sparkles className="w-4 h-4" />
            4. Face-to-Sketch GAN
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">Task 4</span>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-6">
        {activeTab === 'task1' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Controls Panel */}
            <div className="lg:col-span-4 space-y-5">
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">Input Source</h2>
                  {previewUrl && (
                    <span className="text-xs text-emerald-400 flex items-center gap-1">
                      <CheckCircle2 className="w-3.5 h-3.5" /> 128x128 Ready
                    </span>
                  )}
                </div>

                {/* Upload or Sample Selector */}
                <div className="space-y-3">
                  <label className="block w-full cursor-pointer border border-dashed border-slate-700 hover:border-slate-500 rounded-lg p-4 text-center bg-slate-950/40 transition">
                    <input type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
                    <Upload className="w-5 h-5 mx-auto mb-1 text-slate-400" />
                    <span className="text-xs font-medium text-slate-300">Upload any custom photo</span>
                    <span className="block text-[11px] text-slate-500 mt-0.5">PNG, JPG, WebP</span>
                  </label>

                  {sampleList.length > 0 && (
                    <div>
                      <span className="text-xs font-medium text-slate-400 mb-2 block">Or pick dataset sample:</span>
                      <div className="grid grid-cols-4 gap-1.5 max-h-32 overflow-y-auto pr-1">
                        {sampleList.map((sample) => (
                          <button
                            key={sample}
                            onClick={() => {
                              setSelectedSample(sample)
                              setUploadedFile(null)
                            }}
                            className={`p-1.5 text-[11px] truncate rounded border text-left transition ${
                              selectedSample === sample
                                ? 'border-indigo-500 bg-indigo-500/10 text-indigo-300 font-semibold'
                                : 'border-slate-800 bg-slate-950/30 text-slate-400 hover:bg-slate-800/50'
                            }`}
                          >
                            {sample.split('_')[0]}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Corruption Controls */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">Corruption Controls</h2>

                <div className="space-y-3">
                  <div>
                    <label className="text-xs text-slate-400 block mb-1.5">Corruption Type</label>
                    <div className="grid grid-cols-2 gap-2">
                      {[
                        { id: 'clean', label: 'Clean (None)' },
                        { id: 'salt_and_pepper', label: 'Salt & Pepper' },
                        { id: 'gaussian_blur', label: 'Gaussian Blur' },
                        { id: 'rectangular_occlusion', label: 'Occlusion' },
                      ].map((c) => (
                        <button
                          key={c.id}
                          onClick={() => setCorruption(c.id)}
                          className={`px-3 py-2 text-xs rounded-lg border text-left transition ${
                            corruption === c.id
                              ? 'border-indigo-500 bg-indigo-500/15 text-indigo-300 font-medium'
                              : 'border-slate-800 bg-slate-950/40 text-slate-400 hover:border-slate-700'
                          }`}
                        >
                          {c.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {corruption !== 'clean' && (
                    <div>
                      <label className="text-xs text-slate-400 block mb-1.5">Severity Tier</label>
                      <div className="grid grid-cols-3 gap-2">
                        {['low', 'medium', 'high'].map((sev) => (
                          <button
                            key={sev}
                            onClick={() => setSeverity(sev)}
                            className={`py-1.5 text-xs capitalize rounded border text-center transition ${
                              severity === sev
                                ? 'border-indigo-500 bg-indigo-500/15 text-indigo-300 font-medium'
                                : 'border-slate-800 bg-slate-950/40 text-slate-400 hover:border-slate-700'
                            }`}
                          >
                            {sev}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <button
                  onClick={runTask1Restoration}
                  disabled={loading || (!selectedSample && !uploadedFile)}
                  className="w-full mt-2 py-2.5 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm flex items-center justify-center gap-2 transition"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Running ONNX Inference...
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4" />
                      Restore Image
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Right Display Panel */}
            <div className="lg:col-span-8 space-y-5">
              {/* Image Inspection Cards */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-semibold text-slate-200">Reconstruction Inspection (128x128)</h3>
                  {result && (
                    <span className="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700 font-mono">
                      Latency: <span className="text-emerald-400 font-bold">{result.latency_ms} ms</span>
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Panel 1: Input */}
                  <div className="border border-slate-800 bg-slate-950/60 rounded-lg p-3 text-center">
                    <span className="text-xs text-slate-400 mb-2 block font-medium">Input (Damaged)</span>
                    <div className="aspect-square bg-slate-900 rounded border border-slate-800 flex items-center justify-center overflow-hidden">
                      {result?.input_image ? (
                        <img src={result.input_image} alt="Input" className="w-full h-full object-contain" />
                      ) : previewUrl ? (
                        <img src={previewUrl} alt="Preview" className="w-full h-full object-contain" />
                      ) : (
                        <ImageIcon className="w-8 h-8 text-slate-700" />
                      )}
                    </div>
                  </div>

                  {/* Panel 2: Restored */}
                  <div className="border border-slate-800 bg-slate-950/60 rounded-lg p-3 text-center">
                    <span className="text-xs text-slate-400 mb-2 block font-medium">Restored Output (ONNX)</span>
                    <div className="aspect-square bg-slate-900 rounded border border-slate-800 flex items-center justify-center overflow-hidden">
                      {result?.restored_image ? (
                        <img src={result.restored_image} alt="Restored" className="w-full h-full object-contain" />
                      ) : (
                        <span className="text-xs text-slate-600">Awaiting inference...</span>
                      )}
                    </div>
                  </div>

                  {/* Panel 3: Error Map */}
                  <div className="border border-slate-800 bg-slate-950/60 rounded-lg p-3 text-center">
                    <span className="text-xs text-slate-400 mb-2 block font-medium">Absolute Error Map</span>
                    <div className="aspect-square bg-slate-900 rounded border border-slate-800 flex items-center justify-center overflow-hidden">
                      {result?.error_map_image ? (
                        <img src={result.error_map_image} alt="Error Map" className="w-full h-full object-contain" />
                      ) : (
                        <span className="text-xs text-slate-600">Heatmap generated on run</span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Quantitative Metrics Bar */}
                {result?.metrics && Object.keys(result.metrics).length > 0 && (
                  <div className="mt-4 pt-4 border-t border-slate-800 grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                    <div className="bg-slate-950/50 p-2.5 rounded border border-slate-800">
                      <span className="text-[11px] text-slate-400 block">L1 Pixel Error</span>
                      <span className="text-sm font-bold text-slate-200">{result.metrics.l1_error}</span>
                    </div>
                    <div className="bg-slate-950/50 p-2.5 rounded border border-slate-800">
                      <span className="text-[11px] text-slate-400 block">PSNR Metric</span>
                      <span className="text-sm font-bold text-slate-200">{result.metrics.psnr_db} dB</span>
                    </div>
                    <div className="bg-slate-950/50 p-2.5 rounded border border-slate-800">
                      <span className="text-[11px] text-slate-400 block">Corruption</span>
                      <span className="text-sm font-medium text-slate-300 capitalize">{result.corruption_applied?.type}</span>
                    </div>
                    <div className="bg-slate-950/50 p-2.5 rounded border border-slate-800">
                      <span className="text-[11px] text-slate-400 block">Runtime Engine</span>
                      <span className="text-sm font-medium text-indigo-400">ONNX C++</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Task 2 Workspace */}
        {activeTab === 'task2' && (
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-8 text-center max-w-2xl mx-auto space-y-4">
            <GitFork className="w-12 h-12 text-indigo-400 mx-auto" />
            <h2 className="text-lg font-bold text-white">Task 2: Hard-Routed Specialist Autoencoders</h2>
            <p className="text-sm text-slate-400 leading-relaxed">
              This workspace will host your 4-class corruption classifier and the 3 specialist autoencoders. Once trained, it will display class probabilities, routing decision, and expert output.
            </p>
            <div className="p-4 bg-slate-950/60 border border-slate-800 rounded-lg text-left text-xs space-y-2 text-slate-400">
              <span className="font-semibold text-slate-300 block">Ready to Connect:</span>
              <p>• Endpoint: <code className="text-indigo-400">POST /api/hard-routing</code></p>
              <p>• Mode: Oracle Routing vs Predicted Routing comparison</p>
              <p>• Outputs: Normalized 4-class probabilities + selected expert</p>
            </div>
          </div>
        )}

        {/* Task 3 Workspace */}
        {activeTab === 'task3' && (
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-8 text-center max-w-2xl mx-auto space-y-4">
            <Activity className="w-12 h-12 text-indigo-400 mx-auto" />
            <h2 className="text-lg font-bold text-white">Task 3: Jointly Trained Soft Mixture-of-Experts</h2>
            <p className="text-sm text-slate-400 leading-relaxed">
              This workspace will host your differentiable soft gating network. It will display the continuous weights [w0, w1, w2, w3] and show which specialists contributed to the final reconstruction.
            </p>
            <div className="p-4 bg-slate-950/60 border border-slate-800 rounded-lg text-left text-xs space-y-2 text-slate-400">
              <span className="font-semibold text-slate-300 block">Ready to Connect:</span>
              <p>• Endpoint: <code className="text-indigo-400">POST /api/soft-mixture</code></p>
              <p>• Live Temperature Slider for routing sharpness</p>
              <p>• Dynamic weighting bar charts for Clean, Noise, Blur, Occlusion</p>
            </div>
          </div>
        )}

        {/* Task 4 Workspace */}
        {activeTab === 'task4' && (
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-8 text-center max-w-2xl mx-auto space-y-4">
            <Sparkles className="w-12 h-12 text-indigo-400 mx-auto" />
            <h2 className="text-lg font-bold text-white">Task 4: Style-Conditioned Face-to-Sketch (cGAN)</h2>
            <p className="text-sm text-slate-400 leading-relaxed">
              This workspace will allow evaluators to upload a face or take a picture with their webcam, choose Style 1, 2, or 3, and generate a sketch in real time.
            </p>
            <div className="p-4 bg-slate-950/60 border border-slate-800 rounded-lg text-left text-xs space-y-2 text-slate-400">
              <span className="font-semibold text-slate-300 block">Ready to Connect:</span>
              <p>• Endpoint: <code className="text-indigo-400">POST /api/face-to-sketch</code></p>
              <p>• Webcam live capture + image upload</p>
              <p>• Style selection buttons: Style 1, Style 2, Style 3</p>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
