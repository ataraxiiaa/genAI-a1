import React, { useState, useEffect, useRef } from 'react'
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
  Sliders,
  Palette,
  Eye,
  Video,
  VideoOff,
  Smile
} from 'lucide-react'

const API_BASE = 'http://127.0.0.1:8000'

export default function App() {
  const [activeTab, setActiveTab] = useState('task4') // Start directly on Task 4 Face-to-Sketch!
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

  // Task 2 States
  const [task2RoutingMode, setTask2RoutingMode] = useState('predicted')
  const [task2OracleLabel, setTask2OracleLabel] = useState(1)
  const [task2Loading, setTask2Loading] = useState(false)
  const [task2Result, setTask2Result] = useState(null)

  // Task 3 States (Soft MoE)
  const [softMoeTemp, setSoftMoeTemp] = useState(1.0)
  const [task3Loading, setTask3Loading] = useState(false)
  const [task3Result, setTask3Result] = useState(null)

  // Task 4 States (Face-to-Sketch cGAN)
  const [faceSamples, setFaceSamples] = useState([])
  const [selectedFaceSample, setSelectedFaceSample] = useState(null)
  const [faceUploadedFile, setFaceUploadedFile] = useState(null)
  const [faceWebcamData, setFaceWebcamData] = useState(null)
  const [facePreviewUrl, setFacePreviewUrl] = useState(null)
  const [selectedStyle, setSelectedStyle] = useState(0) // 0: Style 1, 1: Style 2, 2: Style 3
  const [task4Loading, setTask4Loading] = useState(false)
  const [task4Result, setTask4Result] = useState(null)
  const [webcamActive, setWebcamActive] = useState(false)
  const [webcamStream, setWebcamStream] = useState(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  // Check backend health, fetch pet samples, and fetch FS2K face samples
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

    fetch(`${API_BASE}/api/face-samples`)
      .then((res) => res.json())
      .then((data) => {
        if (data.samples && data.samples.length > 0) {
          setFaceSamples(data.samples)
          setSelectedFaceSample(data.samples[0])
        }
      })
      .catch(() => {})
  }, [])

  // Update pet preview when sample selection changes
  useEffect(() => {
    if (selectedSample && !uploadedFile) {
      fetch(`${API_BASE}/api/sample-images/${selectedSample}`)
        .then((res) => res.json())
        .then((data) => setPreviewUrl(data.image_base64))
        .catch(() => {})
    }
  }, [selectedSample, uploadedFile])

  // Update face preview when face sample selection changes
  useEffect(() => {
    if (selectedFaceSample && !faceUploadedFile && !faceWebcamData) {
      fetch(`${API_BASE}/api/face-samples/${selectedFaceSample.subfolder}/${selectedFaceSample.filename}`)
        .then((res) => res.json())
        .then((data) => setFacePreviewUrl(data.image_base64))
        .catch(() => {})
    }
  }, [selectedFaceSample, faceUploadedFile, faceWebcamData])

  // Webcam Management
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 400 }, height: { ideal: 400 }, facingMode: 'user' }
      })
      setWebcamStream(stream)
      setWebcamActive(true)
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }
    } catch (err) {
      alert('Could not access webcam: ' + err.message)
    }
  }

  const stopWebcam = () => {
    if (webcamStream) {
      webcamStream.getTracks().forEach((track) => track.stop())
      setWebcamStream(null)
    }
    setWebcamActive(false)
  }

  const captureWebcamSnapshot = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current
      const canvas = canvasRef.current
      canvas.width = 128
      canvas.height = 128
      const ctx = canvas.getContext('2d')
      ctx.drawImage(video, 0, 0, 128, 128)
      const dataUrl = canvas.toDataURL('image/jpeg', 0.95)
      setFacePreviewUrl(dataUrl)
      setFaceWebcamData(dataUrl)
      setFaceUploadedFile(null)
      setSelectedFaceSample(null)
      stopWebcam()
    }
  }

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

  const handleFaceFileUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      setFaceUploadedFile(file)
      setSelectedFaceSample(null)
      setFaceWebcamData(null)
      const reader = new FileReader()
      reader.onload = () => setFacePreviewUrl(reader.result)
      reader.readAsDataURL(file)
    }
  }

  const downloadImage = (dataUrl, filename) => {
    const link = document.createElement('a')
    link.href = dataUrl
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
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

  const runTask2Restoration = async () => {
    setTask2Loading(true)
    setTask2Result(null)
    const formData = new FormData()

    if (uploadedFile) {
      formData.append('file', uploadedFile)
    } else if (selectedSample) {
      formData.append('sample_name', selectedSample)
    }

    formData.append('corruption', corruption)
    formData.append('severity', severity)
    formData.append('mode', task2RoutingMode)
    if (task2RoutingMode === 'oracle') {
      formData.append('oracle_label', task2OracleLabel)
    }

    try {
      const res = await fetch(`${API_BASE}/api/hard-routing`, {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (res.ok) {
        setTask2Result(data)
      } else {
        alert(data.detail || 'Task 2 Hard-Routing failed')
      }
    } catch (err) {
      alert('Could not connect to FastAPI backend: ' + err.message)
    } finally {
      setTask2Loading(false)
    }
  }

  const runTask3Restoration = async () => {
    setTask3Loading(true)
    setTask3Result(null)
    const formData = new FormData()

    if (uploadedFile) {
      formData.append('file', uploadedFile)
    } else if (selectedSample) {
      formData.append('sample_name', selectedSample)
    }

    formData.append('corruption', corruption)
    formData.append('severity', severity)
    formData.append('temperature', softMoeTemp)

    try {
      const res = await fetch(`${API_BASE}/api/soft-mixture`, {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (res.ok) {
        setTask3Result(data)
      } else {
        alert(data.detail || 'Task 3 Soft MoE failed')
      }
    } catch (err) {
      alert('Could not connect to FastAPI backend: ' + err.message)
    } finally {
      setTask3Loading(false)
    }
  }

  const runTask4Generation = async (overrideStyle = null) => {
    const styleToUse = overrideStyle !== null ? overrideStyle : selectedStyle
    setTask4Loading(true)
    const formData = new FormData()

    if (faceUploadedFile) {
      formData.append('file', faceUploadedFile)
    } else if (faceWebcamData) {
      formData.append('webcam_image', faceWebcamData)
    } else if (selectedFaceSample) {
      formData.append('sample_sub', selectedFaceSample.subfolder)
      formData.append('sample_name', selectedFaceSample.filename)
    }

    formData.append('style', `Style ${styleToUse + 1}`)

    try {
      const res = await fetch(`${API_BASE}/api/face-to-sketch`, {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (res.ok) {
        setTask4Result(data)
      } else {
        alert(data.detail || 'Face-to-Sketch generation failed')
      }
    } catch (err) {
      alert('Could not connect to FastAPI backend: ' + err.message)
    } finally {
      setTask4Loading(false)
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

        {/* Task 2 Workspace: Hard-Routed Specialist Autoencoders */}
        {activeTab === 'task2' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Left Controls Column */}
            <div className="lg:col-span-4 space-y-6">
              {/* Image Input Selection */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
                  Select Evaluation Pet
                </span>
                <div className="grid grid-cols-4 gap-2">
                  {sampleList.slice(0, 8).map((sample) => (
                    <button
                      key={sample}
                      type="button"
                      onClick={() => {
                        setSelectedSample(sample)
                        setUploadedFile(null)
                        setTask2Result(null)
                      }}
                      className={`relative aspect-square rounded-lg border overflow-hidden p-0.5 transition-all ${
                        selectedSample === sample && !uploadedFile
                          ? 'border-indigo-500 ring-2 ring-indigo-500/30'
                          : 'border-slate-800 hover:border-slate-700 opacity-70 hover:opacity-100'
                      }`}
                    >
                      <img
                        src={`${API_BASE}/api/sample-images/${sample}`}
                        alt={sample}
                        className="w-full h-full object-cover rounded"
                        loading="lazy"
                      />
                    </button>
                  ))}
                </div>

                <div className="relative flex items-center justify-center my-2">
                  <div className="border-t border-slate-800 w-full" />
                  <span className="bg-slate-900 px-2 text-[10px] text-slate-400 uppercase font-mono absolute">or upload</span>
                </div>

                <label className="flex flex-col items-center justify-center border border-dashed border-slate-800 hover:border-slate-700 rounded-lg p-3 cursor-pointer bg-slate-950/40 transition-colors">
                  <Upload className="w-4 h-4 text-slate-400 mb-1" />
                  <span className="text-xs text-slate-300">Upload Custom Pet Image</span>
                  <span className="text-[10px] text-slate-400">JPG, PNG (Auto-resized to 128x128)</span>
                  <input type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
                </label>
              </div>

              {/* Corruption Injector */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
                  Simulate Degradation
                </span>

                <div className="space-y-1.5">
                  <label className="text-xs text-slate-400">Corruption Type:</label>
                  <select
                    value={corruption}
                    onChange={(e) => setCorruption(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                  >
                    <option value="none">Clean (Identity Bypass Test)</option>
                    <option value="salt_and_pepper">Salt-and-Pepper Noise</option>
                    <option value="gaussian_blur">Gaussian Blur</option>
                    <option value="rectangular_occlusion">Rectangular Box Occlusion</option>
                  </select>
                </div>

                {corruption !== 'none' && (
                  <div className="space-y-1.5">
                    <label className="text-xs text-slate-400">Severity Level:</label>
                    <div className="grid grid-cols-3 gap-2">
                      {['low', 'medium', 'high'].map((sev) => (
                        <button
                          key={sev}
                          type="button"
                          onClick={() => setSeverity(sev)}
                          className={`py-1.5 px-3 rounded-lg text-xs font-medium border capitalize transition-all ${
                            severity === sev
                              ? 'bg-indigo-600/20 border-indigo-500 text-indigo-300 font-semibold'
                              : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          {sev}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Hard-Routing Policy Controls */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
                  Routing Policy Mode
                </span>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setTask2RoutingMode('predicted')}
                    className={`px-3 py-2 rounded-lg text-xs font-medium border text-center transition-all ${
                      task2RoutingMode === 'predicted'
                        ? 'bg-indigo-600/20 border-indigo-500 text-indigo-300 font-semibold'
                        : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Classifier Auto-Route
                  </button>
                  <button
                    type="button"
                    onClick={() => setTask2RoutingMode('oracle')}
                    className={`px-3 py-2 rounded-lg text-xs font-medium border text-center transition-all ${
                      task2RoutingMode === 'oracle'
                        ? 'bg-amber-600/20 border-amber-500 text-amber-300 font-semibold'
                        : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Oracle Ground-Truth
                  </button>
                </div>

                {task2RoutingMode === 'oracle' && (
                  <div className="space-y-1.5 pt-2">
                    <label className="text-[11px] text-slate-400 block font-medium">Forced Oracle Target Specialist:</label>
                    <div className="grid grid-cols-2 gap-1.5 text-xs">
                      {[
                        { id: 0, label: 'Clean Bypass' },
                        { id: 1, label: 'Noise Specialist' },
                        { id: 2, label: 'Blur Specialist' },
                        { id: 3, label: 'Occlusion Specialist' },
                      ].map((item) => (
                        <button
                          key={item.id}
                          type="button"
                          onClick={() => setTask2OracleLabel(item.id)}
                          className={`px-2 py-1.5 rounded text-[11px] border text-left truncate ${
                            task2OracleLabel === item.id
                              ? 'bg-amber-500/20 border-amber-500/60 text-amber-200 font-semibold'
                              : 'bg-slate-950/40 border-slate-800 text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          {item.label}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Action Button */}
              <button
                type="button"
                onClick={runTask2Restoration}
                disabled={task2Loading || (!selectedSample && !uploadedFile)}
                className="w-full py-3 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:hover:bg-indigo-600 text-white font-semibold text-sm transition-all flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/20"
              >
                {task2Loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Routing to Specialist...</span>
                  </>
                ) : (
                  <>
                    <GitFork className="w-4 h-4" />
                    <span>Execute Hard-Routing Restoration</span>
                  </>
                )}
              </button>
            </div>

            {/* Right Display Column */}
            <div className="lg:col-span-8 space-y-6">
              {/* Classification Decision Banner & Probabilities */}
              {task2Result && (
                <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-400 font-medium">Selected Expert:</span>
                      <span className="text-xs px-2.5 py-1 rounded-md bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 font-semibold">
                        {task2Result.selected_expert}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-400 font-medium">Mode:</span>
                      <span className={`text-[11px] px-2 py-0.5 rounded font-mono uppercase ${
                        task2Result.mode === 'oracle'
                          ? 'bg-amber-500/10 border border-amber-500/30 text-amber-300'
                          : 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-300'
                      }`}>
                        {task2Result.mode}
                      </span>
                    </div>
                  </div>

                  {/* 4-Class Probability Distribution Bars */}
                  {task2Result.probabilities && (
                    <div className="space-y-2 pt-1">
                      <span className="text-[11px] text-slate-400 uppercase tracking-wider block font-medium">
                        Classifier Confidence Distribution
                      </span>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                        {[
                          { key: 'clean', label: 'Clean' },
                          { key: 'salt_and_pepper', label: 'Salt & Pepper' },
                          { key: 'gaussian_blur', label: 'Gaussian Blur' },
                          { key: 'rectangular_occlusion', label: 'Box Occlusion' },
                        ].map(({ key, label }) => {
                          const prob = task2Result.probabilities[key] || 0
                          const isWinner = task2Result.predicted_corruption === key
                          return (
                            <div key={key} className={`p-2 rounded-lg border ${isWinner ? 'bg-slate-950/80 border-indigo-500/40' : 'bg-slate-950/30 border-slate-800/60'}`}>
                              <div className="flex justify-between items-center mb-1">
                                <span className={`text-[11px] ${isWinner ? 'text-indigo-300 font-semibold' : 'text-slate-400'}`}>
                                  {label}
                                </span>
                                <span className="font-mono text-[11px] text-slate-300 font-medium">
                                  {(prob * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                                <div
                                  className={`h-full rounded-full transition-all duration-500 ${isWinner ? 'bg-indigo-500' : 'bg-slate-700'}`}
                                  style={{ width: `${Math.max(2, prob * 100)}%` }}
                                />
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* 3-Panel Inspection View */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Restoration Inspection Matrix
                  </span>
                  {task2Result && (
                    <span className="text-xs text-slate-400 flex items-center gap-1 font-mono">
                      <Zap className="w-3.5 h-3.5 text-amber-400" />
                      Latency: <strong className="text-slate-200">{task2Result.latency_ms} ms</strong>
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Panel 1: Input */}
                  <div className="border border-slate-800 bg-slate-950/60 rounded-lg p-3 text-center">
                    <span className="text-xs text-slate-400 mb-2 block font-medium">Degraded Input</span>
                    <div className="aspect-square bg-slate-900 rounded border border-slate-800 flex items-center justify-center overflow-hidden">
                      {task2Result?.input_image ? (
                        <img src={task2Result.input_image} alt="Input" className="w-full h-full object-contain" />
                      ) : previewUrl ? (
                        <img src={previewUrl} alt="Preview" className="w-full h-full object-contain opacity-60" />
                      ) : (
                        <ImageIcon className="w-8 h-8 text-slate-700" />
                      )}
                    </div>
                  </div>

                  {/* Panel 2: Specialist Restored */}
                  <div className="border border-indigo-500/30 bg-slate-950/60 rounded-lg p-3 text-center ring-1 ring-indigo-500/20">
                    <span className="text-xs text-indigo-400 mb-2 block font-medium">Specialist Restored</span>
                    <div className="aspect-square bg-slate-900 rounded border border-slate-800 flex items-center justify-center overflow-hidden">
                      {task2Result?.restored_image ? (
                        <img src={task2Result.restored_image} alt="Restored" className="w-full h-full object-contain" />
                      ) : (
                        <span className="text-xs text-slate-600">Restoration pending run</span>
                      )}
                    </div>
                  </div>

                  {/* Panel 3: Error Map */}
                  <div className="border border-slate-800 bg-slate-950/60 rounded-lg p-3 text-center">
                    <span className="text-xs text-slate-400 mb-2 block font-medium">Absolute Error Map</span>
                    <div className="aspect-square bg-slate-900 rounded border border-slate-800 flex items-center justify-center overflow-hidden">
                      {task2Result?.error_map_image ? (
                        <img src={task2Result.error_map_image} alt="Error Map" className="w-full h-full object-contain" />
                      ) : (
                        <span className="text-xs text-slate-600">Heatmap generated on run</span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Quantitative Metrics Bar */}
                {task2Result?.metrics && Object.keys(task2Result.metrics).length > 0 && (
                  <div className="mt-4 pt-4 border-t border-slate-800 grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                    <div className="bg-slate-950/50 p-2.5 rounded border border-slate-800">
                      <span className="text-[11px] text-slate-400 block">L1 Pixel Error</span>
                      <span className="text-sm font-bold text-slate-200">{task2Result.metrics.l1_error}</span>
                    </div>
                    <div className="bg-slate-950/50 p-2.5 rounded border border-slate-800">
                      <span className="text-[11px] text-slate-400 block">PSNR Metric</span>
                      <span className="text-sm font-bold text-slate-200">{task2Result.metrics.psnr_db} dB</span>
                    </div>
                    <div className="bg-slate-950/50 p-2.5 rounded border border-slate-800">
                      <span className="text-[11px] text-slate-400 block">Corruption</span>
                      <span className="text-sm font-medium text-slate-300 capitalize">{task2Result.corruption_applied?.type}</span>
                    </div>
                    <div className="bg-slate-950/50 p-2.5 rounded border border-slate-800">
                      <span className="text-[11px] text-slate-400 block">Runtime Engine</span>
                      <span className="text-sm font-medium text-indigo-400">ONNX C++ (Dual)</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}


        {/* Task 3 Workspace: Soft Mixture-of-Experts */}
        {activeTab === 'task3' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Left Controls */}
            <div className="lg:col-span-4 space-y-6">
              {/* Pet Selector */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
                  Select Evaluation Pet
                </span>
                <div className="grid grid-cols-4 gap-2">
                  {sampleList.slice(0, 8).map((sample) => (
                    <button
                      key={sample}
                      type="button"
                      onClick={() => {
                        setSelectedSample(sample)
                        setUploadedFile(null)
                        setTask3Result(null)
                      }}
                      className={`relative aspect-square rounded-lg border overflow-hidden p-0.5 transition-all ${
                        selectedSample === sample && !uploadedFile
                          ? 'border-indigo-500 ring-2 ring-indigo-500/30'
                          : 'border-slate-800 hover:border-slate-700 opacity-70 hover:opacity-100'
                      }`}
                    >
                      <img
                        src={`${API_BASE}/api/sample-images/${sample}`}
                        alt={sample}
                        className="w-full h-full object-cover rounded"
                        loading="lazy"
                      />
                    </button>
                  ))}
                </div>

                <div className="relative flex items-center justify-center my-2">
                  <div className="border-t border-slate-800 w-full" />
                  <span className="bg-slate-900 px-2 text-[10px] text-slate-400 uppercase font-mono absolute">or upload</span>
                </div>

                <label className="flex flex-col items-center justify-center border border-dashed border-slate-800 hover:border-slate-700 rounded-lg p-3 cursor-pointer bg-slate-950/40 transition-colors">
                  <Upload className="w-4 h-4 text-slate-400 mb-1" />
                  <span className="text-xs text-slate-300">Upload Custom Image</span>
                  <input type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
                </label>
              </div>

              {/* Degradation Controls */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
                  Simulate Degradation
                </span>
                <select
                  value={corruption}
                  onChange={(e) => setCorruption(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  <option value="none">Clean (Identity Bypass Test)</option>
                  <option value="salt_and_pepper">Salt-and-Pepper Noise</option>
                  <option value="gaussian_blur">Gaussian Blur</option>
                  <option value="rectangular_occlusion">Rectangular Box Occlusion</option>
                </select>

                {corruption !== 'none' && (
                  <div className="grid grid-cols-3 gap-2">
                    {['low', 'medium', 'high'].map((sev) => (
                      <button
                        key={sev}
                        type="button"
                        onClick={() => setSeverity(sev)}
                        className={`py-1.5 px-3 rounded-lg text-xs font-medium border capitalize transition-all ${
                          severity === sev
                            ? 'bg-indigo-600/20 border-indigo-500 text-indigo-300 font-semibold'
                            : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {sev}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Action Button */}
              <button
                type="button"
                onClick={runTask3Restoration}
                disabled={task3Loading || (!selectedSample && !uploadedFile)}
                className="w-full py-3 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white font-semibold text-sm transition-all flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/20"
              >
                {task3Loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Blending Specialists...</span>
                  </>
                ) : (
                  <>
                    <Activity className="w-4 h-4" />
                    <span>Execute Soft MoE Restoration</span>
                  </>
                )}
              </button>
            </div>

            {/* Right Display */}
            <div className="lg:col-span-8 space-y-6">
              {/* Continuous Gating Distribution Card */}
              {task3Result && (
                <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-400 font-medium">Dominant Specialist:</span>
                      <span className="text-xs px-2.5 py-1 rounded-md bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 font-semibold">
                        {task3Result.dominant_expert}
                      </span>
                    </div>
                    <span className="text-xs text-slate-400 font-mono">
                      Latency: <strong className="text-emerald-400">{task3Result.latency_ms} ms</strong>
                    </span>
                  </div>

                  {/* 4 Soft Weights */}
                  {task3Result.routing_weights && (
                    <div className="space-y-2">
                      <span className="text-[11px] text-slate-400 uppercase tracking-wider block font-medium">
                        Continuous Routing Gating Weights (Softmax Blending)
                      </span>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                        {[
                          { key: 'clean_identity', label: 'Clean Bypass' },
                          { key: 'salt_and_pepper_expert', label: 'Noise Specialist' },
                          { key: 'gaussian_blur_expert', label: 'Blur Specialist' },
                          { key: 'occlusion_expert', label: 'Occlusion Specialist' },
                        ].map(({ key, label }) => {
                          const weight = task3Result.routing_weights[key] || 0
                          const isWinner = task3Result.dominant_expert_key === key
                          return (
                            <div key={key} className={`p-2.5 rounded-lg border ${isWinner ? 'bg-slate-950/80 border-indigo-500/50' : 'bg-slate-950/30 border-slate-800/60'}`}>
                              <div className="flex justify-between items-center mb-1">
                                <span className={`text-[11px] ${isWinner ? 'text-indigo-300 font-semibold' : 'text-slate-400'}`}>
                                  {label}
                                </span>
                                <span className="font-mono text-[11px] text-slate-300 font-medium">
                                  {(weight * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                                <div
                                  className={`h-full rounded-full transition-all duration-500 ${isWinner ? 'bg-indigo-500' : 'bg-slate-700'}`}
                                  style={{ width: `${Math.max(2, weight * 100)}%` }}
                                />
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* 3-Panel Inspection Matrix */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block border-b border-slate-800 pb-3">
                  Soft MoE Restoration Inspection
                </span>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="border border-slate-800 bg-slate-950/60 rounded-lg p-3 text-center">
                    <span className="text-xs text-slate-400 mb-2 block font-medium">Input Image</span>
                    <div className="aspect-square bg-slate-900 rounded border border-slate-800 flex items-center justify-center overflow-hidden">
                      {task3Result?.input_image ? (
                        <img src={task3Result.input_image} alt="Input" className="w-full h-full object-contain" />
                      ) : previewUrl ? (
                        <img src={previewUrl} alt="Preview" className="w-full h-full object-contain opacity-60" />
                      ) : (
                        <ImageIcon className="w-8 h-8 text-slate-700" />
                      )}
                    </div>
                  </div>

                  <div className="border border-indigo-500/30 bg-slate-950/60 rounded-lg p-3 text-center ring-1 ring-indigo-500/20">
                    <span className="text-xs text-indigo-400 mb-2 block font-medium">Soft-Blended Output</span>
                    <div className="aspect-square bg-slate-900 rounded border border-slate-800 flex items-center justify-center overflow-hidden">
                      {task3Result?.restored_image ? (
                        <img src={task3Result.restored_image} alt="Restored" className="w-full h-full object-contain" />
                      ) : (
                        <span className="text-xs text-slate-600">Awaiting inference...</span>
                      )}
                    </div>
                  </div>

                  <div className="border border-slate-800 bg-slate-950/60 rounded-lg p-3 text-center">
                    <span className="text-xs text-slate-400 mb-2 block font-medium">Absolute Error Map</span>
                    <div className="aspect-square bg-slate-900 rounded border border-slate-800 flex items-center justify-center overflow-hidden">
                      {task3Result?.error_map_image ? (
                        <img src={task3Result.error_map_image} alt="Error Map" className="w-full h-full object-contain" />
                      ) : (
                        <span className="text-xs text-slate-600">Heatmap generated on run</span>
                      )}
                    </div>
                  </div>
                </div>

                {task3Result?.metrics && Object.keys(task3Result.metrics).length > 0 && (
                  <div className="mt-4 pt-4 border-t border-slate-800 grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                    <div className="bg-slate-950/50 p-2.5 rounded border border-slate-800">
                      <span className="text-[11px] text-slate-400 block">L1 Pixel Error</span>
                      <span className="text-sm font-bold text-slate-200">{task3Result.metrics.l1_error}</span>
                    </div>
                    <div className="bg-slate-950/50 p-2.5 rounded border border-slate-800">
                      <span className="text-[11px] text-slate-400 block">PSNR Metric</span>
                      <span className="text-sm font-bold text-slate-200">{task3Result.metrics.psnr_db} dB</span>
                    </div>
                    <div className="bg-slate-950/50 p-2.5 rounded border border-slate-800">
                      <span className="text-[11px] text-slate-400 block">Corruption</span>
                      <span className="text-sm font-medium text-slate-300 capitalize">{task3Result.corruption_applied?.type}</span>
                    </div>
                    <div className="bg-slate-950/50 p-2.5 rounded border border-slate-800">
                      <span className="text-[11px] text-slate-400 block">Runtime Engine</span>
                      <span className="text-sm font-medium text-indigo-400">ONNX C++ (Soft MoE)</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Task 4 Workspace: Style-Conditioned Face-to-Sketch GAN */}
        {activeTab === 'task4' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Hidden canvas for webcam snapshot */}
            <canvas ref={canvasRef} className="hidden" />

            {/* Left Controls Column */}
            <div className="lg:col-span-5 space-y-6">
              {/* Portrait Input Card */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
                    1. Input Facial Portrait
                  </span>
                  <span className="text-[11px] text-indigo-400 font-mono">128x128 Target</span>
                </div>

                {/* Preset FS2K Face Samples */}
                {faceSamples.length > 0 && (
                  <div>
                    <label className="text-[11px] text-slate-400 mb-2 block font-medium">
                      Select FS2K Dataset Sample:
                    </label>
                    <div className="grid grid-cols-4 sm:grid-cols-6 gap-2">
                      {faceSamples.map((sample) => {
                        const isSelected = selectedFaceSample?.filename === sample.filename && !faceUploadedFile && !faceWebcamData
                        return (
                          <button
                            key={sample.filename}
                            type="button"
                            onClick={() => {
                              setSelectedFaceSample(sample)
                              setFaceUploadedFile(null)
                              setFaceWebcamData(null)
                              stopWebcam()
                            }}
                            className={`relative aspect-square rounded-lg border overflow-hidden p-0.5 transition-all ${
                              isSelected
                                ? 'border-indigo-500 ring-2 ring-indigo-500/40 scale-105 z-10'
                                : 'border-slate-800 hover:border-slate-700 opacity-70 hover:opacity-100'
                            }`}
                            title={sample.label}
                          >
                            <img
                              src={`${API_BASE}/api/face-samples/${sample.subfolder}/${sample.filename}`}
                              alt={sample.label}
                              className="w-full h-full object-cover rounded"
                              loading="lazy"
                            />
                          </button>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* Upload Custom & Webcam Options */}
                <div className="grid grid-cols-2 gap-2 pt-1">
                  <label className="flex flex-col items-center justify-center border border-dashed border-slate-800 hover:border-slate-700 rounded-lg p-3 cursor-pointer bg-slate-950/40 transition">
                    <Upload className="w-4 h-4 text-slate-400 mb-1" />
                    <span className="text-xs text-slate-300 font-medium">Upload Photo</span>
                    <span className="text-[10px] text-slate-500">JPG, PNG</span>
                    <input type="file" accept="image/*" onChange={handleFaceFileUpload} className="hidden" />
                  </label>

                  <button
                    type="button"
                    onClick={() => {
                      if (webcamActive) {
                        stopWebcam()
                      } else {
                        startWebcam()
                      }
                    }}
                    className={`flex flex-col items-center justify-center border rounded-lg p-3 transition ${
                      webcamActive
                        ? 'border-rose-500/60 bg-rose-500/10 text-rose-300'
                        : 'border-slate-800 hover:border-slate-700 bg-slate-950/40 text-slate-300'
                    }`}
                  >
                    {webcamActive ? (
                      <>
                        <VideoOff className="w-4 h-4 text-rose-400 mb-1" />
                        <span className="text-xs font-medium">Close Camera</span>
                        <span className="text-[10px] text-rose-400/80">Active</span>
                      </>
                    ) : (
                      <>
                        <Camera className="w-4 h-4 text-indigo-400 mb-1" />
                        <span className="text-xs font-medium">Live Webcam</span>
                        <span className="text-[10px] text-slate-500">Take Snapshot</span>
                      </>
                    )}
                  </button>
                </div>

                {/* Webcam Stream Viewfinder Modal / Inline */}
                {webcamActive && (
                  <div className="p-3 bg-slate-950 rounded-lg border border-indigo-500/30 space-y-3">
                    <div className="relative aspect-square max-w-[240px] mx-auto rounded overflow-hidden bg-black border border-slate-800">
                      <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover mirror" />
                      <div className="absolute inset-0 border-2 border-indigo-500/40 rounded pointer-events-none" />
                    </div>
                    <button
                      type="button"
                      onClick={captureWebcamSnapshot}
                      className="w-full py-2 px-3 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs flex items-center justify-center gap-1.5 transition"
                    >
                      <Camera className="w-3.5 h-3.5" />
                      Capture Photo Snapshot
                    </button>
                  </div>
                )}
              </div>

              {/* Style Category Selector Card */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
                    2. Select Sketch Style Condition
                  </span>
                  <span className="text-[11px] text-slate-500 font-mono">FS2K Classes</span>
                </div>

                <div className="space-y-2.5">
                  {[
                    {
                      idx: 0,
                      title: "Style 1: Shaded Pencil",
                      badge: "Fine Graphite",
                      desc: "Soft tonal pencil gradients, smooth skin shading, and lifelike graphite depth.",
                      color: "from-blue-500/20 to-indigo-500/10 border-blue-500/40 text-blue-300",
                      tagBg: "bg-blue-500/20 text-blue-300 border-blue-500/30",
                    },
                    {
                      idx: 1,
                      title: "Style 2: Clean Contour",
                      badge: "Minimalist Line",
                      desc: "Clean structural outlines, delicate boundary linework, and minimal interior hatching.",
                      color: "from-emerald-500/20 to-teal-500/10 border-emerald-500/40 text-emerald-300",
                      tagBg: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
                    },
                    {
                      idx: 2,
                      title: "Style 3: Graphic Comic",
                      badge: "Expressive Ink",
                      desc: "Bold high-contrast ink strokes, expressive anime/comic contours, and stylized contours.",
                      color: "from-amber-500/20 to-orange-500/10 border-amber-500/40 text-amber-300",
                      tagBg: "bg-amber-500/20 text-amber-300 border-amber-500/30",
                    },
                  ].map((s) => {
                    const isSelected = selectedStyle === s.idx
                    return (
                      <button
                        key={s.idx}
                        type="button"
                        onClick={() => setSelectedStyle(s.idx)}
                        className={`w-full p-3 rounded-lg border text-left transition-all ${
                          isSelected
                            ? `bg-gradient-to-r ${s.color} border-indigo-500 shadow-md ring-1 ring-indigo-500/40`
                            : 'bg-slate-950/40 border-slate-800 hover:border-slate-700 text-slate-400'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className={`text-xs font-bold ${isSelected ? 'text-white' : 'text-slate-300'}`}>
                            {s.title}
                          </span>
                          <span className={`text-[10px] px-2 py-0.5 rounded font-mono border ${isSelected ? s.tagBg : 'bg-slate-900 text-slate-500 border-slate-800'}`}>
                            {s.badge}
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-400 leading-relaxed">
                          {s.desc}
                        </p>
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Action Button */}
              <button
                type="button"
                onClick={() => runTask4Generation()}
                disabled={task4Loading || (!selectedFaceSample && !faceUploadedFile && !faceWebcamData)}
                className="w-full py-3.5 px-4 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-40 disabled:hover:from-indigo-600 disabled:hover:to-purple-600 text-white font-semibold text-sm transition-all flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/25"
              >
                {task4Loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Synthesizing Sketch with cGAN...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 text-amber-300" />
                    <span>Generate Face Sketch (Style {selectedStyle + 1})</span>
                  </>
                )}
              </button>
            </div>

            {/* Right Display Column */}
            <div className="lg:col-span-7 space-y-6">
              {/* Main Synthesis Result Canvas */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                      Face-to-Sketch Synthesis Matrix
                    </span>
                    {task4Result && (
                      <span className="text-[11px] px-2 py-0.5 rounded-full bg-purple-500/15 border border-purple-500/30 text-purple-300 font-semibold">
                        {task4Result.style_badge}
                      </span>
                    )}
                  </div>
                  {task4Result && (
                    <span className="text-xs text-slate-400 flex items-center gap-1 font-mono">
                      <Zap className="w-3.5 h-3.5 text-amber-400" />
                      Latency: <strong className="text-emerald-400">{task4Result.latency_ms} ms</strong>
                    </span>
                  )}
                </div>

                {/* Side-by-Side Dual Display */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* Left: Input Photograph */}
                  <div className="border border-slate-800 bg-slate-950/60 rounded-xl p-4 text-center space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-400 font-medium">Input Photograph (x)</span>
                      <span className="text-[10px] text-slate-500 font-mono">128x128 RGB</span>
                    </div>
                    <div className="aspect-square bg-slate-900 rounded-lg border border-slate-800 flex items-center justify-center overflow-hidden relative group">
                      {task4Result?.input_photo ? (
                        <img src={task4Result.input_photo} alt="Photo" className="w-full h-full object-contain" />
                      ) : facePreviewUrl ? (
                        <img src={facePreviewUrl} alt="Preview" className="w-full h-full object-contain opacity-80" />
                      ) : (
                        <div className="text-center p-4">
                          <ImageIcon className="w-10 h-10 text-slate-700 mx-auto mb-1" />
                          <span className="text-xs text-slate-600 block">Select or upload a portrait</span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Right: Synthesized Sketch */}
                  <div className="border border-indigo-500/40 bg-slate-950/60 rounded-xl p-4 text-center space-y-2 ring-1 ring-indigo-500/20">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-indigo-300 font-semibold flex items-center gap-1.5">
                        <Sparkles className="w-3.5 h-3.5 text-amber-300" />
                        Generated Sketch G(x, s)
                      </span>
                      {task4Result && (
                        <button
                          type="button"
                          onClick={() => downloadImage(task4Result.sketch_image, `sketch_style_${task4Result.style_idx + 1}.jpg`)}
                          className="text-[11px] text-indigo-400 hover:text-indigo-200 flex items-center gap-1 transition"
                          title="Download Sketch"
                        >
                          <Download className="w-3 h-3" />
                          Save
                        </button>
                      )}
                    </div>
                    <div className="aspect-square bg-slate-900 rounded-lg border border-slate-800 flex items-center justify-center overflow-hidden">
                      {task4Result?.sketch_image ? (
                        <img src={task4Result.sketch_image} alt="Generated Sketch" className="w-full h-full object-contain" />
                      ) : (
                        <div className="text-center p-4">
                          <Sparkles className="w-10 h-10 text-slate-700 mx-auto mb-1" />
                          <span className="text-xs text-slate-600 block">Awaiting cGAN generation...</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Instant Multi-Style Switcher Toolbar */}
                {task4Result && (
                  <div className="mt-4 pt-3 border-t border-slate-800 flex flex-wrap items-center justify-between gap-2">
                    <span className="text-xs text-slate-400 font-medium">Re-render in another style:</span>
                    <div className="flex items-center gap-1.5">
                      {[0, 1, 2].map((s_idx) => (
                        <button
                          key={s_idx}
                          type="button"
                          onClick={() => {
                            setSelectedStyle(s_idx)
                            runTask4Generation(s_idx)
                          }}
                          disabled={task4Loading}
                          className={`px-2.5 py-1 text-xs rounded border transition ${
                            task4Result.style_idx === s_idx
                              ? 'bg-indigo-600/30 border-indigo-500 text-indigo-200 font-bold'
                              : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          Style {s_idx + 1}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Architecture & Verification Specs */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
                  Model Specifications & System Pipeline
                </span>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                  <div className="bg-slate-950/50 p-2.5 rounded border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Architecture</span>
                    <span className="text-xs font-semibold text-slate-200">U-Net (Pix2Pix)</span>
                  </div>
                  <div className="bg-slate-950/50 p-2.5 rounded border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Conditioning</span>
                    <span className="text-xs font-semibold text-indigo-300">Spatial Embedding</span>
                  </div>
                  <div className="bg-slate-950/50 p-2.5 rounded border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Discriminator</span>
                    <span className="text-xs font-semibold text-slate-200">70x70 PatchGAN</span>
                  </div>
                  <div className="bg-slate-950/50 p-2.5 rounded border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Runtime Engine</span>
                    <span className="text-xs font-semibold text-emerald-400">ONNX C++</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
