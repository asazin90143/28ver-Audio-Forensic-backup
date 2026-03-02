"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Download, Database, AlertCircle, CheckCircle, Brain, Cable as Cube } from "lucide-react"
import { generatePDFReport } from "@/lib/pdf-generator"
import { DatabaseService } from "@/lib/database"
import { getSupabaseStatus } from "@/lib/supabase"
import { useState, useEffect, useRef, useCallback } from "react"

interface AudioData {
  blob: Blob
  url: string
  name: string
  duration: number
  analysisResults?: any
}

interface EnhancedAudioAnalysisProps {
  audioData: AudioData | null
  visualRefs?: any // Added Refs
}

export default function EnhancedAudioAnalysis({ audioData, ...props }: EnhancedAudioAnalysisProps) {
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false)
  const [pdfProgress, setPdfProgress] = useState(0)
  const [pdfStage, setPdfStage] = useState("")
  const [isSaving, setIsSaving] = useState(false)

  const [saveStatus, setSaveStatus] = useState<string>("")
  const [dbStatus, setDbStatus] = useState<{ connected: boolean; message?: string }>({ connected: false })
  const [rotation3D, setRotation3D] = useState({ x: 0, y: 0 })
  const [zoom3D, setZoom3D] = useState(1)
  const [autoRotate3D, setAutoRotate3D] = useState(false)

  // 2D Spectrum State
  const [zoom2D, setZoom2D] = useState(1)
  const [pan2D, setPan2D] = useState(0)
  const [hoveredFreq, setHoveredFreq] = useState<{ frequency: number; magnitude: number } | null>(null)

  const spectrum3DCanvasRef = useRef<HTMLCanvasElement>(null)
  const spectrum2DCanvasRef = useRef<HTMLCanvasElement>(null)

  // Check database connection on component mount
  useEffect(() => {
    const checkDatabaseConnection = async () => {
      console.log("Audio Data received:", audioData)
      console.log("Analysis Results:", audioData?.analysisResults)
      console.log("Frequency Spectrum:", audioData?.analysisResults?.frequencySpectrum)

      const supabaseStatus = getSupabaseStatus()
      if (!supabaseStatus.connected) {
        setDbStatus({
          connected: false,
          message: "Supabase not configured",
        })
        return
      }

      try {
        const testResult = await DatabaseService.testConnection()
        setDbStatus({
          connected: testResult.success,
          message: testResult.message,
        })
      } catch (error) {
        setDbStatus({
          connected: false,
          message: "Connection test failed",
        })
      }
    }

    checkDatabaseConnection()
  }, [])

  // 3D Frequency Spectrum Visualization
  const draw3DFrequencySpectrum = useCallback(() => {
    const canvas = spectrum3DCanvasRef.current
    if (!canvas || !audioData?.analysisResults?.frequencySpectrum) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const { width, height } = canvas
    ctx.clearRect(0, 0, width, height)

    // Draw 3D background gradient
    const gradient = ctx.createRadialGradient(
      width / 2,
      height / 2,
      0,
      width / 2,
      height / 2,
      Math.max(width, height) / 2,
    )
    gradient.addColorStop(0, "#001122")
    gradient.addColorStop(1, "#000000")
    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, width, height)

    const centerX = width / 2
    const centerY = height / 2
    const spectrum = audioData.analysisResults.frequencySpectrum

    // 3D projection parameters
    const perspective = 600
    const cameraZ = 400

    // Helper function for 3D to 2D projection
    const project3D = (x: number, y: number, z: number) => {
      // Apply rotations
      const cosX = Math.cos(rotation3D.x)
      const sinX = Math.sin(rotation3D.x)
      const cosY = Math.cos(rotation3D.y)
      const sinY = Math.sin(rotation3D.y)

      // Rotate around X axis
      const y1 = y * cosX - z * sinX
      const z1 = y * sinX + z * cosX

      // Rotate around Y axis
      const x2 = x * cosY + z1 * sinY
      const z2 = -x * sinY + z1 * cosY

      // Apply perspective projection
      const scale = perspective / (perspective + z2 * zoom3D)
      return {
        x: centerX + x2 * scale,
        y: centerY + y1 * scale,
        z: z2,
        scale: scale,
      }
    }

    // Draw 3D grid
    ctx.strokeStyle = "rgba(0, 100, 200, 0.2)"
    ctx.lineWidth = 1

    // Grid lines for frequency axis
    for (let i = -5; i <= 5; i++) {
      ctx.beginPath()
      let firstPoint = true
      for (let j = -5; j <= 5; j++) {
        const x = i * 30
        const y = 0
        const z = j * 30
        const projected = project3D(x, y, z)

        if (projected.z > -perspective) {
          if (firstPoint) {
            ctx.moveTo(projected.x, projected.y)
            firstPoint = false
          } else {
            ctx.lineTo(projected.x, projected.y)
          }
        }
      }
      ctx.stroke()
    }

    // Grid lines for time axis
    for (let i = -5; i <= 5; i++) {
      ctx.beginPath()
      let firstPoint = true
      for (let j = -5; j <= 5; j++) {
        const x = j * 30
        const y = 0
        const z = i * 30
        const projected = project3D(x, y, z)

        if (projected.z > -perspective) {
          if (firstPoint) {
            ctx.moveTo(projected.x, projected.y)
            firstPoint = false
          } else {
            ctx.lineTo(projected.x, projected.y)
          }
        }
      }
      ctx.stroke()
    }

    // Draw 3D frequency spectrum bars
    const maxFreq = Math.max(...spectrum.map((s: any) => s.frequency))
    const maxMag = Math.max(...spectrum.map((s: any) => s.magnitude))

    spectrum.forEach((point: any, index: number) => {
      if (index % 3 !== 0) return // Sample every 3rd point for performance

      // Map frequency to X axis
      const x = (point.frequency / maxFreq - 0.5) * 300

      // Map magnitude to Y axis (height)
      const y = -point.magnitude * 150

      // Map time/phase to Z axis
      const z = ((point.time || index / spectrum.length) - 0.5) * 200

      // Project to 2D
      const base = project3D(x, 0, z)
      const top = project3D(x, y, z)

      if (base.z > -perspective && top.z > -perspective) {
        // Draw 3D bar
        const intensity = point.magnitude
        const hue = (point.frequency / maxFreq) * 240 // Blue to red
        const saturation = 70 + intensity * 30
        const lightness = 30 + intensity * 50

        // Draw bar with gradient
        const gradient = ctx.createLinearGradient(base.x, base.y, top.x, top.y)
        gradient.addColorStop(0, `hsla(${hue}, ${saturation}%, ${lightness * 0.6}%, 0.8)`)
        gradient.addColorStop(1, `hsla(${hue}, ${saturation}%, ${lightness}%, 0.9)`)

        ctx.strokeStyle = gradient
        ctx.lineWidth = Math.max(1, 3 * base.scale * zoom3D)
        ctx.beginPath()
        ctx.moveTo(base.x, base.y)
        ctx.lineTo(top.x, top.y)
        ctx.stroke()

        // Draw top cap
        ctx.fillStyle = `hsla(${hue}, ${saturation}%, ${lightness}%, 0.9)`
        ctx.beginPath()
        ctx.arc(top.x, top.y, Math.max(1, 2 * top.scale * zoom3D), 0, 2 * Math.PI)
        ctx.fill()

        // Add labels for prominent frequencies
        if (intensity > 0.7 && index % 10 === 0) {
          ctx.fillStyle = "white"
          ctx.font = `${Math.max(8, 10 * top.scale)}px Arial`
          ctx.shadowColor = "black"
          ctx.shadowBlur = 2
          ctx.fillText(`${point.frequency.toFixed(0)}Hz`, top.x + 5, top.y - 5)
          ctx.shadowBlur = 0
        }
      }
    })

    // Draw 3D axes with labels
    const axisLength = 150

    // X axis (Frequency) - Red
    const xAxis = project3D(axisLength, 0, 0)
    if (xAxis.z > -perspective) {
      ctx.strokeStyle = "rgba(255, 100, 100, 0.8)"
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(centerX, centerY)
      ctx.lineTo(xAxis.x, xAxis.y)
      ctx.stroke()

      ctx.fillStyle = "rgba(255, 100, 100, 0.9)"
      ctx.font = "14px Arial"
      ctx.fillText("Frequency (Hz)", xAxis.x + 10, xAxis.y)
    }

    // Y axis (Magnitude) - Green
    const yAxis = project3D(0, -axisLength, 0)
    if (yAxis.z > -perspective) {
      ctx.strokeStyle = "rgba(100, 255, 100, 0.8)"
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(centerX, centerY)
      ctx.lineTo(yAxis.x, yAxis.y)
      ctx.stroke()

      ctx.fillStyle = "rgba(100, 255, 100, 0.9)"
      ctx.font = "14px Arial"
      ctx.fillText("Magnitude", yAxis.x + 10, yAxis.y)
    }

    // Z axis (Time/Phase) - Blue
    const zAxis = project3D(0, 0, axisLength)
    if (zAxis.z > -perspective) {
      ctx.strokeStyle = "rgba(100, 100, 255, 0.8)"
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(centerX, centerY)
      ctx.lineTo(zAxis.x, zAxis.y)
      ctx.stroke()

      ctx.fillStyle = "rgba(100, 100, 255, 0.9)"
      ctx.font = "14px Arial"
      ctx.fillText("Time/Phase", zAxis.x + 10, zAxis.y)
    }

    // Draw center point
    ctx.fillStyle = "rgba(255, 255, 255, 0.8)"
    ctx.beginPath()
    ctx.arc(centerX, centerY, 4, 0, 2 * Math.PI)
    ctx.fill()

    // --- LEGEND ---
    const legendX = 10
    const legendY = 10
    const legendW = 120
    const legendH = 90

    ctx.fillStyle = "rgba(0, 0, 0, 0.6)"
    ctx.fillRect(legendX, legendY, legendW, legendH)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)"
    ctx.strokeRect(legendX, legendY, legendW, legendH)

    ctx.fillStyle = "white"
    ctx.font = "bold 11px Arial"
    ctx.textAlign = "left"
    ctx.fillText("3D Spectrum Legend", legendX + 10, legendY + 20)

    ctx.font = "10px Arial"
    // Freq Color Gradient
    const grad = ctx.createLinearGradient(legendX + 10, legendY + 30, legendX + 110, legendY + 30)
    grad.addColorStop(0, "blue")
    grad.addColorStop(1, "red")
    ctx.fillStyle = grad
    ctx.fillRect(legendX + 10, legendY + 30, 100, 10)
    ctx.fillStyle = "#cccccc"
    ctx.fillText("Low Freq → High Freq", legendX + 10, legendY + 52)

    // Height = Amplitude
    ctx.fillStyle = "#cccccc"
    ctx.fillText("Height = Amplitude", legendX + 10, legendY + 68)
    // Depth = Time/Phase
    ctx.fillText("Depth = Time/Phase", legendX + 10, legendY + 82)

    // Add rotation info
    ctx.fillStyle = "rgba(255, 255, 255, 0.6)"
    ctx.font = "10px Arial"
    ctx.fillText(
      `Rotation: X=${((rotation3D.x * 180) / Math.PI).toFixed(0)}° Y=${((rotation3D.y * 180) / Math.PI).toFixed(0)}°`,
      10,
      height - 30,
    )
    ctx.fillText(`Zoom: ${zoom3D.toFixed(1)}x`, 10, height - 15)
  }, [audioData, rotation3D, zoom3D])

  // Auto-rotation effect
  useEffect(() => {
    if (!autoRotate3D) return
    const interval = setInterval(() => {
      setRotation3D((prev) => ({ ...prev, y: prev.y + 0.005 }))
    }, 20)
    return () => clearInterval(interval)
  }, [autoRotate3D])

  // 2D Frequency Spectrum Visualization
  const draw2DSpectrum = useCallback(() => {
    const canvas = spectrum2DCanvasRef.current
    if (!canvas || !audioData?.analysisResults?.frequencySpectrum) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const { width, height } = canvas
    ctx.clearRect(0, 0, width, height)

    // Background
    ctx.fillStyle = "#f8f9fa"
    ctx.fillRect(0, 0, width, height)

    // Grid
    ctx.strokeStyle = "#e9ecef"
    ctx.lineWidth = 1
    ctx.beginPath()

    // Vertical Grid & Freq Labels
    const stepsX = 10
    for (let i = 0; i <= stepsX; i++) {
      const x = (i / stepsX) * width
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
    }

    // Horizontal Grid & Magnitude Labels
    const stepsY = 5
    for (let i = 0; i <= stepsY; i++) {
      const y = (i / stepsY) * height
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
    }
    ctx.stroke()

    // Draw Labels (Separate Pass for Clarity)
    ctx.fillStyle = "#6b7280" // Gray-500
    ctx.font = "10px Inter, sans-serif"

    // Y-Axis Labels (Magnitude %)
    ctx.textAlign = "left"
    for (let i = 0; i <= stepsY; i++) {
      const y = height - (i / stepsY) * height
      // value 0 to 100%
      const val = (i / stepsY) * 100
      if (i > 0) ctx.fillText(`${val.toFixed(0)}%`, 5, y + 3)
    }

    const spectrum = audioData.analysisResults.frequencySpectrum
    if (!spectrum || spectrum.length === 0) return

    const results = audioData.analysisResults
    const isEnhanced = results.enhanced || results.analysisType === "enhanced_classification"
    // Safety check for empty spectrum
    const lastPoint = spectrum[spectrum.length - 1]
    const maxFreq = lastPoint ? lastPoint.frequency : 20000

    // Draw Bars
    const barWidth = (width / spectrum.length) * zoom2D
    const offsetX = pan2D

    ctx.fillStyle = isEnhanced ? "rgba(59, 130, 246, 0.6)" : "rgba(168, 85, 247, 0.6)" // Blue or Purple
    ctx.strokeStyle = isEnhanced ? "rgba(59, 130, 246, 1)" : "rgba(168, 85, 247, 1)"

    spectrum.forEach((point: any, index: number) => {
      const x = index * barWidth + offsetX
      const barHeight = point.magnitude * height
      const y = height - barHeight

      // Only draw if visible
      if (x + barWidth > 0 && x < width) {
        ctx.fillRect(x, y, barWidth - 1, barHeight)
        ctx.strokeRect(x, y, barWidth - 1, barHeight)
      }
    })

    // Hover Line & Label
    if (hoveredFreq) {
      // Re-calculate x position for the hovered frequency
      const index = spectrum.findIndex((s: any) => s.frequency === hoveredFreq.frequency)
      if (index !== -1) {
        const x = index * barWidth + offsetX + barWidth / 2

        if (x > 0 && x < width) {
          ctx.strokeStyle = "red"
          ctx.lineWidth = 1
          ctx.setLineDash([5, 5])
          ctx.beginPath()
          ctx.moveTo(x, 0)
          ctx.lineTo(x, height)
          ctx.stroke()
          ctx.setLineDash([])

          // Tooltip
          const fontSize = 12
          ctx.font = `bold ${fontSize}px Inter, sans-serif`
          const text = `${hoveredFreq.frequency} Hz: ${(hoveredFreq.magnitude * 100).toFixed(1)}%`
          const textWidth = ctx.measureText(text).width
          const padding = 6

          let tooltipX = x + 10
          let tooltipY = height / 2

          // Edge cases handling
          if (tooltipX + textWidth + padding * 2 > width) {
            tooltipX = x - 10 - textWidth - padding * 2
          }

          // Tooltip box
          ctx.fillStyle = "rgba(0, 0, 0, 0.8)"
          ctx.beginPath()
          ctx.roundRect(tooltipX, tooltipY - fontSize - padding, textWidth + padding * 2, fontSize + padding * 2, 4)
          ctx.fill()

          // Tooltip text
          ctx.fillStyle = "white"
          ctx.fillText(text, tooltipX + padding, tooltipY)
        }
      }
    }

    // Draw X-Axis Labels (Frequency in Hz)
    const totalSpectrumWidth = width * zoom2D
    const hzPerPixel = maxFreq / totalSpectrumWidth

    // Draw Ticks every ~100px
    const tickSpacing = 100
    // Find the first tick position that is visible (x > 0)
    // x = virtualX + pan2D
    // We want x >= 0 => virtualX >= -pan2D

    const startPixel = -pan2D
    const startTickIndex = Math.floor(startPixel / tickSpacing)
    const startX = startTickIndex * tickSpacing

    ctx.textAlign = "center"
    ctx.textBaseline = "bottom"
    ctx.fillStyle = "#6b7280"

    // Iterate pixels in screen space logic effectively
    for (let virtualX = startX; virtualX < startPixel + width; virtualX += tickSpacing) {
      const screenX = virtualX + pan2D

      if (screenX >= -50 && screenX <= width + 50) {
        const freq = virtualX * hzPerPixel
        if (freq >= 0) {
          ctx.fillText(`${freq.toFixed(0)}Hz`, screenX, height - 2)

          ctx.strokeStyle = "#e9ecef"
          ctx.beginPath()
          ctx.moveTo(screenX, height)
          ctx.lineTo(screenX, height - 5)
          ctx.stroke()
        }
      }
    }

  }, [audioData, zoom2D, pan2D, hoveredFreq])

  // 2D Interaction Handlers
  useEffect(() => {
    const canvas = spectrum2DCanvasRef.current
    if (!canvas) return

    let isDragging = false
    let lastX = 0

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault()
      const zoomSensitivity = 0.001
      const newZoom = Math.max(1, Math.min(20, zoom2D + e.deltaY * -zoomSensitivity))

      // Try to zoom towards mouse position? Simplified: Center zoom for now or just standard zoom
      // Better: Zoom relative to left for simplicity first, or improve if user asks.
      // Let's just update scale.
      setZoom2D(newZoom)
    }

    const handleMouseDown = (e: MouseEvent) => {
      isDragging = true
      lastX = e.clientX
      canvas.style.cursor = "grabbing"
    }

    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect()
      const x = e.clientX - rect.left

      // Handle Hover
      if (audioData?.analysisResults?.frequencySpectrum) {
        const spectrum = audioData.analysisResults.frequencySpectrum
        const barWidth = (canvas.width / spectrum.length) * zoom2D
        const index = Math.floor((x - pan2D) / barWidth)

        if (index >= 0 && index < spectrum.length) {
          setHoveredFreq(spectrum[index])
        } else {
          setHoveredFreq(null)
        }
      }

      // Handle Drag
      if (isDragging) {
        const deltaX = e.clientX - lastX
        setPan2D(prev => {
          const spectrum = audioData?.analysisResults?.frequencySpectrum || []
          const totalWidth = (canvas.width / spectrum.length) * zoom2D * spectrum.length
          // Limit pan?
          // Allow some over-scroll but generally keep content in view
          return prev + deltaX
        })
        lastX = e.clientX
      }
    }

    const handleMouseUp = () => {
      isDragging = false
      canvas.style.cursor = "crosshair"
    }

    const handleMouseLeave = () => {
      isDragging = false
      setHoveredFreq(null)
      canvas.style.cursor = "default"
    }

    canvas.addEventListener("wheel", handleWheel, { passive: false })
    canvas.addEventListener("mousedown", handleMouseDown)
    canvas.addEventListener("mousemove", handleMouseMove)
    canvas.addEventListener("mouseup", handleMouseUp)
    canvas.addEventListener("mouseleave", handleMouseLeave)

    return () => {
      canvas.removeEventListener("wheel", handleWheel)
      canvas.removeEventListener("mousedown", handleMouseDown)
      canvas.removeEventListener("mousemove", handleMouseMove)
      canvas.removeEventListener("mouseup", handleMouseUp)
      canvas.removeEventListener("mouseleave", handleMouseLeave)
    }

  }, [zoom2D, pan2D, audioData])

  // Draw 2D spectrum when relevant state changes
  useEffect(() => {
    draw2DSpectrum()
  }, [draw2DSpectrum])

  // Mouse interaction for 3D spectrum
  useEffect(() => {
    const canvas = spectrum3DCanvasRef.current
    if (!canvas) return

    let isDragging = false
    let lastX = 0
    let lastY = 0

    const handleMouseDown = (e: MouseEvent) => {
      isDragging = true
      lastX = e.clientX
      lastY = e.clientY
    }

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return

      const deltaX = e.clientX - lastX
      const deltaY = e.clientY - lastY

      setRotation3D((prev) => ({
        x: prev.x + deltaY * 0.01,
        y: prev.y + deltaX * 0.01,
      }))

      lastX = e.clientX
      lastY = e.clientY
    }

    const handleMouseUp = () => {
      isDragging = false
    }

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault()
      setZoom3D((prev) => Math.max(0.3, Math.min(3, prev + e.deltaY * -0.001)))
    }

    canvas.addEventListener("mousedown", handleMouseDown)
    canvas.addEventListener("mousemove", handleMouseMove)
    canvas.addEventListener("mouseup", handleMouseUp)
    canvas.addEventListener("wheel", handleWheel)
    canvas.style.cursor = "grab"

    return () => {
      canvas.removeEventListener("mousedown", handleMouseDown)
      canvas.removeEventListener("mousemove", handleMouseMove)
      canvas.removeEventListener("mouseup", handleMouseUp)
      canvas.removeEventListener("wheel", handleWheel)
    }
  }, [])

  // Draw 3D spectrum when data changes
  useEffect(() => {
    if (audioData?.analysisResults) {
      draw3DFrequencySpectrum()

      // Animation loop for smooth rotation
      const animationId = setInterval(() => {
        draw3DFrequencySpectrum()
      }, 50)

      return () => clearInterval(animationId)
    }
  }, [draw3DFrequencySpectrum, audioData?.analysisResults])

  const handleDownloadPDF = async () => {
    if (!audioData?.analysisResults) {
      alert("No analysis results available for PDF generation")
      return
    }

    setIsGeneratingPDF(true)
    setPdfProgress(0)
    setPdfStage("Starting comprehensive PDF generation...")

    try {
      // Capture Visuals from Shared Refs
      const visuals = {
        radarCanvas: props.visualRefs?.canvas2DRef?.current?.toDataURL("image/png"),
        topographyCanvas: props.visualRefs?.canvas3DRef?.current?.toDataURL("image/png"),
        oscilloscopeCanvas: props.visualRefs?.oscRef?.current?.toDataURL("image/png"),
        spectrogramCanvas: props.visualRefs?.specRef?.current?.toDataURL("image/png"),
        // Live Visualization captures
        stftCanvas: props.visualRefs?.stftRef?.current?.toDataURL("image/png"),
        fftCanvas: props.visualRefs?.fftRef?.current?.toDataURL("image/png"),
        liveSpectrogramCanvas: props.visualRefs?.liveSpectrogramRef?.current?.toDataURL("image/png"),
        energyCanvas: props.visualRefs?.energyRef?.current?.toDataURL("image/png"),
      }

      await generatePDFReport(audioData, (progress, stage) => {
        setPdfProgress(progress)
        setPdfStage(stage)
      }, visuals)

      setSaveStatus("✅ Enhanced PDF report generated successfully!")
      setPdfStage("PDF generation completed!")
      setTimeout(() => {
        setSaveStatus("")
        setPdfStage("")
      }, 3000)
    } catch (error) {
      console.error("PDF generation error:", error)
      setSaveStatus("❌ Failed to generate PDF report")
      setPdfStage("PDF generation failed")
      setTimeout(() => {
        setSaveStatus("")
        setPdfStage("")
      }, 3000)
    } finally {
      setIsGeneratingPDF(false)
      setPdfProgress(0)
    }
  }

  const handleSaveToDatabase = async () => {
    if (!audioData?.analysisResults) {
      alert("No analysis results available to save")
      return
    }

    if (!dbStatus.connected) {
      alert("Database is not available. Please check your Supabase configuration.")
      return
    }

    setIsSaving(true)
    try {
      const analysisId = await DatabaseService.saveAnalysis(audioData)
      if (analysisId) {
        setSaveStatus("✅ Enhanced analysis saved to database successfully!")
      } else {
        setSaveStatus("❌ Failed to save analysis to database")
      }
      setTimeout(() => setSaveStatus(""), 3000)
    } catch (error: any) {
      console.error("Database save error:", error)
      setSaveStatus(`❌ Database error: ${error.message || "Unknown error"}`)
      setTimeout(() => setSaveStatus(""), 5000)
    } finally {
      setIsSaving(false)
    }
  }

  if (!audioData) {
    return (
      <div className="text-center">
        <h2 className="text-2xl font-bold text-purple-600 mb-4">Enhanced Audio Analysis</h2>
        <p className="text-gray-600 mb-8">Upload or record audio to begin advanced forensic analysis</p>
        <div className="bg-gray-100 rounded-lg p-8">
          <p className="text-gray-500">No audio data available for analysis</p>
        </div>
      </div>
    )
  }

  if (!audioData.analysisResults) {
    return (
      <div className="text-center">
        <h2 className="text-2xl font-bold text-purple-600 mb-4">Enhanced Audio Analysis</h2>
        <p className="text-gray-600 mb-8">Analyze audio to see advanced classification results</p>
        <div className="bg-gray-100 rounded-lg p-8">
          <p className="text-gray-500">Analysis not yet performed</p>
        </div>
      </div>
    )
  }

  const results = audioData.analysisResults
  const isEnhanced = results.enhanced || results.analysisType === "enhanced_classification"

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold text-purple-600">Enhanced Audio Analysis Results</h2>
          <div className="flex items-center mt-2">
            <Brain className="w-4 h-4 text-green-600 mr-2" />
            <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
              Advanced Classification System
            </Badge>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-3">
          <Button
            onClick={handleDownloadPDF}
            disabled={isGeneratingPDF}
            className="bg-red-600 hover:bg-red-700 text-white"
          >
            <Download className="w-4 h-4 mr-2" />
            {isGeneratingPDF ? "Generating..." : "Download Enhanced Report"}
          </Button>

          <Button
            onClick={handleSaveToDatabase}
            disabled={isSaving || !dbStatus.connected}
            className={`text-white ${dbStatus.connected ? "bg-green-600 hover:bg-green-700" : "bg-gray-400 cursor-not-allowed"
              }`}
            title={!dbStatus.connected ? "Database not available" : "Save to Supabase database"}
          >
            <Database className="w-4 h-4 mr-2" />
            {isSaving ? "Saving..." : "Save to Database"}
          </Button>
        </div>
      </div>

      {/* PDF Generation Progress */}
      {isGeneratingPDF && (
        <Card className="mb-4">
          <CardContent className="p-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-purple-700">Generating Comprehensive PDF Report</span>
                <span className="text-sm text-purple-600">{pdfProgress}%</span>
              </div>
              <Progress value={pdfProgress} className="w-full" />
              <p className="text-xs text-gray-600">{pdfStage}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Database Status */}
      <div className="mb-4 p-3 rounded-lg border flex items-center space-x-2">
        {dbStatus.connected ? (
          <>
            <CheckCircle className="w-4 h-4 text-green-600" />
            <span className="text-green-700 text-sm">Database Connected</span>
          </>
        ) : (
          <>
            <AlertCircle className="w-4 h-4 text-orange-600" />
            <span className="text-orange-700 text-sm">
              Database Offline - {dbStatus.message || "Connection unavailable"}
            </span>
          </>
        )}
      </div>

      {/* Status Messages */}
      {saveStatus && (
        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-blue-700 text-sm">{saveStatus}</p>
        </div>
      )}

      {/* Enhanced Classification Results */}
      {results.enhanced && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="text-lg flex items-center">
              <Brain className="w-5 h-5 mr-2 text-blue-600" />
              Advanced Classification Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Overall Statistics */}
              <div>
                <h4 className="font-semibold mb-3">Top Sound Categories</h4>
                <div className="space-y-2">
                  {results.soundEvents?.slice(0, 8).map((event: any, index: number) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <div>
                        <span className="font-medium text-sm">{event.type}</span>
                        <div className="text-xs text-gray-600">
                          {event.time}s • {event.frequency}Hz
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium">{(event.amplitude * 100).toFixed(1)}%</div>
                        <Progress value={event.amplitude * 100} className="w-16 h-2" />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Analysis Summary */}
              <div>
                <h4 className="font-semibold mb-3">Analysis Summary</h4>
                <div className="space-y-3">
                  <div className="p-3 bg-blue-50 rounded-lg">
                    <div className="text-sm font-medium text-blue-800">Classification Quality</div>
                    <div className="text-xs text-blue-600 mt-1">
                      High precision analysis with {results.detectedSounds} events detected
                    </div>
                  </div>

                  <div className="p-3 bg-green-50 rounded-lg">
                    <div className="text-sm font-medium text-green-800">Detection Confidence</div>
                    <div className="text-xs text-green-600 mt-1">Advanced algorithms with enhanced accuracy</div>
                  </div>

                  <div className="p-3 bg-purple-50 rounded-lg">
                    <div className="text-sm font-medium text-purple-800">Processing Method</div>
                    <div className="text-xs text-purple-600 mt-1">
                      Multi-layer analysis with intelligent classification
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Basic Analysis Info */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {/* File Info */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">File Information</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <p>
                <strong>File:</strong> {audioData.name}
              </p>
              <p>
                <strong>Duration:</strong> {results?.duration || 0}s
              </p>
              <p>
                <strong>Sample Rate:</strong> {results?.sampleRate || 0} Hz
              </p>
              <div className="flex items-center">
                <strong>Analysis Type:</strong>
                <Badge variant="outline" className="ml-2">
                  {isEnhanced ? "Enhanced" : "Standard"}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Sound Detection */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Sound Detection</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <p>
                <strong>Detected Events:</strong> {results?.detectedSounds || 0}
              </p>
              <p>
                <strong>Dominant Frequency:</strong> {results?.dominantFrequency || 0} Hz
              </p>
              <p>
                <strong>Max Decibels:</strong> {results?.maxDecibels || 0} dB
              </p>
              {isEnhanced && (
                <div className="flex items-center">
                  <strong>Confidence:</strong>
                  <Badge variant="outline" className="ml-2 bg-green-50 text-green-700">
                    High (Enhanced)
                  </Badge>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Primary Classification */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Primary Classification</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {results?.soundEvents && results.soundEvents.length > 0 ? (
                <>
                  <p>
                    <strong>Type:</strong> {results.soundEvents[0].type}
                  </p>
                  <p>
                    <strong>Time:</strong> {results.soundEvents[0].time}s
                  </p>
                  <p>
                    <strong>Frequency:</strong> {results.soundEvents[0].frequency} Hz
                  </p>
                  <p>
                    <strong>Amplitude:</strong> {(results.soundEvents[0].amplitude * 100).toFixed(1)}%
                  </p>
                  {isEnhanced && (
                    <div className="flex items-center">
                      <strong>Source:</strong>
                      <Badge variant="outline" className="ml-2 bg-blue-50 text-blue-700">
                        Enhanced Classification
                      </Badge>
                    </div>
                  )}
                </>
              ) : (
                <p className="text-gray-500">No classification available</p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Enhanced Sound Events Timeline */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-lg">{isEnhanced ? "Enhanced Sound Events" : "Detected Sound Events"}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {results?.soundEvents && results.soundEvents.length > 0 ? (
              results.soundEvents.map((event: any, index: number) => (
                <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-4">
                    <Badge variant="outline" className={isEnhanced ? "bg-blue-50 text-blue-700" : ""}>
                      {event.type}
                    </Badge>
                    <span className="font-mono text-sm">{event.time}s</span>
                    {isEnhanced && event.confidence && (
                      <Badge variant="outline" className="bg-green-50 text-green-700 text-xs">
                        {(event.confidence * 100).toFixed(0)}% conf.
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center space-x-4">
                    <span className="text-sm">{event.frequency} Hz</span>
                    <Progress value={event.amplitude * 100} className="w-20" />
                    <span className="text-sm font-medium">{(event.amplitude * 100).toFixed(1)}%</span>
                    <span className="text-xs text-gray-600">{event.decibels} dB</span>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-gray-500 text-center">No sound events to display</p>
            )}
          </div>
        </CardContent>
      </Card>

      {/* 3D Frequency Spectrum Analysis */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-lg flex items-center">
            <Cube className="w-5 h-5 mr-2 text-purple-600" />
            3D Frequency Spectrum Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <canvas
              ref={spectrum3DCanvasRef}
              width={800}
              height={500}
              className="w-full border rounded-lg bg-gray-900"
            />
            <div className="flex justify-between items-center">
              <p className="text-sm text-gray-600">Interactive 3D visualization: Drag to rotate, scroll to zoom</p>
              <div className="flex space-x-4">
                <Button onClick={() => setRotation3D({ x: 0, y: 0 })} variant="outline" size="sm">
                  Reset View
                </Button>
                <Button onClick={() => setZoom3D(1)} variant="outline" size="sm">
                  Reset Zoom
                </Button>
                <Button
                  onClick={() => setAutoRotate3D(!autoRotate3D)}
                  variant={autoRotate3D ? "default" : "outline"}
                  size="sm"
                >
                  {autoRotate3D ? "Stop Rotation" : "Auto Rotate"}
                </Button>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="p-3 bg-red-50 rounded-lg">
                <h4 className="font-semibold text-red-800 mb-2">X-Axis: Frequency</h4>
                <p className="text-red-700">Represents frequency components from low to high Hz</p>
              </div>
              <div className="p-3 bg-green-50 rounded-lg">
                <h4 className="font-semibold text-green-800 mb-2">Y-Axis: Magnitude</h4>
                <p className="text-green-700">Shows the amplitude/strength of each frequency</p>
              </div>
              <div className="p-3 bg-blue-50 rounded-lg">
                <h4 className="font-semibold text-blue-800 mb-2">Z-Axis: Time/Phase</h4>
                <p className="text-blue-700">Temporal distribution and phase information</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Traditional Frequency Spectrum */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Traditional Frequency Spectrum Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative w-full h-64 bg-gray-100 rounded-lg overflow-hidden border border-gray-200">
            <canvas
              ref={spectrum2DCanvasRef}
              width={800}
              height={300}
              className="w-full h-full cursor-crosshair"
            />
            <div className="absolute top-2 right-2 bg-white/80 p-2 rounded text-xs text-gray-600 shadow-sm pointer-events-none">
              Scroll to Zoom • Drag to Pan
            </div>
          </div>
          <div className="flex justify-between items-center mt-2">
            <p className="text-sm text-gray-600">
              Frequency range: 0 - {results?.frequencySpectrum?.[results.frequencySpectrum.length - 1]?.frequency || 0} Hz
              {isEnhanced && <span className="text-blue-600 ml-2">• Enhanced Analysis</span>}
            </p>
            <div className="space-x-2">
              <Button size="sm" variant="outline" onClick={() => { setZoom2D(1); setPan2D(0); }}>Reset View</Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Technical Information */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle className="text-lg">3D Analysis Technical Specifications</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="p-3 bg-purple-50 rounded-lg">
              <h4 className="font-semibold text-purple-800 mb-2">3D Visualization Features</h4>
              <ul className="text-purple-700 space-y-1">
                <li>• Real-time 3D frequency spectrum rendering</li>
                <li>• Interactive rotation and zoom controls</li>
                <li>• Multi-dimensional data mapping</li>
                <li>• Perspective projection with depth</li>
                <li>• Color-coded frequency visualization</li>
                <li>• Integrated with sonar and live analysis</li>
              </ul>
            </div>

            <div className="p-3 bg-blue-50 rounded-lg">
              <h4 className="font-semibold text-blue-800 mb-2">Analysis Integration</h4>
              <ul className="text-blue-700 space-y-1">
                <li>• Connected to sonar view system</li>
                <li>• Synchronized with live visualization</li>
                <li>• Enhanced classification integration</li>
                <li>• Real-time data processing</li>
                <li>• Cross-platform compatibility</li>
                <li>• Professional forensic standards</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
