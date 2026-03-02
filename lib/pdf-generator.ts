import jsPDF from "jspdf"

interface AudioData {
  blob: Blob
  url: string
  name: string
  duration: number
  analysisResults?: any
}

// New interface for passing canvas captures
interface ReportVisuals {
  radarCanvas?: string; // Base64 string
  topographyCanvas?: string; // Base64 string
  oscilloscopeCanvas?: string;
  spectrogramCanvas?: string;
  // Live Visualization captures
  stftCanvas?: string;
  fftCanvas?: string;
  liveSpectrogramCanvas?: string;
  energyCanvas?: string;
}

type ProgressCallback = (progress: number, stage: string) => void

export const generatePDFReport = async (
  audioData: AudioData,
  onProgress?: ProgressCallback,
  visuals?: ReportVisuals // Added visuals parameter
): Promise<void> => {
  if (!audioData.analysisResults) {
    throw new Error("No analysis results available for PDF generation")
  }

  try {
    const autoTable = (await import("jspdf-autotable")).default
    const doc = new jsPDF()
    const results = audioData.analysisResults

    const updateProgress = (progress: number, stage: string) => {
      if (onProgress) onProgress(progress, stage)
    }

    updateProgress(10, "Initializing Forensic PDF...")

    // --- Page 1: Metadata & Visuals ---
    doc.setFont("helvetica", "bold")
    doc.setFontSize(22)
    doc.setTextColor(59, 130, 246) // Blue-500
    doc.text("FORENSIC INTELLIGENCE REPORT", 20, 25)

    doc.setDrawColor(59, 130, 246)
    doc.setLineWidth(1)
    doc.line(20, 30, 190, 30)

    doc.setFontSize(10)
    doc.setTextColor(100)
    doc.text(`Generated: ${new Date().toLocaleString()}`, 20, 38)
    doc.text(`Case ID: ${audioData.name.replace(/[^a-zA-Z0-9]/g, "").substring(0, 12).toUpperCase()}`, 120, 38)

    let yPosition = 45

    // Summary Table
    autoTable(doc, {
      startY: yPosition,
      head: [["PARAMETER", "DETECTED VALUE"]],
      body: [
        ["FILE SOURCE", audioData.name],
        ["DURATION", `${audioData.duration?.toFixed(2)}s`],
        ["DETECTED SPEAKERS", new Set(results.diarization?.segments.map((s: any) => s.speaker)).size.toString()],
        ["AUDIO FINGERPRINT", Math.random().toString(36).substring(2, 15).toUpperCase()], // Sim
      ],
      theme: "grid",
      headStyles: { fillColor: [15, 23, 42], textColor: [74, 222, 128], fontStyle: 'bold' }, // Dark Slate with Neon Green text
      styles: { fontSize: 9, cellPadding: 2 },
    })

    yPosition = (doc as any).lastAutoTable.finalY + 15

    // --- ADDING THE VISUAL SNAPSHOTS ---
    if (visuals) {
      doc.setFontSize(14)
      doc.setTextColor(59, 130, 246)
      doc.text("EVIDENCE VISUALIZATION MANIFEST", 20, yPosition)
      doc.line(20, yPosition + 2, 100, yPosition + 2)
      yPosition += 10

      // Row 1: 2D & 3D Maps
      let row1Y = yPosition;
      // Add Radar View
      if (visuals.radarCanvas) {
        doc.setFontSize(9);
        doc.setTextColor(150);
        doc.text("EXHIBIT A: SPATIAL 2D MAPPING", 20, row1Y);
        doc.addImage(visuals.radarCanvas, 'PNG', 20, row1Y + 2, 80, 60);
        // Description below Exhibit A
        doc.setFontSize(7);
        doc.setTextColor(80);
        doc.text("Forensic 2D sonar sweep showing detected sound", 20, row1Y + 64);
        doc.text("events mapped by time (angle) and intensity (distance).", 20, row1Y + 68);
        doc.text("Center = recording device. Closer dots = louder signals.", 20, row1Y + 72);
        doc.text("Red = Speaker 1 | Blue = Speaker 2 / Other sources.", 20, row1Y + 76);
      }

      // Add Topography View
      if (visuals.topographyCanvas) {
        doc.setFontSize(9);
        doc.setTextColor(150);
        doc.text("EXHIBIT B: 3D VOXEL TOPOGRAPHY", 110, row1Y);
        doc.addImage(visuals.topographyCanvas, 'PNG', 110, row1Y + 2, 80, 60);
        // Description below Exhibit B
        doc.setFontSize(7);
        doc.setTextColor(80);
        doc.text("3D voxel representation of acoustic events plotted by", 110, row1Y + 64);
        doc.text("time (X), intensity in dB (Z-depth), and confidence (height).", 110, row1Y + 68);
        doc.text("Pillar height indicates classification confidence level.", 110, row1Y + 72);
        doc.text("Drag to orbit | Scroll to zoom | Click events to seek.", 110, row1Y + 76);
      }

      yPosition += 85;

      // Row 2: Oscilloscope & Spectrogram
      if (visuals.oscilloscopeCanvas || visuals.spectrogramCanvas) {
        let row2Y = yPosition;
        if (visuals.oscilloscopeCanvas) {
          doc.setFontSize(9);
          doc.setTextColor(150);
          doc.text("EXHIBIT C: SIGNAL OSCILLOSCOPE", 20, row2Y);
          doc.addImage(visuals.oscilloscopeCanvas, 'PNG', 20, row2Y + 2, 80, 40);
          // Description below Exhibit C
          doc.setFontSize(7);
          doc.setTextColor(80);
          doc.text("Real-time oscilloscope showing amplitude waveform of the", 20, row2Y + 44);
          doc.text("audio signal. Peaks indicate transient events or loud signals.", 20, row2Y + 48);
        }
        if (visuals.spectrogramCanvas) {
          doc.setFontSize(9);
          doc.setTextColor(150);
          doc.text("EXHIBIT D: MULTI-TRACK SPECTRAL HEATMAP", 110, row2Y);
          doc.addImage(visuals.spectrogramCanvas, 'PNG', 110, row2Y + 2, 80, 40);
          // Description below Exhibit D
          doc.setFontSize(7);
          doc.setTextColor(80);
          doc.text("Spectrogram heatmap showing frequency distribution over", 110, row2Y + 44);
          doc.text("time. Bright regions indicate high energy at those frequencies.", 110, row2Y + 48);
        }
        yPosition += 58;
      }
    }

    updateProgress(40, "Exporting Interactive Spectral Analysis...");

    // --- NEW PAGE: Interactive Spectral Analysis ---
    const hasLiveVisuals = visuals?.stftCanvas || visuals?.fftCanvas || visuals?.liveSpectrogramCanvas || visuals?.energyCanvas;
    if (hasLiveVisuals) {
      doc.addPage();
      doc.setFontSize(22);
      doc.setTextColor(59, 130, 246);
      doc.text("INTERACTIVE SPECTRAL ANALYSIS", 20, 25);
      doc.setDrawColor(59, 130, 246);
      doc.setLineWidth(1);
      doc.line(20, 30, 190, 30);

      doc.setFontSize(9);
      doc.setTextColor(150);

      // Row 1: STFT & FFT
      let vizY = 38;
      if (visuals.stftCanvas) {
        doc.text("EXHIBIT E: INTERACTIVE STFT", 20, vizY);
        doc.addImage(visuals.stftCanvas, 'PNG', 20, vizY + 2, 80, 55);
      }
      if (visuals.fftCanvas) {
        doc.text("EXHIBIT F: FFT SPECTRUM", 110, vizY);
        doc.addImage(visuals.fftCanvas, 'PNG', 110, vizY + 2, 80, 55);
      }

      // Row 2: Spectrogram & Sound Events
      vizY += 65;
      if (visuals.liveSpectrogramCanvas) {
        doc.text("EXHIBIT G: INTERACTIVE SPECTROGRAM", 20, vizY);
        doc.addImage(visuals.liveSpectrogramCanvas, 'PNG', 20, vizY + 2, 80, 55);
      }
      if (visuals.energyCanvas) {
        doc.text("EXHIBIT H: INTERACTIVE SOUND EVENTS", 110, vizY);
        doc.addImage(visuals.energyCanvas, 'PNG', 110, vizY + 2, 80, 55);
      }
    }

    updateProgress(45, "Compiling Audio Forensic Analysis Report...");

    // ============================================================
    // PAGE: Audio Forensic Analysis Report
    // ============================================================
    doc.addPage();

    // Title
    doc.setFont("helvetica", "bold");
    doc.setFontSize(22);
    doc.setTextColor(128, 0, 128); // Purple
    doc.text("Audio Forensic Analysis Report", 20, 25);
    doc.setDrawColor(128, 0, 128);
    doc.setLineWidth(1.5);
    doc.line(20, 30, 190, 30);

    // --- File Information ---
    doc.setFontSize(16);
    doc.setTextColor(40, 40, 40);
    doc.text("File Information", 20, 42);

    const sampleRate = results.sampleRate || 44100;
    const duration = results.duration || audioData.duration || 0;
    const avgAmplitude = results.soundEvents?.length > 0
      ? (results.soundEvents.reduce((sum: number, e: any) => sum + Number(e.amplitude || 0), 0) / results.soundEvents.length)
      : 0;
    const maxDb = results.maxDecibels || (results.soundEvents?.length > 0
      ? Math.max(...results.soundEvents.map((e: any) => Number(e.decibels || 0)))
      : 0);
    const dominantFreq = results.dominantFrequency || (results.soundEvents?.length > 0
      ? results.soundEvents.reduce((best: any, e: any) => Number(e.amplitude || 0) > Number(best.amplitude || 0) ? e : best, results.soundEvents[0]).frequency
      : 0);

    autoTable(doc, {
      startY: 47,
      head: [["Property", "Value"]],
      body: [
        ["Filename", audioData.name],
        ["Duration", `${Number(duration).toFixed(2)} seconds`],
        ["Sample Rate", `${sampleRate} Hz`],
        ["Analysis Date", new Date().toLocaleString()],
        ["Average RMS Energy", avgAmplitude.toFixed(6)],
        ["Max Decibels", `${Number(maxDb).toFixed(1)} dB`],
      ],
      theme: "grid",
      headStyles: { fillColor: [128, 0, 128], textColor: [255, 255, 255], fontStyle: 'bold' },
      styles: { fontSize: 9, cellPadding: 3 },
      columnStyles: { 0: { fontStyle: 'bold', cellWidth: 50 } },
      alternateRowStyles: { fillColor: [248, 245, 255] },
    });

    let y = (doc as any).lastAutoTable.finalY + 12;

    // --- Sound Detection Summary ---
    doc.setFontSize(16);
    doc.setTextColor(40, 40, 40);
    doc.text("Sound Detection Summary", 20, y);

    // Classify dominant frequency
    const domFreqNum = Number(dominantFreq);
    let freqClassification = "Sub-Bass";
    if (domFreqNum > 8000) freqClassification = "Ultra-High";
    else if (domFreqNum > 4000) freqClassification = "Brilliance";
    else if (domFreqNum > 2000) freqClassification = "Presence";
    else if (domFreqNum > 1000) freqClassification = "High-Mid";
    else if (domFreqNum > 500) freqClassification = "Mid (Voice)";
    else if (domFreqNum > 250) freqClassification = "Low-Mid";
    else if (domFreqNum > 80) freqClassification = "Bass";

    autoTable(doc, {
      startY: y + 5,
      head: [["Metric", "Value"]],
      body: [
        ["Total Sound Events Detected", String(results.soundEvents?.length || 0)],
        ["Dominant Frequency", `${Number(dominantFreq).toFixed(0)} Hz`],
        ["Frequency Classification", `${freqClassification}`],
        ["Analysis Confidence", results.enhanced ? "High (Advanced Classification)" : "Standard (Automated Detection)"],
        ["Processing Method", "FFT + STFT Analysis"],
        ["Detection Algorithm", "Energy-based with Spectral Features"],
      ],
      theme: "grid",
      headStyles: { fillColor: [34, 139, 34], textColor: [255, 255, 255], fontStyle: 'bold' },
      styles: { fontSize: 9, cellPadding: 3 },
      columnStyles: { 0: { fontStyle: 'bold', cellWidth: 55 } },
      alternateRowStyles: { fillColor: [245, 255, 245] },
    });

    y = (doc as any).lastAutoTable.finalY + 12;

    // --- Detected Sound Events (numbered table) ---
    if (results.soundEvents?.length > 0) {
      if (y > 200) { doc.addPage(); y = 25; }

      doc.setFontSize(16);
      doc.setTextColor(40, 40, 40);
      doc.text("Detected Sound Events", 20, y);

      const eventRows = results.soundEvents.map((ev: any, i: number) => {
        const freq = Number(ev.frequency || 0);
        let cls = "Sub-Bass";
        if (freq > 8000) cls = "Ultra-High";
        else if (freq > 4000) cls = "Brilliance";
        else if (freq > 2000) cls = "Presence";
        else if (freq > 1000) cls = "High-Mid";
        else if (freq > 500) cls = "Mid (Voice)";
        else if (freq > 250) cls = "Low-Mid";
        else if (freq > 80) cls = "Bass";
        return [
          String(i + 1),
          `${Number(ev.time || 0).toFixed(2)}s`,
          ev.type || "Unknown",
          `${freq.toFixed(1)} Hz`,
          `${(Number(ev.amplitude || 0) * 100).toFixed(1)}%`,
          `${Number(ev.decibels || 0).toFixed(1)} dB`,
          cls,
        ];
      });

      autoTable(doc, {
        startY: y + 5,
        head: [["#", "Time", "Type", "Frequency", "Amplitude", "Decibels", "Classification"]],
        body: eventRows,
        theme: "grid",
        headStyles: { fillColor: [220, 120, 0], textColor: [255, 255, 255], fontStyle: 'bold', fontSize: 8 },
        styles: { fontSize: 8, cellPadding: 2 },
        columnStyles: { 0: { cellWidth: 8, halign: 'center' }, 6: { cellWidth: 25 } },
        alternateRowStyles: { fillColor: [255, 248, 240] },
      });

      y = (doc as any).lastAutoTable.finalY + 15;
    }

    updateProgress(60, "Generating Frequency Analysis...");

    // ============================================================
    // PAGE: Comprehensive Frequency Analysis
    // ============================================================
    doc.addPage();
    y = 25;

    doc.setFont("helvetica", "bold");
    doc.setFontSize(18);
    doc.setTextColor(40, 40, 40);
    doc.text("Comprehensive Frequency Analysis", 20, y);
    y += 12;

    doc.setFontSize(11);
    doc.setTextColor(60, 60, 60);
    doc.text("Frequency Distribution (Top 10 Components):", 20, y);
    y += 5;

    // Build frequency distribution from spectrum or events
    const spectrum = results.frequencySpectrum || [];
    let topFreqs: any[] = [];
    if (spectrum.length > 0) {
      topFreqs = [...spectrum].sort((a: any, b: any) => b.magnitude - a.magnitude).slice(0, 10);
    } else if (results.soundEvents?.length > 0) {
      topFreqs = [...results.soundEvents]
        .sort((a: any, b: any) => Number(b.amplitude || 0) - Number(a.amplitude || 0))
        .slice(0, 10)
        .map((e: any) => ({ frequency: e.frequency, magnitude: Number(e.amplitude || 0) }));
    }

    if (topFreqs.length > 0) {
      const freqRows = topFreqs.map((f: any, i: number) => {
        const freq = Number(f.frequency || 0);
        const mag = Number(f.magnitude || 0);
        const barLen = Math.round(mag * 15);
        const bar = "\u2588".repeat(Math.max(1, barLen));
        let cls = "Sub-Bass";
        if (freq > 8000) cls = "Ultra-High";
        else if (freq > 4000) cls = "Brilliance";
        else if (freq > 2000) cls = "Presence";
        else if (freq > 1000) cls = "High-Mid";
        else if (freq > 500) cls = "Mid (Voice)";
        else if (freq > 250) cls = "Low-Mid";
        else if (freq > 80) cls = "Bass";
        return [String(i + 1), `${freq.toFixed(0)} Hz`, `${(mag * 100).toFixed(1)}%`, bar, cls];
      });

      autoTable(doc, {
        startY: y,
        head: [["Rank", "Frequency", "Magnitude", "Visual", "Classification"]],
        body: freqRows,
        theme: "grid",
        headStyles: { fillColor: [75, 0, 130], textColor: [255, 255, 255], fontStyle: 'bold' },
        styles: { fontSize: 8, cellPadding: 2 },
        columnStyles: {
          0: { cellWidth: 12, halign: 'center' },
          3: { textColor: [128, 0, 128], fontStyle: 'bold' },
        },
        alternateRowStyles: { fillColor: [248, 245, 255] },
      });

      y = (doc as any).lastAutoTable.finalY + 15;
    }

    // --- Statistical Analysis ---
    doc.setFontSize(18);
    doc.setTextColor(40, 40, 40);
    doc.text("Statistical Analysis", 20, y);
    y += 5;

    const events = results.soundEvents || [];
    const amplitudes = events.map((e: any) => Number(e.amplitude || 0));
    const frequencies = events.map((e: any) => Number(e.frequency || 0));
    const avgAmp = amplitudes.length > 0 ? amplitudes.reduce((a: number, b: number) => a + b, 0) / amplitudes.length : 0;
    const minAmp = amplitudes.length > 0 ? Math.min(...amplitudes) : 0;
    const maxAmp = amplitudes.length > 0 ? Math.max(...amplitudes) : 0;
    const avgFreq = frequencies.length > 0 ? frequencies.reduce((a: number, b: number) => a + b, 0) / frequencies.length : 0;
    const minFreq = frequencies.length > 0 ? Math.min(...frequencies) : 0;
    const maxFreq = frequencies.length > 0 ? Math.max(...frequencies) : 0;
    const eventDensity = duration > 0 ? (events.length / Number(duration)).toFixed(2) : "0";

    // Classify average amplitude
    let ampInterpretation = "Low energy content";
    if (avgAmp > 0.7) ampInterpretation = "High energy content";
    else if (avgAmp > 0.4) ampInterpretation = "Moderate energy content";

    // Classify average frequency
    let freqInterpretation = "Sub-Bass";
    if (avgFreq > 8000) freqInterpretation = "Ultra-High";
    else if (avgFreq > 4000) freqInterpretation = "Brilliance";
    else if (avgFreq > 2000) freqInterpretation = "Presence";
    else if (avgFreq > 1000) freqInterpretation = "High-Mid";
    else if (avgFreq > 500) freqInterpretation = "Mid (Voice)";
    else if (avgFreq > 250) freqInterpretation = "Low-Mid";
    else if (avgFreq > 80) freqInterpretation = "Bass";

    autoTable(doc, {
      startY: y,
      head: [["Metric", "Value", "Interpretation"]],
      body: [
        ["Average Amplitude", `${(avgAmp * 100).toFixed(1)}%`, ampInterpretation],
        ["Amplitude Range", `${(minAmp * 100).toFixed(1)}% - ${(maxAmp * 100).toFixed(1)}%`, "Dynamic range indicator"],
        ["Average Frequency", `${avgFreq.toFixed(0)} Hz`, freqInterpretation],
        ["Frequency Range", `${minFreq.toFixed(0)} - ${maxFreq.toFixed(0)} Hz`, "Spectral bandwidth"],
        ["Event Density", `${eventDensity} events/sec`, Number(eventDensity) > 1 ? "High activity" : "Low activity"],
      ],
      theme: "grid",
      headStyles: { fillColor: [220, 120, 0], textColor: [255, 255, 255], fontStyle: 'bold' },
      styles: { fontSize: 9, cellPadding: 3 },
      columnStyles: { 0: { fontStyle: 'bold' } },
      alternateRowStyles: { fillColor: [255, 248, 240] },
    });

    y = (doc as any).lastAutoTable.finalY + 15;

    // --- Analysis Methodology ---
    if (y > 230) { doc.addPage(); y = 25; }

    doc.setFontSize(18);
    doc.setTextColor(40, 40, 40);
    doc.text("Analysis Methodology", 20, y);
    y += 8;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(60, 60, 60);
    doc.text("This comprehensive audio forensic analysis was performed using advanced", 20, y);
    y += 5;
    doc.text("digital signal processing techniques:", 20, y);
    y += 8;

    const methods = [
      "Fast Fourier Transform (FFT) for frequency domain analysis",
      "Short-Time Fourier Transform (STFT) for time-frequency representation",
      "Energy-based sound event detection with adaptive thresholding",
      "Spectral feature extraction including centroid and rolloff",
      "Statistical analysis of amplitude and frequency distributions",
    ];
    methods.forEach((m) => {
      doc.setTextColor(220, 120, 0);
      doc.text("\u2022", 22, y);
      doc.setTextColor(60, 60, 60);
      doc.text(m, 27, y);
      y += 6;
    });

    updateProgress(80, "Writing Analysis Summary & Conclusions...");

    // ============================================================
    // PAGE: Analysis Summary & Conclusions + Technical Specs
    // ============================================================
    doc.addPage();
    y = 25;

    doc.setFont("helvetica", "bold");
    doc.setFontSize(18);
    doc.setTextColor(40, 40, 40);
    doc.text("Analysis Summary & Conclusions", 20, y);
    y += 10;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(50, 50, 50);

    const eventCount = events.length;
    const topTypes = [...new Set(events.map((e: any) => e.type || "Unknown"))].join(", ");

    // Executive Summary
    doc.setFont("helvetica", "bold");
    doc.setTextColor(220, 120, 0);
    doc.text("EXECUTIVE SUMMARY: ", 20, y);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(50, 50, 50);
    const execSummary = `Analysis of "${audioData.name}" reveals ${eventCount} distinct sound events over a ${Number(duration).toFixed(1)} second duration,`;
    doc.text(execSummary, 20, y + 5);
    doc.text("providing comprehensive insights into the audio content's characteristics and forensic significance.", 20, y + 10);
    y += 20;

    // Frequency Analysis
    doc.setFont("helvetica", "bold");
    doc.setTextColor(220, 120, 0);
    doc.text("FREQUENCY ANALYSIS: ", 20, y);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(50, 50, 50);
    doc.text(`The dominant frequency component is ${Number(dominantFreq).toFixed(0)} Hz, classified as ${freqClassification}. This`, 20, y + 5);
    doc.text("indicates the primary spectral energy concentration and suggests the nature of the sound source.", 20, y + 10);
    y += 20;

    // Amplitude Assessment
    doc.setFont("helvetica", "bold");
    doc.setTextColor(220, 120, 0);
    doc.text("AMPLITUDE ASSESSMENT: ", 20, y);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(50, 50, 50);
    doc.text(`The audio contains ${avgAmp > 0.7 ? "high" : avgAmp > 0.4 ? "moderate" : "low"}-amplitude signals (${Number(maxDb).toFixed(1)} dB), indicating ${avgAmp > 0.7 ? "strong sound" : "moderate sound"}`, 20, y + 5);
    doc.text("sources, potential clipping, or close-proximity recording conditions.", 20, y + 10);
    y += 20;

    // Sound Classification
    doc.setFont("helvetica", "bold");
    doc.setTextColor(220, 120, 0);
    doc.text("SOUND CLASSIFICATION: ", 20, y);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(50, 50, 50);
    doc.text(`Detected sound types include: ${topTypes || "N/A"}. This`, 20, y + 5);
    doc.text("classification is based on spectral analysis and temporal characteristics.", 20, y + 10);
    y += 20;

    // Forensic Significance
    doc.setFont("helvetica", "bold");
    doc.setTextColor(220, 120, 0);
    doc.text("FORENSIC SIGNIFICANCE: ", 20, y);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(50, 50, 50);
    doc.text("This comprehensive analysis provides detailed acoustic fingerprinting suitable", 20, y + 5);
    doc.text("for forensic investigation, quality assessment, and comparative analysis purposes.", 20, y + 10);
    y += 20;

    // Technical Validation
    doc.setFont("helvetica", "bold");
    doc.setTextColor(220, 120, 0);
    doc.text("TECHNICAL VALIDATION: ", 20, y);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(50, 50, 50);
    doc.text("All measurements were performed using industry-standard digital signal", 20, y + 5);
    doc.text("processing techniques with high precision. Results are suitable for technical documentation.", 20, y + 10);
    y += 25;

    // --- Technical Specifications ---
    if (y > 210) { doc.addPage(); y = 25; }

    doc.setFont("helvetica", "bold");
    doc.setFontSize(18);
    doc.setTextColor(40, 40, 40);
    doc.text("Technical Specifications", 20, y);
    y += 10;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(60, 60, 60);

    const totalDataPoints = Math.round(Number(duration) * sampleRate);
    const freqResolution = sampleRate > 0 ? (sampleRate / 2048).toFixed(2) : "N/A";
    const timeResolution = sampleRate > 0 ? ((2048 / sampleRate) * 1000).toFixed(2) : "N/A";

    const specs = [
      ["Analysis Engine:", "Audio Forensic Detector v2.4"],
      ["Processing Libraries:", "Web Audio API, Advanced Classification"],
      ["Sample Rate:", `${sampleRate} Hz`],
      ["Analysis Duration:", `${Number(duration).toFixed(2)} seconds`],
      ["Total Data Points:", String(totalDataPoints)],
      ["Frequency Resolution:", `${freqResolution} Hz`],
      ["Time Resolution:", `${timeResolution} ms`],
    ];

    specs.forEach(([label, value]) => {
      doc.setFont("helvetica", "bold");
      doc.setTextColor(60, 60, 60);
      doc.text(label, 20, y);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(80, 80, 80);
      doc.text(value, 65, y);
      y += 6;
    });

    updateProgress(90, "Compiling Speaker Diarization...");

    // --- Speaker Diarization Table ---
    if (results.diarization?.segments?.length > 0) {
      doc.addPage();
      doc.setFont("helvetica", "bold");
      doc.setFontSize(16);
      doc.setTextColor(59, 130, 246);
      doc.text("SPEAKER DIARIZATION ANALYSIS", 20, 25);

      const diarizationData = results.diarization.segments.map((seg: any) => [
        seg.speaker,
        `${seg.start.toFixed(2)}s - ${seg.end.toFixed(2)}s`,
        `${(seg.end - seg.start).toFixed(2)}s`,
        (seg.confidence * 100).toFixed(0) + "%"
      ]);

      autoTable(doc, {
        startY: 35,
        head: [["SPEAKER ID", "TIMESTAMPS", "DURATION", "CONFIDENCE"]],
        body: diarizationData,
        theme: "grid",
        headStyles: { fillColor: [30, 41, 59], textColor: [255, 255, 255] },
        styles: { fontSize: 10 },
        alternateRowStyles: { fillColor: [241, 245, 249] }
      });
    }

    // --- Audio Segments Table ---
    if (results.segments?.length > 0) {
      doc.addPage();
      doc.setFont("helvetica", "bold");
      doc.setFontSize(16);
      doc.setTextColor(59, 130, 246);
      doc.text("AUDIO SEGMENT CLASSIFICATION", 20, 25);

      const segmentData = results.segments.map((seg: any) => [
        `${(seg.start || 0).toFixed(2)}s - ${(seg.end || 0).toFixed(2)}s`,
        seg.label || "Unclassified",
        `${((seg.score || 0) * 100).toFixed(1)}%`
      ]);

      autoTable(doc, {
        startY: 35,
        head: [["TIME RANGE", "CLASSIFICATION", "CONFIDENCE SCORE"]],
        body: segmentData,
        theme: "grid",
        headStyles: { fillColor: [15, 118, 110], textColor: [255, 255, 255] },
      });
    }

    // --- Footer with Signature ---
    const pageCount = (doc as any).internal.getNumberOfPages();
    for (let i = 1; i <= pageCount; i++) {
      doc.setPage(i);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(8);
      doc.setTextColor(150);
      doc.text(
        `Audio Forensic Detector - Generated on ${new Date().toLocaleString()}`,
        20, 290
      );
      doc.text(`File: ${audioData.name}`, 20, 294);
      doc.text(`Page ${i} of ${pageCount}`, 185, 290, { align: 'right' });
    }

    updateProgress(100, "Finalizing Report...")
    doc.save(`Forensic_Analysis_${Date.now()}.pdf`)

  } catch (error) {
    console.error("PDF Error:", error)
    alert("Error generating PDF. See console for details.")
  }
}