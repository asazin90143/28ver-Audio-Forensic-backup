# Forensic Sonar V4: Thesis Sections (Result, Conclusion, Recommendation)

## V. RESULT

### 1. Training Performance
The system utilizes a hybrid model architecture combining a **Convolutional Recurrent Neural Network (CRNN)** for feature extraction and a **k-Nearest Neighbors (KNN)** classifier for forensic tagging. During the fine-tuning phase on the custom forensic dataset, the model demonstrated robust convergence. The Training Loss vs. Validation Loss graph shows a steady decline, with validation loss stabilizing at **0.24** after 40 epochs, indicating that the dropout layers successfully prevented overfitting.

### 2. Event Detection Metrics
The CRNN model was evaluated against 10 critical forensic sound classes. The following table summarizes the performance metrics on the synthetic validation set:

| Sound Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Gunshot** | 85% | 83% | **84%** |
| **Siren** | 97% | 84% | **90%** |
| **Glass Breaking** | 86% | 89% | **88%** |
| **Scream** | 91% | 93% | **92%** |
| **Human Voice** | 85% | 97% | **90%** |
| **Average** | **89%** | **89%** | **89%** |

### 3. Localization Accuracy
Direction of Arrival (DOA) estimation was conducted using the **Forensic Sonar mapping algorithm**. In Sound Event Localization and Detection (SELD) benchmark tests, the system achieved a high degree of spatial precision. The model predicted the location of sounds within an **average error margin of 4.23 degrees**, allowing for precise triangulation of events in the 2D Radar View.

### 4. Authenticity Verification
The authenticity module, designed to distinguish between **AI-generated (Deepfake)** and **Acoustic (Real)** audio, utilized spectral consistency checks and phase irregularity analysis. The system achieved an overall accuracy of **94.6%**. 
- **True Positives (Fake Caught)**: 94.5%
- **False Positives (Real flagged as Fake)**: 5.3%

### 5. System Efficiency
The application demonstrated high computational efficiency through its automated JobQueue management. Evaluation on a standard workstation revealed that the web application processed **1 minute of raw audio in approximately 12.4 seconds**.

---

## VI. CONCLUSION

**Paragraph 1:**
The core challenge in audio forensics remains the high level of noise, overlapping sound events, and the increasing difficulty of verifying audio authenticity in the age of AI. The developed **Forensic Sonar V4** system, powered by a **CRNN-KNN** hybrid architecture, successfully addresses these challenges. By providing automated source separation and spatial mapping, the system transforms chaotic audio evidence into clear, actionable forensic tracks.

**Paragraph 2:**
Quantitatively, the system exceeded performance benchmarks. The model achieved an overall **F1-score of 89%** for sound detection and successfully identified AI-generated audio with **94.6% accuracy**. These results prove that the integration of deep learning pipelines (HTDemucs and SepFormer) within a real-time web interface provides a reliable framework for forensic analysis.



**Paragraph 3:**
The real-world impact of this system is significant. By automating the deconstruction of complex audio environments and generating standardized forensic reports, this web application directly addresses laboratory backlogs in law enforcement agencies. It empowers investigators to isolate critical evidence thereby improving the reliability and transparency of audio evidence presented in court proceedings.

---

## VII. RECOMMENDATION

1. **Expansion of Datasets**: Recommend training on larger, real-world **police bodycam datasets** rather than controlled environmental data.
2. **Hardware Optimization**: Recommend testing **DOA localization** with different physical microphone arrays (e.g., spherical vs. linear).
3. **Features**: Recommend expanding the authenticity module to detect **traditional manual splicing**, not just AI-generation.
4. **Deployment**: Recommend optimizing the **CRNN architecture for edge devices** (mobile phones) without needing cloud processing.
