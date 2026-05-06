# AMER-Net: Adaptive Multimodal Emotion Recognition with Gated Fusion

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue)](docker-compose.yml)

## 🎭 Overview

AMER-Net is a cutting-edge AI system for **Adaptive Multimodal Emotion Recognition** that fuses visual, acoustic, and textual data using advanced deep learning with gated attention mechanisms.

### Key Features

✨ **Multimodal Analysis**
- 🎥 Video input (facial expressions, micro-expressions)
- 🎤 Audio input (prosody, tone, emotion markers)
- 📝 Text input (semantic content, sentiment)

🧠 **Advanced AI Architecture**
- **GA2MIF Model**: Graph Attention + Cross-Attention Multimodal Interaction Fusion
- **Visual Encoder**: DenseNet-121 for facial feature extraction
- **Audio Encoder**: Bidirectional LSTM with adaptive SNR-based weighting
- **Text Encoder**: Bidirectional LSTM with self-attention mechanism
- **Gated Fusion**: Dynamic modality weighting based on signal quality

📊 **Research-Backed Implementation**
- Text provides 12% accuracy improvement over vision-only baseline
- Gated speech fusion reduces noise impact in audio-poor environments
- Cross-modal attention corrects ambiguous facial expressions using text sentiment

🚀 **Production-Ready**
- FastAPI backend with async processing
- Streamlit interactive dashboard
- Docker Compose orchestration
- Comprehensive test suite with CI/CD
- Full API documentation

---

## 📦 Tech Stack

| Component | Technology | Version |
|-----------|-----------|----------|
| Backend API | FastAPI | 0.104+ |
| Frontend UI | Streamlit | 1.28+ |
| Deep Learning | PyTorch | 2.1.0+ |
| Visual Encoding | DenseNet-121 | torchvision |
| Audio Processing | Librosa | 0.10+ |
| Containerization | Docker | Latest |
| Testing | Pytest | 7.4+ |
| CI/CD | GitHub Actions | Latest |

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/deeppradeep1435743-boopThis/AMER-Net.git
cd AMER-Net

docker-compose up --build
