# AMER-Net Setup Guide

## Prerequisites
- Python 3.10+
- CUDA 11.8+ (optional, for GPU)
- Docker & Docker Compose (optional)

## Local Installation

### 1. Clone Repository
\`\`\`bash
git clone https://github.com/deeppradeep1435743-boopThis/AMER-Net.git
cd AMER-Net
\`\`\`

### 2. Setup Backend
\`\`\`bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

### 3. Setup Frontend
\`\`\`bash
cd ../frontend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

### 4. Start Backend
\`\`\`bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
\`\`\`

### 5. Start Frontend (new terminal)
\`\`\`bash
cd frontend
streamlit run app.py --server.port 8501
\`\`\`

## Docker Installation

\`\`\`bash
docker-compose up --build
\`\`\`

## Access Points
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs