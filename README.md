# 🏗️ **Complete Professional GitHub Repo Setup**

## 📁 **Repository Structure**
```
kplc-assistant/
├── README.md                 # Main landing page
├── LICENSE                   # MIT License
├── .gitignore                 
├── requirements.txt          # Dependencies
├── .env                      # API keys template
├── kplc_assistant.py         # Core RAG agent  
├── streamlit_kplc_app.py     # Web UI
├── kplc_document.pdf         # Sample data 
└── docs/
    └── DEPLOYMENT.md         # Production guide
```

***

## 📄 **README.md**
```markdown
# 🔌 KPLC Assistant

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-262727?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

**AI-Powered Kenya Power & Lighting Company (KPLC) Customer Support Assistant** with RAG (PDF search) + web search fallback.

![Demo](docs/demo.gif)

## ✨ **Features**
- 🔍 **RAG Pipeline**: Semantic search over KPLC PDFs/documents
- 🌐 **Web Search**: Tavily search across all `*.kplc.co.ke` domains
- 💾 **Memory**: Persistent conversation history per session
- 🖥️ **Web UI**: Professional Streamlit chat interface
- 📱 **Responsive**: Works on desktop + mobile
- ⚡ **Streaming**: Real-time responses

## 🚀 **Quick Start**

### 1. Clone & Setup
```bash
git clone https://github.com/YOUR_USERNAME/kplc-assistant.git
cd kplc-assistant
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
cp .env .env
# Edit .env:
# GOOGLE_API_KEY=your_key
# TAVILY_API_KEY=your_key
```

### 3. Add KPLC Documents
```
📁 kplc_document.pdf     # Your tariffs/connection docs
```

### 4. Launch Web App
```bash
streamlit run streamlit_kplc_app.py
```
**Open**: `http://localhost:8501`

## 💬 **Sample Queries**
```
❓ "What's 1 token unit price?"
❓ "How do I get new electricity connection?"  
❓ "Prepaid vs postpaid differences?"
```

## 🏗️ **Architecture**
```
User Query → Streamlit UI → LangChain Agent → [RAG Tool | Tavily Search] → Gemini → Clean Response
```

## 📊 **Tech Stack**
| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Backend** | LangChain + LangGraph |
| **Vector DB** | ChromaDB |
| **LLM** | Google Gemini 2.5 Flash |
| **Search** | Tavily (KPLC domains) |
| **Embeddings** | Google text-embedding-004 |

## 🤝 **Contributing**
1. Fork the repo
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push & PR!

## 📄 **License**
MIT License - see [LICENSE](LICENSE) © 2026

## 🛠️ **Deployment**
- [Heroku](docs/DEPLOYMENT.md#heroku)
- [Railway](docs/DEPLOYMENT.md#railway)  
- [Docker](docs/DEPLOYMENT.md#docker)
- [VPS](docs/DEPLOYMENT.md#vps)

---





***

## 📋 **docs/DEPLOYMENT.md**
```markdown
# 🚀 Deployment Guide

## Heroku
```bash
heroku create kplc-assistant
heroku config:set GOOGLE_API_KEY=$GOOGLE_API_KEY
heroku config:set TAVILY_API_KEY=$TAVILY_API_KEY
git push heroku main
```

## Railway
```
1. Connect GitHub repo
2. Add env vars: GOOGLE_API_KEY, TAVILY_API_KEY
3. Deploy → https://railway.app
```

## Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_kplc_app.py", "--server.port=8501"]
```
```

***

## 🎯 **Final Steps**
1. **Create GitHub repo**: `kplc-assistant`
2. **Upload ALL files** above + your `kplc_assistant.py` & `streamlit_kplc_app.py`
3. **Replace `YOUR_USERNAME`** in README links
4. **Commit**: `git add . && git commit -m "🎉 Initial KPLC Assistant"`
5. **Push**: `git push origin main`

**Professional repo ready for stars, forks, and deployment!** 🚀🎉
