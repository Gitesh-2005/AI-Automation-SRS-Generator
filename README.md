# AI-Powered SRS Generation System

A comprehensive Software Requirements Specification (SRS) document generation system powered by AI agents, featuring real-time collaboration and multi-format support.

## ✨ Features

### 🤖 AI-Powered Generation
- **LangGraph Integration**: Advanced AI workflow using LangGraph agents
- **Groq LLM**: Fast and efficient language model processing
- **Intelligent Analysis**: Automatic requirement extraction and structuring

### 📄 Multi-Format Support
- **Input Formats**: PDF, Word (.doc/.docx), PowerPoint (.ppt/.pptx), Markdown (.md), LaTeX (.tex)
- **Export Formats**: PDF, Word, Markdown, LaTeX
- **OCR Processing**: PaddleOCR integration for image text extraction

### ⚡ Real-Time Collaboration
- **Live Editing**: CKEditor integration with real-time synchronization
- **Multiple Users**: Support for concurrent editing sessions
- **WebSocket Communication**: Instant updates across all collaborators

### 🎨 Modern Interface
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Beautiful UI**: Apple-level design aesthetics with smooth animations
- **Intuitive Workflow**: Simple 4-step process from upload to collaboration

## 🏗️ Architecture

### Frontend (React + TypeScript)
- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS with custom design system
- **Editor**: CKEditor 5 for rich text editing
- **Animations**: Framer Motion for smooth transitions
- **Routing**: React Router for navigation

### Backend (FastAPI + Python)
- **Framework**: FastAPI for high-performance API
- **AI Engine**: LangGraph with Groq LLM integration
- **Real-time**: Socket.IO for WebSocket communication
- **File Processing**: Multi-format document parsing
- **OCR**: PaddleOCR for image text extraction

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- Python 3.9+
- Groq API key (get from [Groq Console](https://console.groq.com/))

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd SRS-AI-Automation
   ```

2. **Install frontend dependencies**
   ```bash
   npm install
   ```

3. **Install backend dependencies**
   ```bash
   cd backend
   python -m venv venv
   venv\scripts\activate
   pip install -r requirements.txt
   cd..
   ```

   **Run the Project**
   in Frontend Terminal
   ```bash
   npm run dev
   ```
   in Backend Terminal
   ```bash
   uvicorn backend.app:socket_app --host 0.0.0.0 --port 8000
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp backend/.env.example backend/.env
   
   # Edit the .env file and add your API keys
   # Required: GROQ_API_KEY
   # Optional: LANGCHAIN_API_KEY (for LangSmith tracing)
   ```

### Running the Application

**Development Mode (Full Stack)**
```bash
npm start
```
This command runs both the frontend (React) and backend (FastAPI) concurrently.

**Individual Services**
```bash
# Frontend only (runs on http://localhost:3000)
npm run dev

# Backend only (runs on http://localhost:8000)
npm run backend
```

### Usage

1. **Home Page**: Overview of features and capabilities
2. **Upload Page**: Upload documents in various formats
3. **Generate Page**: Interact with AI to generate SRS documents
4. **Editor Page**: Collaborate and edit documents in real-time

## 📚 API Documentation

### File Management
- `POST /api/upload` - Upload and process documents
- `GET /api/documents/{id}` - Retrieve document
- `PUT /api/documents/{id}` - Update document
- `POST /api/export/{id}` - Export document

### AI Integration
- `POST /api/chat` - Chat with AI for SRS generation
- `GET /health` - Health check

### WebSocket Events
- `join-document` - Join collaborative editing session
- `content-change` - Real-time content updates
- `collaborators-update` - Collaborator status updates

## 🛠️ Technology Stack

### Frontend
- React 18
- TypeScript
- Tailwind CSS
- CKEditor 5
- Framer Motion
- Socket.IO Client
- React Router
- Axios

### Backend
- FastAPI
- LangGraph
- Groq API
- Socket.IO
- PaddleOCR
- PyPDF2
- python-docx
- python-pptx

## 📁 Project Structure

```
SRS-AI-Automation/
├── src/
│   ├── components/     # Reusable UI components
│   ├── pages/         # Page components
│   ├── utils/         # Utility functions
│   ├── types/         # TypeScript type definitions
│   └── App.tsx        # Main application component
├── backend/
│   ├── app.py         # FastAPI application
│   ├── requirements.txt # Python dependencies
│   └── .env.example   # Environment variables template
├── package.json       # Node.js dependencies
└── README.md
```

## 🔧 Development

### Adding New Features

1. **Frontend Components**: Add new React components in `src/components/`
2. **API Endpoints**: Add new FastAPI routes in `backend/app.py`
3. **Types**: Update TypeScript types in `src/types/index.ts`

### Integrating Your LangGraph Workflow

Replace the placeholder SRS generation in `backend/app.py` with your actual LangGraph workflow:

```python
# In backend/app.py, replace the generate_srs_content function
async def generate_srs_content(input_text: str) -> str:
    # Import your LangGraph workflow
    from your_srs_generator import graph, FORMAT
    
    # Set up the workflow state
    thread = {"configurable": {"thread_id": 1}}
    state = {
        "messages": [HumanMessage(content=input_text)],
        "srs_format": [HumanMessage(content=FORMAT)],
        "iteration": 1,
        "max_iteration": 2,
    }

    # Execute the workflow
    async for output in graph.stream(state, config=thread, stream_mode="updates"):
        last_message = next(iter(output.values()))["messages"][-1]
        srs_text = last_message.content

    return srs_text
```

## 🚀 Deployment

### Backend Deployment
```bash
# Using Uvicorn
uvicorn backend.app:socket_app --host 0.0.0.0 --port 8000

# Using Docker
docker build -t srs-generator-backend ./backend
docker run -p 8000:8000 srs-generator-backend
```

### Frontend Deployment
```bash
# Build for production
npm run build

# The built files will be in the 'dist' directory
# Deploy to any static hosting service (Vercel, Netlify, etc.)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast web framework
- [React](https://reactjs.org/) - UI library
- [CKEditor 5](https://ckeditor.com/ckeditor-5/) - Rich text editor
- [LangGraph](https://langchain-ai.github.io/langgraph/) - AI workflow framework
- [Groq](https://groq.com/) - Fast LLM inference
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework

## 🐛 Troubleshooting

### Common Issues

1. **PaddleOCR Installation Issues**
   ```bash
   # On Windows, you might need:
   pip install paddleocr paddlepaddle -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
   ```

2. **Port Already in Use**
   ```bash
   # Change ports in package.json or vite.config.ts
   # Backend default: 8000, Frontend default: 3000
   ```

3. **CORS Issues**
   - Make sure backend CORS settings include your frontend URL
   - Check the `CORS_ORIGINS` in your `.env` file

4. **WebSocket Connection Issues**
   - Ensure both frontend and backend are running
   - Check browser developer tools for WebSocket connection errors

## 📞 Support

For support, please open an issue on GitHub or contact the development team.

---

**Built with ❤️ for the software engineering community**
