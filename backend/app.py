import os
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncGenerator, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import aiofiles
import uuid
from datetime import datetime
import json
import socketio  # type: ignore
import easyocr  # type: ignore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END
import PyPDF2
import docx
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pptx import Presentation
import markdown
from io import BytesIO
from dotenv import load_dotenv
import logging
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from groq import RateLimitError  # Assuming groq SDK is installed; adjust if needed
import hashlib
from collections import deque
from time import time

# Try importing python-magic, with fallback for python-magic-bin (Windows)
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    try:
        import magic  # python-magic-bin provides the same 'magic' module
        MAGIC_AVAILABLE = True
    except ImportError:
        MAGIC_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory setup
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")

for directory in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# Initialize OCR
try:
    ocr = easyocr.Reader(['en'])
    logger.info("EasyOCR initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize EasyOCR: {e}")
    ocr = None

load_dotenv()

API_KEYS = os.getenv("API_KEYS").split(",")

# Global LLMs list and related variables
llms: List[Tuple[ChatGroq, str]] = []  # List of (llm, api_key) tuples
api_calls_per_key: Dict[str, deque] = {}
current_llm_index = 0
MAX_REQUESTS_PER_MINUTE = 25  # Slightly below 30 to be safe
llm_cache: Dict[str, str] = {}

def initialize_groq_llms():
    """Initialize multiple Groq LLMs with different API keys."""
    global llms, api_calls_per_key
    llms = []
    api_calls_per_key = {}
    for idx, api_key in enumerate(API_KEYS):
        if not api_key:
            logger.warning(f"API key {idx+1} not provided")
            continue
        try:
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=api_key,
                temperature=0.1,
                max_tokens=4096
            )
            llms.append((llm, api_key))  # Store LLM and its API key
            api_calls_per_key[api_key] = deque(maxlen=60)  # Track calls per key
            logger.info(f"Groq LLM initialized successfully for API key {idx+1}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM for API key {idx+1}: {e}")
    return len(llms) > 0

def get_next_llm() -> Optional[Tuple[ChatGroq, str]]:
    """Get the next available LLM and its API key, rotating through the list."""
    global current_llm_index
    if not llms:
        return None
    llm_tuple = llms[current_llm_index]
    current_llm_index = (current_llm_index + 1) % len(llms)  # Round-robin
    return llm_tuple

def get_cache_key(input_dict: dict) -> str:
    """Generate a cache key from input dictionary."""
    input_str = json.dumps(input_dict, sort_keys=True)
    return hashlib.md5(input_str.encode('utf-8')).hexdigest()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(RateLimitError)
)
async def invoke_llm_with_retry(chain, input_dict: dict) -> str:
    """Invoke LLM with retry logic and rate limit handling across multiple API keys."""
    cache_key = get_cache_key(input_dict)
    if cache_key in llm_cache:
        logger.info(f"Cache hit for key: {cache_key}")
        return llm_cache[cache_key]
    
    for attempt in range(len(llms)):  # Try each LLM once
        llm_tuple = get_next_llm()
        if not llm_tuple:
            return "No LLM available"
        llm, api_key = llm_tuple
        
        current_time = time()
        api_calls = api_calls_per_key.get(api_key, deque(maxlen=60))
        
        # Remove calls older than 60 seconds
        while api_calls and current_time - api_calls[0] > 60:
            api_calls.popleft()
        
        # Throttle if approaching rate limit
        while len(api_calls) >= MAX_REQUESTS_PER_MINUTE:
            logger.warning(f"Rate limit approaching for API key {api_key[-4:]}: {len(api_calls)} calls in last 60s. Waiting...")
            await asyncio.sleep(1)
            current_time = time()
            while api_calls and current_time - api_calls[0] > 60:
                api_calls.popleft()
        
        api_calls.append(current_time)
        logger.info(f"Making API call with input keys: {input_dict.keys()} using API key {api_key[-4:]}")
        
        try:
            result = await chain.ainvoke(input_dict)
            logger.debug(f"Raw LLM response: {result}")  # Log raw response for debugging
            llm_cache[cache_key] = result  # Cache the result
            await asyncio.sleep(2)  # Reduced delay
            logger.info("API call completed successfully")
            return result
        except RateLimitError as e:
            headers = getattr(e, 'response', {}).headers if hasattr(e, 'response') else {}
            retry_after = headers.get("retry-after", "unknown")
            remaining = headers.get("x-ratelimit-remaining", "unknown")
            reset_time = headers.get("x-ratelimit-reset", "unknown")
            logger.warning(f"Rate limit hit for API key {api_key[-4:]}: Retry-After: {retry_after}, Remaining: {remaining}, Reset: {reset_time}")
            if attempt < len(llms) - 1:  # Try next LLM if not the last one
                logger.info("Switching to next API key")
                continue
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LLM invocation with API key {api_key[-4:]}: {e}")
            if attempt < len(llms) - 1:
                continue
            raise
    
    return "All API keys hit rate limits"

async def invoke_llm_stream_with_retry(chain, input_dict: dict) -> AsyncGenerator[str, None]:
    """Stream LLM response with retry logic and rate limit handling across multiple API keys."""
    cache_key = get_cache_key(input_dict)
    if cache_key in llm_cache:
        logger.info(f"Cache hit for key: {cache_key}")
        # Stream cached response word by word for better UX
        cached_response = llm_cache[cache_key]
        words = cached_response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.05)  # Small delay for streaming effect
        return
    
    full_response = ""
    for attempt in range(len(llms)):  # Try each LLM once
        llm_tuple = get_next_llm()
        if not llm_tuple:
            yield "No LLM available"
            return
        llm, api_key = llm_tuple
        
        current_time = time()
        api_calls = api_calls_per_key.get(api_key, deque(maxlen=60))
        
        while api_calls and current_time - api_calls[0] > 60:
            api_calls.popleft()
        
        while len(api_calls) >= MAX_REQUESTS_PER_MINUTE:
            logger.warning(f"Rate limit approaching for API key {api_key[-4:]}: {len(api_calls)} calls in last 60s. Waiting...")
            await asyncio.sleep(1)
            current_time = time()
            while api_calls and current_time - api_calls[0] > 60:
                api_calls.popleft()
        
        api_calls.append(current_time)
        logger.info(f"Making streaming API call with input keys: {input_dict.keys()} using API key {api_key[-4:]}")
        
        try:
            async for chunk in chain.astream(input_dict):
                full_response += chunk
                yield chunk
            llm_cache[cache_key] = full_response  # Cache the full response
            await asyncio.sleep(2)  # Reduced delay
            logger.info("Streaming API call completed successfully")
            return
        except RateLimitError as e:
            headers = getattr(e, 'response', {}).headers if hasattr(e, 'response') else {}
            retry_after = headers.get("retry-after", "unknown")
            remaining = headers.get("x-ratelimit-remaining", "unknown")
            reset_time = headers.get("x-ratelimit-reset", "unknown")
            logger.warning(f"Rate limit hit for API key {api_key[-4:]}: Retry-After: {retry_after}, Remaining: {remaining}, Reset: {reset_time}")
            if attempt < len(llms) - 1:
                logger.info("Switching to next API key for streaming")
                continue
            yield f"Error: All API keys hit rate limits"
            return
        except Exception as e:
            logger.error(f"Unexpected error in LLM streaming invocation with API key {api_key[-4:]}: {e}")
            yield f"Error: {str(e)}"
            return

# FastAPI app
app = FastAPI(
    title="AI SRS Generator",
    description="AI-powered Software Requirements Specification generation system with Groq LLM",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socket.IO setup
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
socket_app = socketio.ASGIApp(sio, app)

# Pydantic models
class ChatMessage(BaseModel):
    content: str
    user_input: Optional[str] = None
    document_id: Optional[str] = None

class DocumentResponse(BaseModel):
    id: str
    filename: str
    content: str
    created_at: datetime
    file_type: str

class ExportRequest(BaseModel):
    document_id: str
    format: str  # 'pdf', 'docx', 'md', 'latex'
    content: str

class StreamSRSRequest(BaseModel):
    content: str
    document_id: Optional[str] = None

# In-memory storage for demo (replace with database in production)
documents_store: Dict[str, Dict[str, Any]] = {}
active_sessions: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, WebSocket] = {}

# SRS Generation Workflow using LangGraph
class SRSState(MessagesState):
    """State for SRS generation workflow"""
    requirements_analysis: Optional[str] = None
    functional_requirements: Optional[str] = None
    non_functional_requirements: Optional[str] = None
    system_architecture: Optional[str] = None
    final_srs: Optional[str] = None

async def generate_complete_srs_simple(user_content: str) -> str:
    """Generate SRS directly without complex JSON parsing."""
    llm_tuple = get_next_llm()
    if not llm_tuple:
        return "LLM not available"
    llm, _ = llm_tuple
    
    # Simplified prompt that returns markdown directly
    simple_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert software requirements analyst. Generate a comprehensive Software Requirements Specification (SRS) document in markdown format based on the user's input.

Structure your response as a complete SRS document with these sections:
# Software Requirements Specification

## 1. Introduction
### 1.1 Purpose
### 1.2 Scope
### 1.3 Definitions and Abbreviations
### 1.4 References
### 1.5 Overview

## 2. Overall Description
### 2.1 Product Perspective
### 2.2 Product Functions
### 2.3 User Characteristics
### 2.4 Constraints
### 2.5 Assumptions and Dependencies

## 3. Specific Requirements
### 3.1 Functional Requirements
### 3.2 Non-Functional Requirements
### 3.3 External Interface Requirements
### 3.4 Performance Requirements

## 4. System Architecture
### 4.1 System Overview
### 4.2 Component Design
### 4.3 Data Architecture

## 5. Verification and Validation
### 5.1 Testing Requirements
### 5.2 Acceptance Criteria

Provide detailed, professional content for each section. Use proper markdown formatting with headers, lists, and tables where appropriate."""),
        ("human", "{input}")
    ])
    
    try:
        chain = simple_prompt | llm | StrOutputParser()
        result = await invoke_llm_with_retry(chain, {"input": user_content})
        return result
    except Exception as e:
        logger.error(f"Error generating SRS: {e}")
        return f"Error generating SRS: {str(e)}"

async def stream_srs_generation(user_content: str) -> AsyncGenerator[str, None]:
    """Stream SRS generation with simplified approach."""
    llm_tuple = get_next_llm()
    if not llm_tuple:
        yield "LLM not available"
        return
    llm, _ = llm_tuple
    
    # Simplified streaming prompt
    stream_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert software requirements analyst. Generate a comprehensive Software Requirements Specification (SRS) document in markdown format based on the user's input.

Structure your response as a complete SRS document with these sections:
# Software Requirements Specification

## 1. Introduction
### 1.1 Purpose
### 1.2 Scope
### 1.3 Definitions and Abbreviations
### 1.4 References
### 1.5 Overview

## 2. Overall Description
### 2.1 Product Perspective
### 2.2 Product Functions
### 2.3 User Characteristics
### 2.4 Constraints
### 2.5 Assumptions and Dependencies

## 3. Specific Requirements
### 3.1 Functional Requirements
### 3.2 Non-Functional Requirements
### 3.3 External Interface Requirements
### 3.4 Performance Requirements

## 4. System Architecture
### 4.1 System Overview
### 4.2 Component Design
### 4.3 Data Architecture

## 5. Verification and Validation
### 5.1 Testing Requirements
### 5.2 Acceptance Criteria

Provide detailed, professional content for each section. Use proper markdown formatting."""),
        ("human", "{input}")
    ])
    
    try:
        chain = stream_prompt | llm | StrOutputParser()
        async for chunk in invoke_llm_stream_with_retry(chain, {"input": user_content}):
            yield chunk
    except Exception as e:
        logger.error(f"Error streaming SRS: {e}")
        yield f"Error streaming SRS: {str(e)}"

# File processing utilities (keeping existing functions)
def get_file_type(file_path: Path) -> str:
    """Determine file type using python-magic or file extension fallback"""
    if MAGIC_AVAILABLE:
        try:
            return magic.from_file(str(file_path), mime=True)
        except Exception as e:
            logger.warning(f"python-magic failed: {e}. Falling back to extension-based detection.")
    
    # Fallback to extension-based detection
    extension = file_path.suffix.lower()
    mime_types = {
        '.txt': 'text/plain',
        '.pdf': 'application/pdf',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.ppt': 'application/vnd.ms-powerpoint',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.json': 'application/json',
    }
    file_type = mime_types.get(extension, 'application/octet-stream')
    logger.info(f"Using extension-based file type: {file_type} for {file_path}")
    return file_type

async def extract_text_from_file(file_path: Path, file_type: str) -> str:
    """Extract text from various file formats"""
    try:
        if file_type.startswith('text/') or file_type == 'application/json':
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        
        elif file_type == 'application/pdf':
            return extract_pdf_text(file_path)
        
        elif file_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
            return extract_docx_text(file_path)
        
        elif file_type in ['application/vnd.openxmlformats-officedocument.presentationml.presentation', 'application/vnd.ms-powerpoint']:
            return extract_pptx_text(file_path)
        
        elif file_type.startswith('image/') and ocr:
            return extract_text_from_image(file_path)
        
        else:
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return await f.read()
            
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""

def extract_pdf_text(file_path: Path) -> str:
    """Extract text from PDF files"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return ""

def extract_docx_text(file_path: Path) -> str:
    """Extract text from DOCX files"""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {e}")
        return ""

def extract_pptx_text(file_path: Path) -> str:
    """Extract text from PPTX files"""
    try:
        prs = Presentation(file_path)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting PPTX text: {e}")
        return ""

def extract_text_from_image(image_path: Path) -> str:
    """Extract text from images using EasyOCR"""
    if not ocr:
        return ""
    
    try:
        results = ocr.readtext(str(image_path))
        text = ""
        for _, detected_text, conf in results:
            if conf > 0.5:
                text += detected_text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return ""

# Export utilities
def create_pdf_export(content: str, output_path: Path) -> Path:
    """Create a properly formatted PDF export"""
    try:
        doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER,
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
        )
        
        flowables = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                flowables.append(Spacer(1, 6))
                continue
            
            if line.startswith('# '):
                flowables.append(Paragraph(line[2:], title_style))
                flowables.append(Spacer(1, 12))
            elif line.startswith('## '):
                flowables.append(Paragraph(line[3:], heading_style))
            elif line.startswith('### '):
                flowables.append(Paragraph(line[4:], subheading_style))
            else:
                flowables.append(Paragraph(line, body_style))
        
        doc.build(flowables)
        logger.info(f"PDF created successfully: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating PDF: {e}")
        raise

def create_docx_export(content: str, output_path: Path) -> Path:
    """Create a properly formatted DOCX export"""
    try:
        doc = docx.Document()
        
        title = doc.add_heading('Software Requirements Specification', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                doc.add_paragraph()
                continue
            
            if line.startswith('# '):
                doc.add_heading(line[2:], level=1)
            elif line.startswith('## '):
                doc.add_heading(line[3:], level=2)
            elif line.startswith('### '):
                doc.add_heading(line[4:], level=3)
            elif line.startswith('#### '):
                doc.add_heading(line[5:], level=4)
            else:
                doc.add_paragraph(line)
        
        doc.save(str(output_path))
        logger.info(f"DOCX created successfully: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating DOCX: {e}")
        raise

# Routes
@app.get("/")
async def home():
    """Root endpoint"""
    return {
        "message": "Welcome to AI SRS Generator with Groq LLM",
        "version": "2.0.0",
        "status": "active",
        "llm_available": len(llms) > 0,
        "magic_available": MAGIC_AVAILABLE
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ocr_available": ocr is not None,
        "llm_available": len(llms) > 0,
        "magic_available": MAGIC_AVAILABLE,
        "num_llms": len(llms)
    }

# WebSocket endpoint for streaming SRS generation
@app.websocket("/stream-srs")
async def websocket_stream_srs(websocket: WebSocket):
    """WebSocket endpoint for streaming SRS generation"""
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    websocket_connections[connection_id] = websocket
    logger.info(f"WebSocket connection established: {connection_id}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            content = message_data.get("content", "")
            document_id = message_data.get("document_id")
            
            logger.info(f"Received streaming request: {content[:100]}...")
            
            if not content.strip():
                await websocket.send_text(json.dumps({
                    "error": "No content provided"
                }))
                continue
            
            # Add document context if available
            if document_id and document_id in documents_store:
                doc = documents_store[document_id]
                content = f"{content}\n\nDocument Context:\n{doc['content']}"
            
            try:
                full_response = ""
                # Stream the SRS generation
                async for chunk in stream_srs_generation(content):
                    if chunk:
                        full_response += chunk
                        await websocket.send_text(json.dumps({
                            "chunk": chunk
                        }))
                        await asyncio.sleep(0.01)  # Small delay for better streaming experience
                
                # Send final response
                await websocket.send_text(json.dumps({
                    "final": {
                        "final_srs": full_response
                    }
                }))
                
                logger.info("Streaming SRS generation completed")
                
            except Exception as e:
                logger.error(f"Error in streaming SRS generation: {e}")
                await websocket.send_text(json.dumps({
                    "error": f"Error generating SRS: {str(e)}"
                }))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
        if connection_id in websocket_connections:
            del websocket_connections[connection_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_text(json.dumps({
            "error": f"WebSocket error: {str(e)}"
        }))
        if connection_id in websocket_connections:
            del websocket_connections[connection_id]

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process document files"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    document_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    file_path = UPLOAD_DIR / f"{document_id}{file_extension}"
    
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        file_type = get_file_type(file_path)
        extracted_text = await extract_text_from_file(file_path, file_type)
        
        documents_store[document_id] = {
            "id": document_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "file_type": file_type,
            "content": extracted_text,
            "created_at": datetime.now(),
            "size": len(content)
        }
        
        return {
            "status": "success",
            "document_id": document_id,
            "filename": file.filename,
            "file_type": file_type,
            "content_preview": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
            "size": len(content)
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/chat")
async def chat_with_ai(message: ChatMessage):
    """Generate SRS content using simplified AI approach"""
    if not llms:
        raise HTTPException(status_code=503, detail="LLM not available. Please check API keys.")
    
    try:
        user_content = message.content
        document_context = ""
        
        if message.document_id and message.document_id in documents_store:
            doc = documents_store[message.document_id]
            document_context = f"\n\nDocument Content:\n{doc['content']}"
        
        full_input = user_content + document_context
        logger.info("Starting simplified SRS generation...")
        
        # Generate SRS using simplified approach
        result = await generate_complete_srs_simple(full_input)
        
        logger.info("SRS generation completed successfully")
        
        return {
            "response": result,
            "status": "success"
        }
        
    except RateLimitError as e:
        headers = getattr(e, 'response', {}).headers if hasattr(e, 'response') else {}
        retry_after = headers.get("retry-after", "60")
        raise HTTPException(
            status_code=429, 
            detail=f"Rate limit exceeded. Please try again in {retry_after} seconds."
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating SRS: {str(e)}")

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Retrieve document by ID"""
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents_store[document_id]
    return DocumentResponse(
        id=doc["id"],
        filename=doc["filename"],
        content=doc["content"],
        created_at=doc["created_at"],
        file_type=doc["file_type"]
    )

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    docs = []
    for doc in documents_store.values():
        docs.append({
            "id": doc["id"],
            "filename": doc["filename"],
            "file_type": doc["file_type"],
            "created_at": doc["created_at"],
            "size": doc["size"]
        })
    return {"documents": docs}

@app.put("/documents/{document_id}")
async def update_document(document_id: str, content: dict):
    """Update document content"""
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    documents_store[document_id]["content"] = content.get("content", "")
    documents_store[document_id]["updated_at"] = datetime.now()
    return {"status": "success", "document_id": document_id}

@app.post("/export/{document_id}")
async def export_document(document_id: str, export_request: ExportRequest):
    """Export document in specified format"""
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        doc = documents_store[document_id]
        content = export_request.content or doc["content"]
        
        export_path = await create_export_file(document_id, content, export_request.format)
        media_types = {
            'pdf': 'application/pdf',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'md': 'text/markdown',
            'latex': 'application/x-latex'
        }
        media_type = media_types.get(export_request.format, 'application/octet-stream')
        
        return FileResponse(
            path=export_path,
            filename=f"{Path(doc['filename']).stem}_srs.{export_request.format}",
            media_type=media_type
        )
        
    except Exception as e:
        logger.error(f"Error exporting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error exporting document: {str(e)}")

async def create_export_file(document_id: str, content: str, format: str) -> Path:
    """Create export file in specified format"""
    export_filename = f"{document_id}_export.{format}"
    export_path = OUTPUT_DIR / export_filename
    
    if format == "md":
        async with aiofiles.open(export_path, 'w', encoding='utf-8') as f:
            await f.write(content)
    
    elif format == "pdf":
        export_path = export_path.with_suffix('.pdf')
        create_pdf_export(content, export_path)
    
    elif format == "docx":
        export_path = export_path.with_suffix('.docx')
        create_docx_export(content, export_path)
    
    elif format == "latex":
        latex_content = f"""\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{hyperref}}
\\usepackage{{geometry}}
\\geometry{{margin=1in}}

\\title{{Software Requirements Specification}}
\\author{{AI SRS Generator}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle
\\tableofcontents
\\newpage

{content.replace('#', '\\section{').replace('##', '\\subsection{').replace('###', '\\subsubsection{')}

\\end{{document}}"""
        async with aiofiles.open(export_path, 'w', encoding='utf-8') as f:
            await f.write(latex_content)
    
    return export_path

# WebSocket handlers for collaboration
@sio.event
async def connect(sid, environ):
    logger.info(f"Client {sid} connected")
    await sio.emit('connected', {'status': 'connected'}, room=sid)

@sio.event
async def disconnect(sid):
    logger.info(f"Client {sid} disconnected")
    if sid in active_sessions:
        del active_sessions[sid]

@sio.event
async def join_document(sid, data):
    document_id = data.get('document_id')
    user_id = data.get('user_id', f'user_{sid[:8]}')
    
    if document_id:
        active_sessions[sid] = {
            'user_id': user_id,
            'document_id': document_id
        }
        
        await sio.enter_room(sid, document_id)
        
        # Notify other users
        await sio.emit('user_joined', {
            'user_id': user_id,
            'document_id': document_id
        }, room=document_id, skip_sid=sid)
        
        # Send current document state if it exists
        if document_id in documents_store:
            doc = documents_store[document_id]
            await sio.emit('document_state', {
                'content': doc['content'],
                'document_id': document_id
            }, room=sid)

@sio.event
async def content_change(sid, data):
    if sid in active_sessions:
        session = active_sessions[sid]
        document_id = session['document_id']
        content = data.get('content', '')
        
        if document_id in documents_store:
            documents_store[document_id]['content'] = content
            documents_store[document_id]['updated_at'] = datetime.now()
            
            # Broadcast to other users in the document
            await sio.emit('content_updated', {
                'content': content,
                'user_id': session['user_id'],
                'timestamp': datetime.now().isoformat()
            }, room=document_id, skip_sid=sid)

# Workflow management endpoints
@app.get("/workflow/status")
async def get_workflow_status():
    return {
        "llm_available": len(llms) > 0,
        "ocr_available": ocr is not None,
        "magic_available": MAGIC_AVAILABLE,
        "num_llms": len(llms),
        "supported_formats": ["pdf", "docx", "md", "latex"],
        "supported_upload_types": [
            "text/*", "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "image/*"
        ],
        "websocket_connections": len(websocket_connections)
    }

@app.post("/workflow/test")
async def test_workflow():
    if not llms:
        raise HTTPException(status_code=503, detail="LLM not available")
    
    test_input = """
    Test Requirements:
    - Create a web-based task management system
    - Users should be able to create, edit, and delete tasks
    - System should support user authentication
    - Tasks should have priorities and due dates
    - System should send email notifications for due tasks
    """
    
    try:
        result = await generate_complete_srs_simple(test_input)
        return {
            "status": "success",
            "message": "Workflow test completed successfully",
            "preview": result[:500] + "..." if len(result) > 500 else result
        }
    
    except Exception as e:
        logger.error(f"Workflow test failed: {e}")
        return {
            "status": "error",
            "message": f"Workflow test failed: {str(e)}"
        }

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        doc = documents_store[document_id]
        file_path = Path(doc["file_path"])
        if file_path.exists():
            file_path.unlink()
        del documents_store[document_id]
        return {
            "status": "success",
            "message": f"Document {document_id} deleted successfully"
        }
    
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.post("/documents/{document_id}/analyze")
async def analyze_document(document_id: str):
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if not llms:
        raise HTTPException(status_code=503, detail="LLM not available")
    
    try:
        doc = documents_store[document_id]
        content = doc["content"]
        
        analysis_input = f"""
        Analyze the following document content and extract software requirements:
        
        Document: {doc['filename']}
        Content: {content}
        
        Please identify:
        1. Business requirements and objectives
        2. Functional requirements
        3. Non-functional requirements
        4. Constraints and assumptions
        5. Stakeholders and their needs
        
        Provide a structured analysis in markdown format.
        """
        
        result = await generate_complete_srs_simple(analysis_input)
        
        documents_store[document_id]["analysis"] = {
            "analysis_result": result,
            "analyzed_at": datetime.now()
        }
        
        return {
            "status": "success",
            "document_id": document_id,
            "analysis": result[:1000] + "..." if len(result) > 1000 else result,
            "full_analysis_available": True
        }
    
    except Exception as e:
        logger.error(f"Error analyzing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing document: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("AI SRS Generator starting up...")
    initialize_groq_llms()
    logger.info(f"Groq LLMs available: {len(llms) > 0}")
    logger.info(f"OCR available: {ocr is not None}")
    logger.info(f"python-magic available: {MAGIC_AVAILABLE}")
    
    for directory in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
        directory.mkdir(exist_ok=True)
    
    logger.info("AI SRS Generator started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("AI SRS Generator shutting down...")
    try:
        # Close all WebSocket connections
        for connection_id, websocket in websocket_connections.items():
            try:
                await websocket.close()
            except:
                pass
        websocket_connections.clear()
        
        # Clean up temp files
        for file_path in TEMP_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    logger.info("AI SRS Generator shutdown complete")

if __name__ == "__main__":
    import uvicorn
    
    if not any(API_KEYS):
        print("WARNING: No GROQ_API_KEYS provided")
        print("Please add your Groq API keys to the API_KEYS list")
    
    print("Starting AI SRS Generator with Groq LLM integration...")
    print("Features:")
    print("- File upload and text extraction (PDF, DOCX, PPTX, images)")
    print("- AI-powered SRS generation using Groq Llama 3.1 8B")
    print("- WebSocket streaming SRS generation (/stream-srs)")
    print("- Professional PDF and DOCX export")
    print("- Real-time collaboration via Socket.IO")
    print("- RESTful API for integration")
    print("- Simplified LLM approach for better reliability")
    
    uvicorn.run(
        socket_app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )