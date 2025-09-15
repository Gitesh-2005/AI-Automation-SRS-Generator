// api.ts
import axios from 'axios'

import type { 
  Document, 
  ChatMessage, 
  ChatResponse, 
  UploadResponse, 
  ExportRequest, 
  HealthCheck,
  AIEditRequest,
  AIEditResponse
} from '../types'

// Create axios instance with default config
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('[API] Request error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    console.error('[API] Response error:', error)
    if (error.response) {
      // Server responded with error status
      const message = error.response.data?.detail || error.response.statusText
      throw new Error(`API Error: ${message}`)
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('Network Error: Unable to connect to server')
    } else {
      // Something else happened
      throw new Error(`Request Error: ${error.message}`)
    }
  }
)

// Health check
export const healthCheck = async (): Promise<HealthCheck> => {
  const response = await api.get('/health')
  return response.data
}

// Document operations
export const uploadFile = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  
  return response.data
}

export const getDocument = async (documentId: string): Promise<Document> => {
  const response = await api.get(`/documents/${documentId}`)
  return response.data
}

export const listDocuments = async (): Promise<{ documents: Document[] }> => {
  const response = await api.get('/documents')
  return response.data
}

export const updateDocument = async (
  documentId: string, 
  content: string
): Promise<{ status: string; document_id: string }> => {
  const response = await api.put(`/documents/${documentId}`, { content })
  return response.data
}

// Chat operations
export const sendChatMessage = async (message: ChatMessage): Promise<ChatResponse> => {
  const response = await api.post('/chat', message)
  return response.data
}

// AI Edit operations
export const sendAIEditRequest = async (editRequest: AIEditRequest): Promise<AIEditResponse> => {
  const response = await api.post('/ai-edit', editRequest)
  return response.data
}

// Export operations
export const exportDocument = async (
  documentId: string, 
  exportRequest: Omit<ExportRequest, 'document_id'>
): Promise<Blob> => {
  const response = await api.post(`/export/${documentId}`, {
    document_id: documentId,
    ...exportRequest,
  }, {
    responseType: 'blob',
  })
  
  return response.data
}

// Helper function to download blob as file
export const downloadBlob = (blob: Blob, filename: string) => {
  const url = window.URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  window.URL.revokeObjectURL(url)
}

// Direct export function for better control
export const exportDirect = async (content: string, format: string, filename?: string): Promise<Blob> => {
  const response = await api.post('/export-direct', {
    content,
    format,
    filename: filename || 'srs-document'
  }, {
    responseType: 'blob'
  })
  
  return response.data
}

// File type detection
export const getFileType = (filename: string): string => {
  const extension = filename.split('.').pop()?.toLowerCase()
  
  switch (extension) {
    case 'pdf':
      return 'PDF Document'
    case 'doc':
    case 'docx':
      return 'Word Document'
    case 'ppt':
    case 'pptx':
      return 'PowerPoint Presentation'
    case 'md':
      return 'Markdown Document'
    case 'txt':
      return 'Text Document'
    case 'json':
      return 'JSON File'
    default:
      return 'Unknown File Type'
  }
}

// File size formatter
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes'
  
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

// Date formatter
export const formatDate = (dateString: string): string => {
  const date = new Date(dateString)
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export default api