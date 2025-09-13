import React, { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { motion } from 'framer-motion'
import toast from 'react-hot-toast'
import { 
  Upload, 
  FileText, 
  Image, 
  File,
  CheckCircle2, 
  AlertCircle,
  X,
  ArrowRight,
  Loader2
} from 'lucide-react'
import { uploadFile, getFileType, formatFileSize } from '../utils/api'
import type { UploadResponse, ProcessingStatus } from '../types'

interface UploadedFile {
  file: File
  id: string
  status: ProcessingStatus['status']
  progress: number
  result?: UploadResponse
  error?: string
}

const UploadPage: React.FC = () => {
  const navigate = useNavigate()
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [isUploading, setIsUploading] = useState(false)

  const acceptedFiles = {
    'application/pdf': ['.pdf'],
    'application/msword': ['.doc'],
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    'application/vnd.ms-powerpoint': ['.ppt'],
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
    'text/markdown': ['.md'],
    'text/plain': ['.txt'],
    'application/x-latex': ['.tex'],
    'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
  }

  const processFile = async (file: File): Promise<UploadResponse> => {
    try {
      const response = await uploadFile(file)
      return response
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : 'Upload failed')
    }
  }

  const handleFileUpload = async (files: File[]) => {
    if (files.length === 0) return

    setIsUploading(true)

    const newFiles: UploadedFile[] = files.map(file => ({
      file,
      id: Math.random().toString(36).substr(2, 9),
      status: 'uploading',
      progress: 0
    }))

    setUploadedFiles(prev => [...prev, ...newFiles])

    // Process files sequentially
    for (let i = 0; i < newFiles.length; i++) {
      const fileData = newFiles[i]
      
      try {
        // Update status to processing
        setUploadedFiles(prev => 
          prev.map(f => 
            f.id === fileData.id 
              ? { ...f, status: 'processing', progress: 50 }
              : f
          )
        )

        const result = await processFile(fileData.file)
        
        // Update with success
        setUploadedFiles(prev => 
          prev.map(f => 
            f.id === fileData.id 
              ? { ...f, status: 'completed', progress: 100, result }
              : f
          )
        )

        toast.success(`${fileData.file.name} processed successfully!`)

      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Upload failed'
        
        // Update with error
        setUploadedFiles(prev => 
          prev.map(f => 
            f.id === fileData.id 
              ? { ...f, status: 'error', progress: 0, error: errorMessage }
              : f
          )
        )

        toast.error(`Failed to process ${fileData.file.name}: ${errorMessage}`)
      }
    }

    setIsUploading(false)
  }

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    if (rejectedFiles.length > 0) {
      toast.error(`Some files were rejected. Please check file types and sizes.`)
    }
    
    if (acceptedFiles.length > 0) {
      handleFileUpload(acceptedFiles)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: acceptedFiles,
    maxSize: 50 * 1024 * 1024, // 50MB
    multiple: true
  })

  const removeFile = (id: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== id))
  }

  const getFileIcon = (file: File) => {
    const type = file.type
    
    if (type.includes('pdf')) return <FileText className="h-6 w-6 text-red-500" />
    if (type.includes('word') || type.includes('document')) return <FileText className="h-6 w-6 text-blue-500" />
    if (type.includes('presentation')) return <FileText className="h-6 w-6 text-orange-500" />
    if (type.includes('image')) return <Image className="h-6 w-6 text-green-500" />
    return <File className="h-6 w-6 text-secondary-500" />
  }

  const getStatusIcon = (status: ProcessingStatus['status']) => {
    switch (status) {
      case 'uploading':
      case 'processing':
        return <Loader2 className="h-4 w-4 animate-spin text-primary-600" />
      case 'completed':
        return <CheckCircle2 className="h-4 w-4 text-green-600" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-600" />
      default:
        return null
    }
  }

  const canProceed = uploadedFiles.some(f => f.status === 'completed')

  return (
    <div className="min-h-screen px-4 py-12 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-4xl">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8 text-center"
        >
          <h1 className="mb-4 text-3xl font-bold text-secondary-900 sm:text-4xl">
            Upload Your Documents
          </h1>
          <p className="mx-auto max-w-2xl text-lg text-secondary-600">
            Upload your requirements documents in any supported format. Our AI will analyze 
            and extract content to help generate your SRS document.
          </p>
        </motion.div>

        {/* Upload Area */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="mb-8"
        >
          <div
            {...getRootProps()}
            className={`
              relative cursor-pointer rounded-xl border-2 border-dashed p-12 text-center transition-colors
              ${isDragActive && !isDragReject ? 'drag-active' : ''}
              ${isDragReject ? 'drag-reject' : ''}
              ${!isDragActive ? 'border-secondary-300 hover:border-primary-400 hover:bg-primary-50/50' : ''}
            `}
          >
            <input {...getInputProps()} />
            
            <div className="mx-auto max-w-md">
              <motion.div
                animate={isDragActive ? { scale: 1.1 } : { scale: 1 }}
                transition={{ duration: 0.2 }}
              >
                <Upload className="mx-auto h-12 w-12 text-secondary-400" />
              </motion.div>
              
              <h3 className="mt-4 text-lg font-medium text-secondary-900">
                {isDragActive 
                  ? 'Drop your files here...' 
                  : 'Drag & drop files here, or click to select'
                }
              </h3>
              
              <p className="mt-2 text-sm text-secondary-500">
                Supports PDF, Word, PowerPoint, Markdown, LaTeX, and Images
              </p>
              
              <p className="mt-1 text-xs text-secondary-400">
                Maximum file size: 50MB per file
              </p>

              {/* Supported formats */}
              <div className="mt-6 flex flex-wrap justify-center gap-2">
                {['.pdf', '.docx', '.pptx', '.md', '.tex', '.jpg'].map(format => (
                  <span 
                    key={format}
                    className="rounded-full bg-secondary-100 px-3 py-1 text-xs text-secondary-600"
                  >
                    {format}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Uploaded Files */}
        {uploadedFiles.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mb-8"
          >
            <h2 className="mb-4 text-xl font-semibold text-secondary-900">
              Uploaded Files ({uploadedFiles.length})
            </h2>
            
            <div className="space-y-3">
              {uploadedFiles.map((fileData) => (
                <motion.div
                  key={fileData.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="card"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      {getFileIcon(fileData.file)}
                      
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-secondary-900 truncate">
                          {fileData.file.name}
                        </p>
                        <div className="flex items-center space-x-2 text-xs text-secondary-500">
                          <span>{getFileType(fileData.file.name)}</span>
                          <span>•</span>
                          <span>{formatFileSize(fileData.file.size)}</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-3">
                      {getStatusIcon(fileData.status)}
                      
                      <span className="text-sm capitalize">
                        {fileData.status === 'uploading' && 'Uploading...'}
                        {fileData.status === 'processing' && 'Processing...'}
                        {fileData.status === 'completed' && 'Ready'}
                        {fileData.status === 'error' && 'Failed'}
                      </span>

                      <button
                        onClick={() => removeFile(fileData.id)}
                        className="text-secondary-400 hover:text-secondary-600"
                        disabled={fileData.status === 'uploading' || fileData.status === 'processing'}
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                  </div>

                  {/* Progress bar */}
                  {(fileData.status === 'uploading' || fileData.status === 'processing') && (
                    <div className="mt-3">
                      <div className="w-full bg-secondary-200 rounded-full h-2">
                        <motion.div 
                          className="bg-primary-600 h-2 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${fileData.progress}%` }}
                          transition={{ duration: 0.3 }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Error message */}
                  {fileData.status === 'error' && fileData.error && (
                    <div className="mt-3 rounded-lg bg-red-50 p-3">
                      <p className="text-sm text-red-700">{fileData.error}</p>
                    </div>
                  )}

                  {/* Success preview */}
                  {fileData.status === 'completed' && fileData.result && (
                    <div className="mt-3 rounded-lg bg-green-50 p-3">
                      <p className="text-sm text-green-700">
                        Content extracted successfully! Preview:
                      </p>
                      <p className="mt-1 text-xs text-green-600 truncate">
                        {fileData.result.content_preview}
                      </p>
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Next Steps */}
        {canProceed && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="text-center"
          >
            <div className="card bg-gradient-to-r from-primary-50 to-purple-50 border-primary-200">
              <h3 className="mb-2 text-lg font-semibold text-secondary-900">
                Ready to Generate SRS
              </h3>
              <p className="mb-6 text-secondary-600">
                Your documents have been processed successfully. You can now proceed 
                to generate your SRS document using AI.
              </p>
              
              <div className="flex flex-col space-y-3 sm:flex-row sm:justify-center sm:space-x-3 sm:space-y-0">
                <button
                  onClick={() => {
                    const completedFiles = uploadedFiles.filter(f => f.status === 'completed')
                    if (completedFiles.length > 0) {
                      navigate('/generate', { 
                        state: { uploadedFiles: completedFiles.map(f => f.result) }
                      })
                    }
                  }}
                  className="btn-primary group"
                >
                  Generate SRS Document
                  <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                </button>
                
                <button
                  onClick={() => navigate('/editor')}
                  className="btn-secondary"
                >
                  Start with Blank Editor
                </button>
              </div>
            </div>
          </motion.div>
        )}

        {/* Help Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-12 text-center"
        >
          <h3 className="mb-4 text-lg font-semibold text-secondary-900">
            Need Help?
          </h3>
          <div className="grid gap-6 md:grid-cols-3">
            <div className="text-center">
              <div className="mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100">
                <FileText className="h-6 w-6 text-blue-600" />
              </div>
              <h4 className="text-sm font-medium text-secondary-900">Document Formats</h4>
              <p className="text-xs text-secondary-500">
                We support PDF, Word, PowerPoint, Markdown, LaTeX, and image files
              </p>
            </div>
            
            <div className="text-center">
              <div className="mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-lg bg-green-100">
                <Image className="h-6 w-6 text-green-600" />
              </div>
              <h4 className="text-sm font-medium text-secondary-900">OCR Processing</h4>
              <p className="text-xs text-secondary-500">
                Images and scanned documents are processed using advanced OCR
              </p>
            </div>
            
            <div className="text-center">
              <div className="mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-lg bg-purple-100">
                <Upload className="h-6 w-6 text-purple-600" />
              </div>
              <h4 className="text-sm font-medium text-secondary-900">Batch Upload</h4>
              <p className="text-xs text-secondary-500">
                Upload multiple files at once for comprehensive analysis
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default UploadPage
