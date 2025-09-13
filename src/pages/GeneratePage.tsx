import React, { useState, useRef, useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import toast from 'react-hot-toast'
import { 
  Send, 
  Bot, 
  User, 
  FileText,
  Loader2,
  Copy,
  Download,
  Edit3,
  ArrowRight,
  Sparkles,
  MessageSquare
} from 'lucide-react'
import { sendChatMessage } from '../utils/api'
import type { ChatMessage, ChatResponse, UploadResponse } from '../types'

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: Date
  isLoading?: boolean
}

const GeneratePage: React.FC = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedSRS, setGeneratedSRS] = useState<string>('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Get uploaded files from location state
  const uploadedFiles: UploadResponse[] = location.state?.uploadedFiles || []

  useEffect(() => {
    // Add welcome message
    const welcomeMessage: Message = {
      id: '1',
      type: 'assistant',
      content: uploadedFiles.length > 0 
        ? `Hello! I've analyzed your ${uploadedFiles.length} uploaded document(s). I'm ready to help you generate a comprehensive SRS document. What specific requirements would you like me to focus on?`
        : `Hello! I'm your AI assistant for SRS generation. Describe your project requirements, and I'll help you create a professional Software Requirements Specification document.`,
      timestamp: new Date()
    }
    setMessages([welcomeMessage])

    // Focus input
    inputRef.current?.focus()
  }, [uploadedFiles])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputValue.trim() || isGenerating) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    }

    const loadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: 'assistant',
      content: 'Generating your SRS document...',
      timestamp: new Date(),
      isLoading: true
    }

    setMessages(prev => [...prev, userMessage, loadingMessage])
    setInputValue('')
    setIsGenerating(true)

    try {
      const chatMessage: ChatMessage = {
        content: inputValue,
        document_id: uploadedFiles[0]?.document_id
      }

      const response: ChatResponse = await sendChatMessage(chatMessage)
      
      // Remove loading message and add AI response
      setMessages(prev => prev.filter(m => !m.isLoading))
      
      const aiMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: response.response,
        timestamp: new Date()
      }

      setMessages(prev => [...prev, aiMessage])
      setGeneratedSRS(response.response)
      
      toast.success('SRS document generated successfully!')

    } catch (error) {
      // Remove loading message
      setMessages(prev => prev.filter(m => !m.isLoading))
      
      const errorMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: 'Sorry, I encountered an error while generating your SRS document. Please try again.',
        timestamp: new Date()
      }

      setMessages(prev => [...prev, errorMessage])
      toast.error(error instanceof Error ? error.message : 'Failed to generate SRS')
    } finally {
      setIsGenerating(false)
    }
  }

  const copyToClipboard = (content: string) => {
    navigator.clipboard.writeText(content)
    toast.success('Content copied to clipboard!')
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const suggestedPrompts = [
    "Create a complete SRS for a web application",
    "Focus on functional requirements and user stories",
    "Generate system architecture and technical specifications",
    "Include security and performance requirements",
    "Create a mobile app SRS with UI/UX requirements"
  ]

  return (
    <div className="flex h-screen bg-secondary-50">
      {/* Sidebar */}
      <div className="w-80 bg-white border-r border-secondary-200 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-secondary-200">
          <div className="flex items-center space-x-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary-500 to-purple-600 text-white">
              <Bot className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-secondary-900">
                AI SRS Generator
              </h1>
              <p className="text-sm text-secondary-500">
                Intelligent Assistant
              </p>
            </div>
          </div>
        </div>

        {/* Uploaded Files */}
        {uploadedFiles.length > 0 && (
          <div className="p-6 border-b border-secondary-200">
            <h2 className="mb-3 text-sm font-medium text-secondary-900">
              Uploaded Documents
            </h2>
            <div className="space-y-2">
              {uploadedFiles.map((file, index) => (
                <div key={index} className="flex items-center space-x-2 rounded-lg bg-secondary-50 p-2">
                  <FileText className="h-4 w-4 text-secondary-500" />
                  <span className="text-xs text-secondary-600 truncate">
                    {file.filename}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Suggested Prompts */}
        <div className="flex-1 p-6">
          <h2 className="mb-3 text-sm font-medium text-secondary-900">
            Suggested Prompts
          </h2>
          <div className="space-y-2">
            {suggestedPrompts.map((prompt, index) => (
              <button
                key={index}
                onClick={() => setInputValue(prompt)}
                className="w-full rounded-lg bg-secondary-50 p-3 text-left text-sm text-secondary-700 hover:bg-secondary-100 transition-colors"
              >
                <MessageSquare className="h-4 w-4 mb-1 text-secondary-400" />
                {prompt}
              </button>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="p-6 border-t border-secondary-200">
          <div className="space-y-2">
            <button
              onClick={() => navigate('/upload')}
              className="w-full btn-secondary text-sm"
            >
              Upload More Documents
            </button>
            
            {generatedSRS && (
              <button
                onClick={() => navigate('/editor', { state: { content: generatedSRS } })}
                className="w-full btn-primary text-sm"
              >
                <Edit3 className="mr-2 h-4 w-4" />
                Edit in Editor
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.length === 1 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center py-12"
              >
                <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-gradient-to-r from-primary-500 to-purple-600 text-white">
                  <Sparkles className="h-8 w-8" />
                </div>
                <h2 className="mb-2 text-2xl font-bold text-secondary-900">
                  AI SRS Generation
                </h2>
                <p className="text-secondary-600 max-w-2xl mx-auto">
                  Describe your project requirements and I'll help you create a comprehensive 
                  Software Requirements Specification document.
                </p>
              </motion.div>
            )}

            {messages.map((message, index) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex max-w-4xl space-x-3 ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                  {/* Avatar */}
                  <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
                    message.type === 'user' 
                      ? 'bg-primary-600 text-white' 
                      : 'bg-secondary-100 text-secondary-600'
                  }`}>
                    {message.type === 'user' ? (
                      <User className="h-4 w-4" />
                    ) : (
                      <Bot className="h-4 w-4" />
                    )}
                  </div>

                  {/* Message */}
                  <div className={`rounded-lg p-4 ${
                    message.type === 'user'
                      ? 'bg-primary-600 text-white'
                      : 'bg-white border border-secondary-200 text-secondary-900'
                  }`}>
                    {message.isLoading ? (
                      <div className="flex items-center space-x-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span>Generating SRS document...</span>
                      </div>
                    ) : (
                      <>
                        <div className="prose prose-sm max-w-none">
                          {message.content.split('\n').map((line, i) => (
                            <p key={i} className={message.type === 'user' ? 'text-white' : ''}>
                              {line}
                            </p>
                          ))}
                        </div>

                        {/* Actions for AI messages */}
                        {message.type === 'assistant' && !message.isLoading && (
                          <div className="mt-3 flex items-center space-x-2">
                            <button
                              onClick={() => copyToClipboard(message.content)}
                              className="text-secondary-400 hover:text-secondary-600 transition-colors"
                              title="Copy to clipboard"
                            >
                              <Copy className="h-4 w-4" />
                            </button>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
            
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="border-t border-secondary-200 bg-white p-6">
          <div className="max-w-4xl mx-auto">
            <form onSubmit={handleSubmit} className="flex space-x-4">
              <div className="flex-1">
                <textarea
                  ref={inputRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Describe your project requirements..."
                  className="input-field resize-none"
                  rows={3}
                  disabled={isGenerating}
                />
              </div>
              
              <button
                type="submit"
                disabled={!inputValue.trim() || isGenerating}
                className="btn-primary self-end"
              >
                {isGenerating ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </button>
            </form>

            <div className="mt-3 text-xs text-secondary-500 text-center">
              Press Enter to send, Shift+Enter for new line
            </div>
          </div>
        </div>
      </div>

      {/* Results Panel */}
      {generatedSRS && (
        <motion.div
          initial={{ opacity: 0, x: 100 }}
          animate={{ opacity: 1, x: 0 }}
          className="w-96 bg-white border-l border-secondary-200 flex flex-col"
        >
          {/* Header */}
          <div className="p-6 border-b border-secondary-200">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-secondary-900">
                Generated SRS
              </h2>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => copyToClipboard(generatedSRS)}
                  className="p-2 text-secondary-400 hover:text-secondary-600 hover:bg-secondary-100 rounded-lg transition-colors"
                  title="Copy SRS"
                >
                  <Copy className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6">
            <div className="prose prose-sm max-w-none">
              {generatedSRS.split('\n').map((line, index) => (
                <p key={index}>{line}</p>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div className="p-6 border-t border-secondary-200">
            <div className="space-y-3">
              <button
                onClick={() => navigate('/editor', { state: { content: generatedSRS } })}
                className="w-full btn-primary group"
              >
                <Edit3 className="mr-2 h-4 w-4" />
                Open in Editor
                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </button>
              
              <button
                onClick={() => {
                  // Create download link
                  const blob = new Blob([generatedSRS], { type: 'text/markdown' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = 'srs-document.md'
                  a.click()
                  URL.revokeObjectURL(url)
                  toast.success('SRS document downloaded!')
                }}
                className="w-full btn-secondary"
              >
                <Download className="mr-2 h-4 w-4" />
                Download as Markdown
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}

export default GeneratePage
