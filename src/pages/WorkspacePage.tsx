import React, { useState, useEffect, useRef } from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import { CKEditor } from '@ckeditor/ckeditor5-react';
import ClassicEditor from '@ckeditor/ckeditor5-build-classic';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import {
  Save,
  Download,
  Users,
  Share2,
  FileText,
  Loader2,
  Eye,
  EyeOff,
  Maximize2,
  Minimize2,
  Copy,
  AlertCircle,
  CheckCircle,
  Square,
  Send,
  Bot,
  User,
  Edit3,
  ArrowRight,
  Sparkles,
  MessageSquare,
  Wand2,
  Code,
  FileEdit,
  Settings
} from 'lucide-react';
import {
  updateDocument,
  getDocument,
  exportDocument,
  downloadBlob,
  sendChatMessage
} from '../utils/api';
import {
  useDocumentCollaboration,
  generateUserId,
  generateUserColor,
  connectSocket,
  disconnectSocket
} from '../utils/socket';
import type { Document, ExportRequest, ChatMessage, ChatResponse, UploadResponse } from '../types';

interface Collaborator {
  id: string;
  name: string;
  color: string;
  lastSeen: Date;
}

interface StreamingState {
  isStreaming: boolean;
  streamedContent: string;
  error: string | null;
}

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isLoading?: boolean;
}

interface AIEditState {
  isEditing: boolean;
  instructions: string;
  editedContent: string;
  error: string | null;
}

const WorkspacePage: React.FC = () => {
  const { documentId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  // Workspace state
  const [activeTab, setActiveTab] = useState<'editor' | 'generator' | 'ai-edit'>('editor');

  // Editor state
  const [content, setContent] = useState<string>('');
  const [document, setDocument] = useState<Document | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  // Collaboration state
  const [collaborators, setCollaborators] = useState<Collaborator[]>([]);
  const [userId] = useState(() => generateUserId());
  const [userColor] = useState(() => generateUserColor());
  const [isConnected, setIsConnected] = useState(false);

  // UI state
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  // Generator state
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedSRS, setGeneratedSRS] = useState<string>('');

  // Streaming state
  const [streamingState, setStreamingState] = useState<StreamingState>({
    isStreaming: false,
    streamedContent: '',
    error: null
  });

  // AI Edit state
  const [aiEditState, setAIEditState] = useState<AIEditState>({
    isEditing: false,
    instructions: '',
    editedContent: '',
    error: null
  });

  const editorRef = useRef<any>(null);
  const saveTimeoutRef = useRef<NodeJS.Timeout>();
  const wsRef = useRef<WebSocket | null>(null);
  const aiEditWsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Get uploaded files from location state
  const uploadedFiles: UploadResponse[] = location.state?.uploadedFiles || [];

  // Initialize
  useEffect(() => {
    const initialContent = location.state?.content;
    if (initialContent) {
      setContent(initialContent);
      setHasUnsavedChanges(true);
    } else if (documentId) {
      loadDocument();
    }

    // Initialize generator messages
    const welcomeMessage: Message = {
      id: '1',
      type: 'assistant',
      content: uploadedFiles.length > 0
        ? `Hello! I've analyzed your ${uploadedFiles.length} uploaded document(s). I'm ready to help you generate a comprehensive SRS document. What specific requirements would you like me to focus on?`
        : `Hello! I'm your AI assistant for SRS generation. Describe your project requirements, and I'll help you create a professional Software Requirements Specification document.`,
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
  }, [documentId, location.state]);

  // Setup real-time collaboration
  useEffect(() => {
    if (documentId) {
      connectSocket();
      setIsConnected(true);

      const collaboration = useDocumentCollaboration(documentId, userId);
      collaboration.joinDocument();

      collaboration.onUserJoined((data) => {
        const newCollaborator: Collaborator = {
          id: data.user_id,
          name: `User ${data.user_id.slice(-4)}`,
          color: generateUserColor(),
          lastSeen: new Date()
        };

        setCollaborators(prev => {
          const exists = prev.find(c => c.id === data.user_id);
          return exists ? prev : [...prev, newCollaborator];
        });

        toast.success(`${newCollaborator.name} joined the document`);
      });

      collaboration.onContentUpdated((data) => {
        if (data.user_id !== userId) {
          setContent(data.content);
          if (editorRef.current) {
            editorRef.current.setData(data.content);
          }
        }
      });

      collaboration.onDocumentState((data) => {
        setContent(data.content);
        if (editorRef.current) {
          editorRef.current.setData(data.content);
        }
      });

      return () => {
        disconnectSocket();
        setIsConnected(false);
      };
    }
  }, [documentId, userId]);

  // Setup WebSocket for streaming SRS generation
  useEffect(() => {
    setupWebSocket();
    setupAIEditWebSocket();

    return () => {
      cleanup();
    };
  }, []);

  // Auto-save functionality
  useEffect(() => {
    if (hasUnsavedChanges && documentId && !streamingState.isStreaming && !aiEditState.isEditing) {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }

      saveTimeoutRef.current = setTimeout(() => {
        handleSave();
      }, 3000);
    }

    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [hasUnsavedChanges, content, streamingState.isStreaming, aiEditState.isEditing]);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const setupWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const wsUrl = `ws://localhost:8000/stream-srs`;
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      console.log('WebSocket connected for streaming');
      setStreamingState(prev => ({ ...prev, error: null }));
    };

    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.chunk) {
          setStreamingState(prev => ({
            ...prev,
            streamedContent: prev.streamedContent + data.chunk
          }));
          setContent(prev => prev + data.chunk);
          if (editorRef.current) {
            const currentData = editorRef.current.getData();
            editorRef.current.setData(currentData + data.chunk);
          }
          setHasUnsavedChanges(true);
        } else if (data.final) {
          const finalContent = data.final.final_srs;
          setContent(finalContent);
          if (editorRef.current) {
            editorRef.current.setData(finalContent);
          }
          setHasUnsavedChanges(true);
          setStreamingState(prev => ({
            ...prev,
            isStreaming: false,
            streamedContent: finalContent
          }));
          toast.success('SRS generation complete!');
        } else if (data.error) {
          setStreamingState(prev => ({
            ...prev,
            isStreaming: false,
            error: data.error
          }));
          toast.error(data.error);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
        toast.error('Error processing streaming response');
      }
    };

    wsRef.current.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason);
      setStreamingState(prev => ({ ...prev, isStreaming: false }));

      if (event.code !== 1000 && event.code !== 1001) {
        reconnectTimeoutRef.current = setTimeout(() => {
          setupWebSocket();
        }, 3000);
      }
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setStreamingState(prev => ({
        ...prev,
        isStreaming: false,
        error: 'WebSocket connection error'
      }));
      toast.error('Connection error occurred');
    };
  };

  const setupAIEditWebSocket = () => {
    if (aiEditWsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const wsUrl = `ws://localhost:8000/stream-ai-edit`;
    aiEditWsRef.current = new WebSocket(wsUrl);

    aiEditWsRef.current.onopen = () => {
      console.log('AI Edit WebSocket connected');
      setAIEditState(prev => ({ ...prev, error: null }));
    };

    aiEditWsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.chunk) {
          setAIEditState(prev => ({
            ...prev,
            editedContent: prev.editedContent + data.chunk
          }));
        } else if (data.final) {
          const editedContent = data.final.edited_content;
          setContent(editedContent);
          if (editorRef.current) {
            editorRef.current.setData(editedContent);
          }
          setHasUnsavedChanges(true);
          setAIEditState(prev => ({
            ...prev,
            isEditing: false,
            editedContent: editedContent
          }));
          toast.success('AI editing complete!');
          setActiveTab('editor'); // Switch back to editor
        } else if (data.error) {
          setAIEditState(prev => ({
            ...prev,
            isEditing: false,
            error: data.error
          }));
          toast.error(data.error);
        }
      } catch (error) {
        console.error('Error parsing AI Edit WebSocket message:', error);
        toast.error('Error processing AI edit response');
      }
    };

    aiEditWsRef.current.onclose = (event) => {
      console.log('AI Edit WebSocket disconnected');
      setAIEditState(prev => ({ ...prev, isEditing: false }));
    };

    aiEditWsRef.current.onerror = (error) => {
      console.error('AI Edit WebSocket error:', error);
      setAIEditState(prev => ({
        ...prev,
        isEditing: false,
        error: 'AI Edit connection error'
      }));
      toast.error('AI Edit connection error occurred');
    };
  };

  const cleanup = () => {
    if (wsRef.current) {
      wsRef.current.close(1000, 'Component unmounting');
      wsRef.current = null;
    }
    if (aiEditWsRef.current) {
      aiEditWsRef.current.close(1000, 'Component unmounting');
      aiEditWsRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }
  };

  const loadDocument = async () => {
    if (!documentId) return;

    setIsLoading(true);
    try {
      const doc = await getDocument(documentId);
      setDocument(doc);
      setContent(doc.content);
    } catch (error) {
      toast.error('Failed to load document');
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSave = async () => {
    if (!documentId || !hasUnsavedChanges || streamingState.isStreaming || aiEditState.isEditing) return;

    setIsSaving(true);
    try {
      await updateDocument(documentId, content);
      setHasUnsavedChanges(false);
      setLastSaved(new Date());
      toast.success('Document saved successfully');
    } catch (error) {
      toast.error('Failed to save document');
      console.error(error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleEditorChange = (event: any, editor: any) => {
    if (streamingState.isStreaming || aiEditState.isEditing) {
      return;
    }

    const data = editor.getData();
    setContent(data);
    setHasUnsavedChanges(true);

    if (documentId && isConnected) {
      const collaboration = useDocumentCollaboration(documentId, userId);
      collaboration.sendContentChange(data);
    }
  };

  const handleExport = async (format: 'pdf' | 'docx' | 'md') => {
    if (!documentId) {
      const blob = new Blob([content], {
        type: format === 'md' ? 'text/markdown' : 'text/plain'
      });
      downloadBlob(blob, `srs-document.${format}`);
      toast.success('Document downloaded!');
      return;
    }

    setIsExporting(true);
    try {
      const exportRequest: Omit<ExportRequest, 'document_id'> = {
        format,
        content
      };

      const blob = await exportDocument(documentId, exportRequest);
      const filename = `${document?.filename || 'srs-document'}.${format}`;
      downloadBlob(blob, filename);
      toast.success('Document exported successfully!');
    } catch (error) {
      toast.error('Failed to export document');
      console.error(error);
    } finally {
      setIsExporting(false);
    }
  };

  const handleGenerateSRS = async () => {
    if (!inputValue.trim() || isGenerating) return;

    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      toast.error('Connection not available. Reconnecting...');
      setupWebSocket();
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    const loadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: 'assistant',
      content: 'Generating your SRS document...',
      timestamp: new Date(),
      isLoading: true
    };

    setMessages(prev => [...prev, userMessage, loadingMessage]);
    setInputValue('');
    setIsGenerating(true);

    setStreamingState({
      isStreaming: true,
      streamedContent: '',
      error: null
    });

    setContent('');
    if (editorRef.current) {
      editorRef.current.setData('');
    }

    const request = {
      content: inputValue,
      document_id: documentId
    };

    try {
      wsRef.current.send(JSON.stringify(request));
      toast.success('Starting SRS generation...');
    } catch (error) {
      console.error('Error sending WebSocket message:', error);
      setStreamingState(prev => ({
        ...prev,
        isStreaming: false,
        error: 'Failed to send request'
      }));
      toast.error('Failed to start SRS generation');
      setIsGenerating(false);
    }
  };

  const handleAIEdit = async () => {
    if (!aiEditState.instructions.trim() || aiEditState.isEditing) return;

    if (!aiEditWsRef.current || aiEditWsRef.current.readyState !== WebSocket.OPEN) {
      toast.error('AI Edit connection not available. Reconnecting...');
      setupAIEditWebSocket();
      return;
    }

    setAIEditState(prev => ({
      ...prev,
      isEditing: true,
      editedContent: '',
      error: null
    }));

    const request = {
      current_content: content,
      edit_instructions: aiEditState.instructions
    };

    try {
      aiEditWsRef.current.send(JSON.stringify(request));
      toast.success('Starting AI editing...');
    } catch (error) {
      console.error('Error sending AI Edit WebSocket message:', error);
      setAIEditState(prev => ({
        ...prev,
        isEditing: false,
        error: 'Failed to send edit request'
      }));
      toast.error('Failed to start AI editing');
    }
  };

  const copyShareLink = () => {
    const shareUrl = `${window.location.origin}/workspace/${documentId}`;
    navigator.clipboard.writeText(shareUrl);
    toast.success('Share link copied to clipboard!');
  };

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  const copyToClipboard = (content: string) => {
    navigator.clipboard.writeText(content);
    toast.success('Content copied to clipboard!');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleGenerateSRS();
    }
  };

  const getConnectionStatus = () => {
    const wsConnected = wsRef.current?.readyState === WebSocket.OPEN;
    const aiEditWsConnected = aiEditWsRef.current?.readyState === WebSocket.OPEN;
    const allConnected = isConnected && wsConnected && aiEditWsConnected;
    
    return {
      color: allConnected ? 'text-green-600' : 'text-red-600',
      bgColor: allConnected ? 'bg-green-500' : 'bg-red-500',
      text: allConnected ? 'Connected' : 'Offline'
    };
  };

  const connectionStatus = getConnectionStatus();

  const suggestedPrompts = [
    "Create a complete SRS for a web application",
    "Focus on functional requirements and user stories",
    "Generate system architecture and technical specifications",
    "Include security and performance requirements",
    "Create a mobile app SRS with UI/UX requirements"
  ];

  return (
    <div className={`flex flex-col ${isFullscreen ? 'fixed inset-0 z-50 bg-white' : 'min-h-screen'}`}>
      {/* Header */}
      <div className="border-b border-secondary-200 bg-white px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <FileEdit className="h-5 w-5 text-secondary-600" />
              <h1 className="text-lg font-semibold text-secondary-900">
                {document?.filename || 'AI SRS Workspace'}
              </h1>
              {hasUnsavedChanges && (
                <span className="text-xs text-orange-600">• Unsaved</span>
              )}
              {(streamingState.isStreaming || aiEditState.isEditing) && (
                <div className="flex items-center space-x-1 text-xs text-blue-600">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  <span>{streamingState.isStreaming ? 'Generating...' : 'Editing...'}</span>
                </div>
              )}
            </div>

            {lastSaved && (
              <span className="text-sm text-secondary-500">
                Last saved: {lastSaved.toLocaleTimeString()}
              </span>
            )}
          </div>

          <div className="flex items-center space-x-2">
            {/* Collaborators */}
            {collaborators.length > 0 && (
              <div className="flex items-center space-x-1">
                <Users className="h-4 w-4 text-secondary-500" />
                <div className="flex -space-x-1">
                  {collaborators.slice(0, 3).map((collaborator) => (
                    <div
                      key={collaborator.id}
                      className="h-6 w-6 rounded-full border-2 border-white flex items-center justify-center text-xs font-medium text-white"
                      style={{ backgroundColor: collaborator.color }}
                      title={collaborator.name}
                    >
                      {collaborator.name.charAt(0)}
                    </div>
                  ))}
                  {collaborators.length > 3 && (
                    <div className="h-6 w-6 rounded-full border-2 border-white bg-secondary-500 flex items-center justify-center text-xs font-medium text-white">
                      +{collaborators.length - 3}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Connection status */}
            <div className={`flex items-center space-x-1 text-xs ${connectionStatus.color}`}>
              <div className={`h-2 w-2 rounded-full ${connectionStatus.bgColor}`} />
              <span>{connectionStatus.text}</span>
            </div>

            {/* Actions */}
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setShowPreview(!showPreview)}
                className="btn-ghost"
                title="Toggle preview"
              >
                {showPreview ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>

              <button
                onClick={toggleFullscreen}
                className="btn-ghost"
                title="Toggle fullscreen"
              >
                {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
              </button>

              {documentId && (
                <button
                  onClick={copyShareLink}
                  className="btn-ghost"
                  title="Copy share link"
                >
                  <Share2 className="h-4 w-4" />
                </button>
              )}

              <button
                onClick={handleSave}
                disabled={!hasUnsavedChanges || isSaving || streamingState.isStreaming || aiEditState.isEditing}
                className="btn-secondary flex items-center space-x-1"
              >
                {isSaving ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Save className="h-4 w-4" />
                )}
                <span>Save</span>
              </button>

              {/* Export dropdown */}
              <div className="relative group">
                <button
                  className="btn-primary flex items-center space-x-1"
                  disabled={isExporting}
                >
                  {isExporting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Download className="h-4 w-4" />
                  )}
                  <span>Export</span>
                </button>

                <div className="absolute right-0 top-full mt-1 w-40 rounded-lg bg-white shadow-lg ring-1 ring-secondary-200 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                  <div className="py-1">
                    <button
                      onClick={() => handleExport('md')}
                      className="block w-full px-4 py-2 text-left text-sm text-secondary-700 hover:bg-secondary-50"
                    >
                      Markdown (.md)
                    </button>
                    <button
                      onClick={() => handleExport('pdf')}
                      className="block w-full px-4 py-2 text-left text-sm text-secondary-700 hover:bg-secondary-50"
                    >
                      PDF (.pdf)
                    </button>
                    <button
                      onClick={() => handleExport('docx')}
                      className="block w-full px-4 py-2 text-left text-sm text-secondary-700 hover:bg-secondary-50"
                    >
                      Word (.docx)
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="mt-4 border-t border-secondary-200 pt-4">
          <nav className="flex space-x-8">
            <button
              onClick={() => setActiveTab('editor')}
              className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'editor'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-secondary-500 hover:text-secondary-700 hover:border-secondary-300'
              }`}
            >
              <Edit3 className="h-4 w-4" />
              <span>Editor</span>
            </button>
            <button
              onClick={() => setActiveTab('generator')}
              className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'generator'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-secondary-500 hover:text-secondary-700 hover:border-secondary-300'
              }`}
            >
              <Bot className="h-4 w-4" />
              <span>AI Generator</span>
            </button>
            <button
              onClick={() => setActiveTab('ai-edit')}
              className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'ai-edit'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-secondary-500 hover:text-secondary-700 hover:border-secondary-300'
              }`}
            >
              <Wand2 className="h-4 w-4" />
              <span>AI Edit</span>
            </button>
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Editor Tab */}
        {activeTab === 'editor' && (
          <div className={`${showPreview ? 'w-1/2' : 'w-full'} flex flex-col`}>
            <div className="flex-1 p-6">
              {isLoading ? (
                <div className="flex items-center justify-center h-96">
                  <div className="text-center">
                    <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary-600" />
                    <p className="text-secondary-600">Loading document...</p>
                  </div>
                </div>
              ) : (
                <CKEditor
                  editor={ClassicEditor}
                  data={content}
                  onChange={handleEditorChange}
                  onReady={(editor) => {
                    editorRef.current = editor;
                    editor.editing.view.change((writer) => {
                      writer.setStyle('min-height', '600px', editor.editing.view.document.getRoot());
                    });
                  }}
                  config={{
                    placeholder: 'Start writing your SRS document...',
                    toolbar: [
                      'heading', '|',
                      'bold', 'italic', 'underline', '|',
                      'bulletedList', 'numberedList', '|',
                      'outdent', 'indent', '|',
                      'blockQuote', 'insertTable', '|',
                      'undo', 'redo'
                    ],
                    heading: {
                      options: [
                        { model: 'paragraph', title: 'Paragraph', class: 'ck-heading_paragraph' },
                        { model: 'heading1', view: 'h1', title: 'Heading 1', class: 'ck-heading_heading1' },
                        { model: 'heading2', view: 'h2', title: 'Heading 2', class: 'ck-heading_heading2' },
                        { model: 'heading3', view: 'h3', title: 'Heading 3', class: 'ck-heading_heading3' }
                      ]
                    },
                    table: {
                      contentToolbar: ['tableColumn', 'tableRow', 'mergeTableCells']
                    }
                  }}
                  disabled={streamingState.isStreaming || aiEditState.isEditing}
                />
              )}
            </div>
          </div>
        )}

        {/* Generator Tab */}
        {activeTab === 'generator' && (
          <div className="flex flex-1">
            {/* Sidebar */}
            <div className="w-80 bg-white border-r border-secondary-200 flex flex-col">
              {/* Header */}
              <div className="p-6 border-b border-secondary-200">
                <div className="flex items-center space-x-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary-500 to-purple-600 text-white">
                    <Bot className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-secondary-900">
                      AI SRS Generator
                    </h2>
                    <p className="text-sm text-secondary-500">
                      Intelligent Assistant
                    </p>
                  </div>
                </div>
              </div>

              {/* Uploaded Files */}
              {uploadedFiles.length > 0 && (
                <div className="p-6 border-b border-secondary-200">
                  <h3 className="mb-3 text-sm font-medium text-secondary-900">
                    Uploaded Documents
                  </h3>
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
                <h3 className="mb-3 text-sm font-medium text-secondary-900">
                  Suggested Prompts
                </h3>
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
                  <div className="flex space-x-4">
                    <div className="flex-1">
                      <textarea
                        ref={inputRef}
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Describe your project requirements..."
                        className="input-field resize-none"
                        rows={3}
                        disabled={isGenerating || streamingState.isStreaming}
                      />
                    </div>

                    <button
                      onClick={handleGenerateSRS}
                      disabled={!inputValue.trim() || isGenerating || streamingState.isStreaming}
                      className="btn-primary self-end"
                    >
                      {isGenerating || streamingState.isStreaming ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Send className="h-4 w-4" />
                      )}
                    </button>
                  </div>

                  <div className="mt-3 text-xs text-secondary-500 text-center">
                    Press Enter to send, Shift+Enter for new line
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* AI Edit Tab */}
        {activeTab === 'ai-edit' && (
          <div className="flex-1 flex flex-col">
            <div className="p-6 border-b border-secondary-200 bg-secondary-50">
              <div className="max-w-4xl mx-auto">
                <div className="flex items-center space-x-3 mb-4">
                  <Wand2 className="h-6 w-6 text-primary-600" />
                  <h2 className="text-lg font-semibold text-secondary-900">AI Content Editor</h2>
                </div>
                <p className="text-sm text-secondary-600 mb-6">
                  Provide instructions for how you want to modify your current document content. The AI will edit it according to your specifications.
                </p>

                <div className="flex space-x-4">
                  <div className="flex-1">
                    <textarea
                      value={aiEditState.instructions}
                      onChange={(e) => setAIEditState(prev => ({ ...prev, instructions: e.target.value }))}
                      placeholder="Enter your editing instructions... (e.g., 'Add more details to the security requirements section', 'Rewrite the introduction to be more formal', 'Include user acceptance criteria')"
                      className="input-field resize-none"
                      rows={3}
                      disabled={aiEditState.isEditing}
                    />
                  </div>

                  <button
                    onClick={handleAIEdit}
                    disabled={!aiEditState.instructions.trim() || aiEditState.isEditing || !content.trim()}
                    className="btn-primary self-end flex items-center space-x-2"
                  >
                    {aiEditState.isEditing ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Wand2 className="h-4 w-4" />
                    )}
                    <span>Edit with AI</span>
                  </button>
                </div>

                {aiEditState.error && (
                  <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <AlertCircle className="h-4 w-4 text-red-600" />
                      <span className="text-sm text-red-600">{aiEditState.error}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Current content preview */}
            <div className="flex-1 p-6">
              <div className="max-w-4xl mx-auto">
                <h3 className="text-sm font-medium text-secondary-900 mb-4">Current Content</h3>
                <div className="bg-white border border-secondary-200 rounded-lg p-6 max-h-96 overflow-y-auto">
                  {content ? (
                    <div className="prose prose-sm max-w-none text-secondary-700">
                      {content.split('\n').slice(0, 50).map((line, index) => (
                        <p key={index}>{line}</p>
                      ))}
                      {content.split('\n').length > 50 && (
                        <p className="text-secondary-500 italic">... and {content.split('\n').length - 50} more lines</p>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-12 text-secondary-500">
                      <FileText className="h-12 w-12 mx-auto mb-4 text-secondary-300" />
                      <p>No content to edit yet. Generate some content first using the AI Generator tab.</p>
                    </div>
                  )}
                </div>

                {aiEditState.isEditing && (
                  <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                      <span className="text-sm text-blue-600">AI is editing your content...</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Preview Panel */}
        {showPreview && activeTab === 'editor' && (
          <div className="w-1/2 border-l border-secondary-200 flex flex-col">
            <div className="p-4 border-b border-secondary-200 bg-secondary-50">
              <div className="flex items-center justify-between">
                <h3 className="font-medium text-secondary-900">Preview</h3>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(content);
                    toast.success('Content copied!');
                  }}
                  className="text-secondary-500 hover:text-secondary-700"
                  title="Copy content"
                >
                  <Copy className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div className="flex-1 p-6 overflow-y-auto">
              <div
                className="prose prose-sm max-w-none"
                dangerouslySetInnerHTML={{ __html: content }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-secondary-200 bg-secondary-50 px-6 py-3">
        <div className="flex items-center justify-between text-sm text-secondary-600">
          <div className="flex items-center space-x-4">
            <span>Words: {content.replace(/<[^>]*>/g, '').split(/\s+/).filter(Boolean).length}</span>
            <span>Characters: {content.replace(/<[^>]*>/g, '').length}</span>
            <span>Active: {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}</span>
          </div>

          <div className="flex items-center space-x-4">
            {hasUnsavedChanges && (
              <span className="text-orange-600">Unsaved changes</span>
            )}
            {(streamingState.error || aiEditState.error) && (
              <span className="text-red-600">{streamingState.error || aiEditState.error}</span>
            )}
            <button
              onClick={() => navigate('/')}
              className="hover:text-secondary-900"
            >
              ← Back to Home
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WorkspacePage;
