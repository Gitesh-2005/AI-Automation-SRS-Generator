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
  Square
} from 'lucide-react';
import {
  updateDocument,
  getDocument,
  exportDocument,
  downloadBlob
} from '../utils/api';
import {
  useDocumentCollaboration,
  generateUserId,
  generateUserColor,
  connectSocket,
  disconnectSocket
} from '../utils/socket';
import type { Document, ExportRequest } from '../types';

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

const EditorPage: React.FC = () => {
  const { documentId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

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
  const [inputPrompt, setInputPrompt] = useState('');

  // Streaming state
  const [streamingState, setStreamingState] = useState<StreamingState>({
    isStreaming: false,
    streamedContent: '',
    error: null
  });

  const editorRef = useRef<any>(null);
  const saveTimeoutRef = useRef<NodeJS.Timeout>();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  // Get initial content from location state or load document
  useEffect(() => {
    const initialContent = location.state?.content;
    if (initialContent) {
      setContent(initialContent);
      setHasUnsavedChanges(true);
    } else if (documentId) {
      loadDocument();
    }
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

    return () => {
      cleanup();
    };
  }, []);

  const setupWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const wsUrl = `ws://localhost:8000/stream-srs`;
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      console.log('WebSocket connected for streaming');
      setStreamingState(prev => ({ ...prev, error: null }));
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = undefined;
      }
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
      setStreamingState(prev => ({
        ...prev,
        isStreaming: false
      }));

      if (event.code !== 1000 && event.code !== 1001) {
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Attempting to reconnect WebSocket...');
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

  const cleanup = () => {
    if (wsRef.current) {
      wsRef.current.close(1000, 'Component unmounting');
      wsRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = undefined;
    }
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = undefined;
    }
  };

  // Auto-save functionality
  useEffect(() => {
    if (hasUnsavedChanges && documentId && !streamingState.isStreaming) {
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
  }, [hasUnsavedChanges, content, streamingState.isStreaming]);

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
    if (!documentId || !hasUnsavedChanges || streamingState.isStreaming) return;

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
    if (streamingState.isStreaming) {
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

  const handleExport = async (format: 'pdf' | 'docx' | 'md' | 'latex') => {
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

  const copyShareLink = () => {
    const shareUrl = `${window.location.origin}/editor/${documentId}`;
    navigator.clipboard.writeText(shareUrl);
    toast.success('Share link copied to clipboard!');
  };

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  const handleGenerateSRS = () => {
    if (!inputPrompt.trim() || streamingState.isStreaming) return;

    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      toast.error('Connection not available. Reconnecting...');
      setupWebSocket();
      return;
    }

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
      content: inputPrompt,
      document_id: documentId
    };

    try {
      wsRef.current.send(JSON.stringify(request));
      setInputPrompt('');
      toast.success('Starting SRS generation...');
    } catch (error) {
      console.error('Error sending WebSocket message:', error);
      setStreamingState(prev => ({
        ...prev,
        isStreaming: false,
        error: 'Failed to send request'
      }));
      toast.error('Failed to start SRS generation');
    }
  };

  const handleStopGeneration = () => {
    if (wsRef.current && streamingState.isStreaming) {
      wsRef.current.close();
      setStreamingState(prev => ({
        ...prev,
        isStreaming: false
      }));
      toast.success('Generation stopped');
      setTimeout(() => {
        setupWebSocket();
      }, 1000);
    }
  };

  const getConnectionStatus = () => {
    const wsConnected = wsRef.current?.readyState === WebSocket.OPEN;
    return {
      color: (isConnected && wsConnected) ? 'text-green-600' : 'text-red-600',
      bgColor: (isConnected && wsConnected) ? 'bg-green-500' : 'bg-red-500',
      text: (isConnected && wsConnected) ? 'Connected' : 'Offline'
    };
  };

  const connectionStatus = getConnectionStatus();

  return (
    <div className={`flex flex-col ${isFullscreen ? 'fixed inset-0 z-50 bg-white' : 'min-h-screen'}`}>
      {/* Header */}
      <div className="border-b border-secondary-200 bg-white px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <FileText className="h-5 w-5 text-secondary-600" />
              <h1 className="text-lg font-semibold text-secondary-900">
                {document?.filename || 'Untitled Document'}
              </h1>
              {hasUnsavedChanges && (
                <span className="text-xs text-orange-600">• Unsaved</span>
              )}
              {streamingState.isStreaming && (
                <div className="flex items-center space-x-1 text-xs text-blue-600">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  <span>Generating...</span>
                </div>
              )}
              {streamingState.error && (
                <div className="flex items-center space-x-1 text-xs text-red-600">
                  <AlertCircle className="h-3 w-3" />
                  <span>Error</span>
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
                disabled={!hasUnsavedChanges || isSaving || streamingState.isStreaming}
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
                    <button
                      onClick={() => handleExport('latex')}
                      className="block w-full px-4 py-2 text-left text-sm text-secondary-700 hover:bg-secondary-50"
                    >
                      LaTeX (.tex)
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* SRS Generation Input */}
      <div className="border-t border-secondary-200 bg-secondary-50 px-6 py-4">
        <div className="flex items-center space-x-4">
          <textarea
            value={inputPrompt}
            onChange={(e) => setInputPrompt(e.target.value)}
            placeholder="Enter requirements to generate SRS..."
            className="input-field flex-1 resize-none rounded-md border border-secondary-300 px-3 py-2"
            rows={2}
            disabled={streamingState.isStreaming}
          />
          <div className="flex space-x-2">
            <button
              onClick={handleGenerateSRS}
              disabled={!inputPrompt.trim() || streamingState.isStreaming}
              className="btn-primary flex items-center space-x-1"
            >
              {streamingState.isStreaming ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <CheckCircle className="h-4 w-4" />
              )}
              <span>Generate SRS</span>
            </button>
            {streamingState.isStreaming && (
              <button
                onClick={handleStopGeneration}
                className="btn-ghost flex items-center space-x-1 text-red-600 hover:text-red-800"
              >
                <Square className="h-4 w-4" />
                <span>Stop</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Editor Content */}
      <div className="flex-1 flex">
        {/* Editor Panel */}
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
                disabled={streamingState.isStreaming}
              />
            )}
          </div>
        </div>

        {/* Preview Panel */}
        {showPreview && (
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
          </div>

          <div className="flex items-center space-x-4">
            {hasUnsavedChanges && (
              <span className="text-orange-600">Unsaved changes</span>
            )}
            {streamingState.error && (
              <span className="text-red-600">{streamingState.error}</span>
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

export default EditorPage;