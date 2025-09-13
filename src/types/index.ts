export interface Document {
  id: string;
  filename: string;
  content: string;
  created_at: string;
  updated_at?: string;
  file_type: string;
  size: number;
}

export interface ChatMessage {
  content: string;
  user_input?: string;
  document_id?: string;
}

export interface ChatResponse {
  response: string;
  timestamp: string;
  document_id?: string;
}

export interface UploadResponse {
  status: string;
  document_id: string;
  filename: string;
  file_type: string;
  content_preview: string;
  size: number;
}

export interface ExportRequest {
  document_id: string;
  format: 'pdf' | 'docx' | 'md' | 'latex';
  content: string;
}

export interface User {
  id: string;
  name: string;
  avatar?: string;
  color: string;
}

export interface CollaborationSession {
  document_id: string;
  users: User[];
  active_cursors: Record<string, { x: number; y: number; user: User }>;
}

export interface WebSocketEvents {
  connect: () => void;
  disconnect: () => void;
  join_document: (data: { document_id: string; user_id: string }) => void;
  content_change: (data: { content: string }) => void;
  user_joined: (data: { user_id: string; document_id: string }) => void;
  content_updated: (data: { content: string; user_id: string; timestamp: string }) => void;
  document_state: (data: { content: string; document_id: string }) => void;
}

export interface HealthCheck {
  status: string;
  timestamp: string;
  ocr_available: boolean;
}

export type FileFormat = 'pdf' | 'docx' | 'pptx' | 'md' | 'latex' | 'txt' | 'json';

export interface ProcessingStatus {
  status: 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  message?: string;
}
