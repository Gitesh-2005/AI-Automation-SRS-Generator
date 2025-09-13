// socket.ts
import { io, Socket } from 'socket.io-client'
import type { WebSocketEvents } from '../types'

class SocketManager {
  private socket: Socket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectInterval = 1000

  connect(): Socket {
    if (this.socket?.connected) {
      return this.socket
    }

    this.socket = io(import.meta.env.VITE_SOCKET_URL || 'http://localhost:8000', {
      transports: ['websocket', 'polling'],
      autoConnect: true,
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectInterval,
    })

    this.setupEventListeners()
    return this.socket
  }

  private setupEventListeners() {
    if (!this.socket) return

    this.socket.on('connect', () => {
      console.log('[Socket] Connected to server')
      this.reconnectAttempts = 0
    })

    this.socket.on('disconnect', (reason) => {
      console.log('[Socket] Disconnected:', reason)
    })

    this.socket.on('connect_error', (error) => {
      console.error('[Socket] Connection error:', error)
      this.reconnectAttempts++
      
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.error('[Socket] Max reconnection attempts reached')
      }
    })

    this.socket.on('reconnect', (attemptNumber) => {
      console.log(`[Socket] Reconnected after ${attemptNumber} attempts`)
      this.reconnectAttempts = 0
    })

    this.socket.on('reconnect_failed', () => {
      console.error('[Socket] Failed to reconnect to server')
    })
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
    }
  }

  emit(event: string, data: any) {
    if (this.socket?.connected) {
      this.socket.emit(event, data)
    } else {
      console.warn('[Socket] Cannot emit - not connected')
    }
  }

  on(event: string, callback: (data: any) => void) {
    if (this.socket) {
      this.socket.on(event, callback)
    }
  }

  off(event: string, callback?: (data: any) => void) {
    if (this.socket) {
      this.socket.off(event, callback)
    }
  }

  // Document collaboration methods
  joinDocument(documentId: string, userId: string) {
    this.emit('join_document', { document_id: documentId, user_id: userId })
  }

  sendContentChange(content: string) {
    this.emit('content_change', { content })
  }

  onUserJoined(callback: (data: { user_id: string; document_id: string }) => void) {
    this.on('user_joined', callback)
  }

  onContentUpdated(callback: (data: { content: string; user_id: string; timestamp: string }) => void) {
    this.on('content_updated', callback)
  }

  onDocumentState(callback: (data: { content: string; document_id: string }) => void) {
    this.on('document_state', callback)
  }

  isConnected(): boolean {
    return this.socket?.connected || false
  }

  getSocket(): Socket | null {
    return this.socket
  }
}

// Create singleton instance
const socketManager = new SocketManager()

export default socketManager

// Export convenience functions
export const connectSocket = () => socketManager.connect()
export const disconnectSocket = () => socketManager.disconnect()
export const emitEvent = (event: string, data: any) => socketManager.emit(event, data)
export const onEvent = (event: string, callback: (data: any) => void) => socketManager.on(event, callback)
export const offEvent = (event: string, callback?: (data: any) => void) => socketManager.off(event, callback)

// Document collaboration helpers
export const useDocumentCollaboration = (documentId: string, userId: string) => {
  const joinDocument = () => socketManager.joinDocument(documentId, userId)
  const sendContentChange = (content: string) => socketManager.sendContentChange(content)
  const onUserJoined = (callback: (data: { user_id: string; document_id: string }) => void) => 
    socketManager.onUserJoined(callback)
  const onContentUpdated = (callback: (data: { content: string; user_id: string; timestamp: string }) => void) => 
    socketManager.onContentUpdated(callback)
  const onDocumentState = (callback: (data: { content: string; document_id: string }) => void) => 
    socketManager.onDocumentState(callback)

  return {
    joinDocument,
    sendContentChange,
    onUserJoined,
    onContentUpdated,
    onDocumentState,
    isConnected: () => socketManager.isConnected(),
  }
}

// Generate random user ID for demo purposes
export const generateUserId = (): string => {
  return `user_${Math.random().toString(36).substr(2, 8)}`
}

// Generate random color for user avatars
export const generateUserColor = (): string => {
  const colors = [
    '#ef4444', // red
    '#f97316', // orange
    '#eab308', // yellow
    '#22c55e', // green
    '#06b6d4', // cyan
    '#3b82f6', // blue
    '#8b5cf6', // violet
    '#ec4899', // pink
  ]
  
  return colors[Math.floor(Math.random() * colors.length)]
}