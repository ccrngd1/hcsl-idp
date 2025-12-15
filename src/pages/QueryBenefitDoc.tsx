import { useState, useEffect } from 'react'
import {
  Container,
  Header,
  SpaceBetween,
  Button,
  Box,
  Alert,
  FormField,
  FileUpload,
  Grid,
  Badge,
  Input,
  Select
} from '@cloudscape-design/components'
import { bedrockService } from '../services/api'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
}

interface UploadedFile {
  file: File
  name: string
  size: number
  uploadedAt: string
}

export default function QueryBenefitDoc() {
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null)
  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [currentMessage, setCurrentMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState({ 
    label: 'Claude 3.7 Sonnet', 
    value: 'us.anthropic.claude-3-7-sonnet-20250219-v1:0' 
  })


  const modelOptions = [
    { label: 'Claude 3.7 Sonnet', value: 'us.anthropic.claude-3-7-sonnet-20250219-v1:0' },
    { label: 'Claude 4 Sonnet', value: 'us.anthropic.claude-sonnet-4-20250514-v1:0' },
    { label: 'Claude 4.5 Sonnet', value: 'us.anthropic.claude-sonnet-4-5-20250929-v1:0' },
    { label: 'Nova 2 Pro', value: 'us.amazon.nova-2-pro-v1:0' },
    { label: 'Nova 2 Premier', value: 'us.amazon.nova-2-premier-v1:0' }
  ]

  const handleFileUpload = (files: File[]) => {
    if (files.length > 0) {
      const file = files[0]
      if (file.type !== 'application/pdf') {
        alert('Please upload a PDF file only.')
        return
      }

      // Check file size (limit to 10MB)
      if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB.')
        return
      }

      const uploadedFile: UploadedFile = {
        file,
        name: file.name,
        size: file.size,
        uploadedAt: new Date().toISOString()
      }

      setUploadedFile(uploadedFile)
      
      // Clean up previous PDF URL
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl)
      }
      
      // Create URL for PDF viewer
      const url = URL.createObjectURL(file)
      setPdfUrl(url)
      
      // Clear previous chat messages when new file is uploaded
      setChatMessages([])
    }
  }

  const handleSendMessage = async () => {
    if (!currentMessage.trim() || !uploadedFile || loading) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: currentMessage,
      timestamp: new Date().toISOString()
    }

    setChatMessages(prev => [...prev, userMessage])
    setCurrentMessage('')
    setLoading(true)
    setError(null)

    try {
      // Prepare chat history for API
      const chatHistory = chatMessages.map(msg => ({
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp
      }))

      const response = await bedrockService.chatWithDocument(
        uploadedFile.file,
        currentMessage,
        chatHistory,
        selectedModel.value,
        { temperature: 0.7 }
      )

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.response,
        timestamp: new Date().toISOString()
      }

      setChatMessages(prev => [...prev, assistantMessage])
    } catch (error: any) {
      console.error('Chat error:', error)
      const errorMsg = error?.response?.data?.detail || error?.message || 'An unexpected error occurred'
      setError(errorMsg)
      
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Sorry, I encountered an error: ${errorMsg}. Please try again.`,
        timestamp: new Date().toISOString()
      }
      setChatMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (event: any) => {
    if (event.detail.key === 'Enter' && !event.detail.shiftKey) {
      event.preventDefault()
      handleSendMessage()
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  // Cleanup PDF URL on component unmount
  useEffect(() => {
    return () => {
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl)
      }
    }
  }, [pdfUrl])

  return (
    <SpaceBetween size="l">
      <Header
        variant="h1"
        description="Upload a PDF document and chat with it using AI"
      >
        Chat with PDF Documents
      </Header>

      <Alert type="info">
        Upload a PDF document and ask questions about its content. The AI will analyze the document and provide contextual answers based on the content.
      </Alert>

      {error && (
        <Alert type="error" dismissible onDismiss={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* File Upload Section */}
      <Container>
        <SpaceBetween size="m">
          <Header variant="h2">Upload Document</Header>
          
          <FormField
            label="PDF Document"
            description="Upload a PDF file to analyze and chat with"
          >
            <FileUpload
              onChange={({ detail }) => handleFileUpload(detail.value)}
              value={uploadedFile ? [uploadedFile.file] : []}
              i18nStrings={{
                uploadButtonText: e => e ? "Choose files" : "Choose file",
                dropzoneText: e => e ? "Drop files to upload" : "Drop file to upload",
                removeFileAriaLabel: e => `Remove file ${e + 1}`,
                limitShowFewer: "Show fewer files",
                limitShowMore: "Show more files",
                errorIconAriaLabel: "Error"
              }}
              showFileLastModified
              showFileSize
              showFileThumbnail
              tokenLimit={3}
              accept=".pdf"
            />
          </FormField>

          {uploadedFile && (
            <Box>
              <Badge color="green">
                File uploaded: {uploadedFile.name} ({formatFileSize(uploadedFile.size)})
              </Badge>
            </Box>
          )}

          <FormField label="AI Model">
            <Select
              selectedOption={selectedModel}
              onChange={({ detail }) => setSelectedModel(detail.selectedOption as { label: string; value: string })}
              options={modelOptions}
              placeholder="Choose AI model"
            />
          </FormField>
        </SpaceBetween>
      </Container>

      {uploadedFile && (
        <Grid gridDefinition={[{ colspan: 6 }, { colspan: 6 }]}>
          {/* PDF Viewer */}
          <Container>
            <Header variant="h2">Document Viewer</Header>
            {pdfUrl && (
              <Box>
                <iframe
                  src={pdfUrl}
                  width="100%"
                  height="600px"
                  style={{ border: '1px solid #ddd', borderRadius: '4px' }}
                  title="PDF Viewer"
                />
              </Box>
            )}
          </Container>

          {/* Chat Interface */}
          <Container>
            <SpaceBetween size="m">
              <Header variant="h2">Chat with Document</Header>
              
              {/* Chat Messages */}
              <Box>
                <div style={{ 
                  height: '400px', 
                  overflowY: 'auto', 
                  border: '1px solid #ddd', 
                  borderRadius: '4px', 
                  padding: '12px',
                  backgroundColor: '#fafafa'
                }}>
                  {chatMessages.length === 0 ? (
                    <Box textAlign="center" color="inherit" padding="l">
                      <Box variant="strong">Start a conversation</Box>
                      <Box variant="p" color="inherit">
                        Ask questions about the uploaded document
                      </Box>
                    </Box>
                  ) : (
                    <SpaceBetween size="s">
                      {chatMessages.map((message) => (
                        <div
                          key={message.id}
                          style={{
                            padding: '12px',
                            backgroundColor: message.role === 'user' ? '#0073bb' : '#ffffff',
                            color: message.role === 'user' ? '#ffffff' : '#000000',
                            borderRadius: '8px',
                            marginLeft: message.role === 'user' ? '48px' : '0',
                            marginRight: message.role === 'assistant' ? '48px' : '0',
                            border: '1px solid #ddd'
                          }}
                        >
                          <Box variant="strong" fontSize="body-s">
                            {message.role === 'user' ? 'You' : 'AI Assistant'}
                          </Box>
                          <Box variant="p" margin={{ top: 'xs' }}>
                            {message.content}
                          </Box>
                          <Box variant="small" color="text-status-inactive" margin={{ top: 'xs' }}>
                            {new Date(message.timestamp).toLocaleTimeString()}
                          </Box>
                        </div>
                      ))}
                      {loading && (
                        <div
                          style={{
                            padding: '12px',
                            backgroundColor: '#ffffff',
                            borderRadius: '8px',
                            marginRight: '48px',
                            border: '1px solid #ddd'
                          }}
                        >
                          <Box variant="strong" fontSize="body-s">AI Assistant</Box>
                          <Box variant="p" margin={{ top: 'xs' }}>
                            <em>Thinking...</em>
                          </Box>
                        </div>
                      )}
                    </SpaceBetween>
                  )}
                </div>
              </Box>

              {/* Message Input */}
              <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-end' }}>
                <div style={{ flex: 1 }}>
                  <Input
                    value={currentMessage}
                    onChange={({ detail }) => setCurrentMessage(detail.value)}
                    placeholder="Ask a question about the document..."
                    onKeyDown={handleKeyPress}
                    disabled={loading}
                  />
                </div>
                <Button
                  variant="primary"
                  onClick={handleSendMessage}
                  loading={loading}
                  disabled={!currentMessage.trim() || loading}
                >
                  Send
                </Button>
              </div>
            </SpaceBetween>
          </Container>
        </Grid>
      )}
    </SpaceBetween>
  )
}