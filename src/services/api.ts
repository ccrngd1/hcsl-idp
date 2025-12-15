import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000, // Increased timeout for Bedrock processing
})

// Bedrock API configuration
const bedrockApi = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 120000, // 2 minutes for large document processing
})

export const apiService = {
  async getItems() {
    const response = await api.get('/items')
    return response.data
  },

  async createItem(data: { name: string; status: string }) {
    const response = await api.post('/items', data)
    return response.data
  },

  async updateItem(id: string, data: { name?: string; status?: string }) {
    const response = await api.put(`/items/${id}`, data)
    return response.data
  },

  async deleteItem(id: string) {
    await api.delete(`/items/${id}`)
  }
}

// Bedrock service for document processing
export const bedrockService = {
  async extractFromDocument(
    pdfFile: File,
    promptTemplate: string,
    modelId: string,
    hyperparameters: {
      temperature: number
    }
  ) {
    const formData = new FormData()
    formData.append('pdf_file', pdfFile)
    formData.append('prompt_template', promptTemplate)
    formData.append('model_id', modelId)
    formData.append('hyperparameters', JSON.stringify(hyperparameters))

    const response = await bedrockApi.post('/bedrock/extract', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  async getAvailableModels() {
    const response = await bedrockApi.get('/bedrock/models')
    return response.data
  },

  async validateExtraction(
    pdfFile: File,
    extractedJson: string,
    modelId: string,
    hyperparameters: {
      temperature: number
    }
  ) {
    const formData = new FormData()
    formData.append('pdf_file', pdfFile)
    formData.append('extracted_json', extractedJson)
    formData.append('model_id', modelId)
    formData.append('hyperparameters', JSON.stringify(hyperparameters))

    const response = await bedrockApi.post('/bedrock/validate', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  async testConnection() {
    const response = await bedrockApi.post('/bedrock/test-connection')
    return response.data
  },

  async chatWithDocument(
    pdfFile: File,
    message: string,
    chatHistory: Array<{ role: string; content: string; timestamp: string }>,
    modelId: string = "anthropic.claude-3-sonnet-20240229-v1:0",
    hyperparameters: { temperature: number } = { temperature: 0.7 }
  ) {
    const formData = new FormData()
    formData.append('pdf_file', pdfFile)
    formData.append('message', message)
    formData.append('chat_history', JSON.stringify(chatHistory))
    formData.append('model_id', modelId)
    formData.append('hyperparameters', JSON.stringify(hyperparameters))

    const response = await bedrockApi.post('/bedrock/chat', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  }
}