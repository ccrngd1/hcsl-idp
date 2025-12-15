import { useState, useRef, useEffect } from 'react'
import {
  Container,
  Header,
  SpaceBetween,
  Button,
  Box,
  Alert,
  ProgressBar,
  Textarea,
  FormField,
  ColumnLayout,
  Select,
  Input
} from '@cloudscape-design/components'
import { JsonViewer } from '@textea/json-viewer'
import { bedrockService } from '../services/api'

interface ModelOption {
  label: string
  value: string
  description?: string
}



export default function DataReportAll() {
  const [loading, setLoading] = useState(false)
  const [uploadingFile, setUploadingFile] = useState(false)
  // Split the prompt into instructions and JSON schema for data reports
  const [dataReportInstructions, setDataReportInstructions] = useState(
    sessionStorage.getItem('spreadsheetDataReportInstructions') || 
    `Analyze this spreadsheet data and generate a comprehensive data report with the following structure:

1. Perform statistical analysis on numerical columns
2. Identify patterns and trends in the data
3. Generate summary insights and key findings
4. Structure the output according to the JSON schema provided

Return only the JSON output.`
  )
  
  const [dataReportSchema, setDataReportSchema] = useState(
    sessionStorage.getItem('spreadsheetDataReportSchema') || 
    `{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["data_summary", "statistical_analysis", "insights", "recommendations"],
  "properties": {
    "data_summary": {
      "type": "object",
      "required": ["total_rows", "total_columns", "column_info", "data_types"],
      "properties": {
        "total_rows": { "type": "number" },
        "total_columns": { "type": "number" },
        "column_info": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "column_name": { "type": "string" },
              "data_type": { "type": "string" },
              "non_null_count": { "type": "number" },
              "unique_values": { "type": "number" }
            }
          }
        },
        "data_types": {
          "type": "object",
          "properties": {
            "numerical_columns": { "type": "array", "items": { "type": "string" } },
            "categorical_columns": { "type": "array", "items": { "type": "string" } },
            "date_columns": { "type": "array", "items": { "type": "string" } }
          }
        }
      }
    },
    "statistical_analysis": {
      "type": "object",
      "properties": {
        "numerical_stats": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "column": { "type": "string" },
              "mean": { "type": "number" },
              "median": { "type": "number" },
              "std_dev": { "type": "number" },
              "min": { "type": "number" },
              "max": { "type": "number" },
              "quartiles": {
                "type": "object",
                "properties": {
                  "q1": { "type": "number" },
                  "q3": { "type": "number" }
                }
              }
            }
          }
        },
        "categorical_stats": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "column": { "type": "string" },
              "unique_count": { "type": "number" },
              "most_frequent": { "type": "string" },
              "frequency_distribution": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "value": { "type": "string" },
                    "count": { "type": "number" },
                    "percentage": { "type": "number" }
                  }
                }
              }
            }
          }
        }
      }
    },
    "insights": {
      "type": "object",
      "required": ["key_findings", "patterns", "anomalies"],
      "properties": {
        "key_findings": {
          "type": "array",
          "items": { "type": "string" }
        },
        "patterns": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "description": { "type": "string" },
              "confidence": { "type": "string", "enum": ["high", "medium", "low"] },
              "supporting_data": { "type": "string" }
            }
          }
        },
        "anomalies": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "type": { "type": "string" },
              "description": { "type": "string" },
              "affected_rows": { "type": "number" },
              "severity": { "type": "string", "enum": ["high", "medium", "low"] }
            }
          }
        }
      }
    },
    "recommendations": {
      "type": "object",
      "required": ["data_quality", "analysis_suggestions", "next_steps"],
      "properties": {
        "data_quality": {
          "type": "array",
          "items": { "type": "string" }
        },
        "analysis_suggestions": {
          "type": "array",
          "items": { "type": "string" }
        },
        "next_steps": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    },
    "metadata": {
      "type": "object",
      "properties": {
        "analysis_date": { "type": "string", "format": "date-time" },
        "file_name": { "type": "string" },
        "processing_time": { "type": "string" }
      }
    }
  }
}`
  )
  const [bedrockOutput, setBedrockOutput] = useState('')
  const [parsedJsonOutput, setParsedJsonOutput] = useState<any>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  // Validation state
  const [validating, setValidating] = useState(false)
  const [validationResult, setValidationResult] = useState('')
  const [hasExtracted, setHasExtracted] = useState(false)
  const [validationAccuracy, setValidationAccuracy] = useState<'High' | 'Medium' | 'Low' | null>(null)
  const [showValidationAlert, setShowValidationAlert] = useState(false)

  // LLM Configuration State
  const [selectedModel, setSelectedModel] = useState<ModelOption | null>(null)
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([])
  const [modelsLoading, setModelsLoading] = useState(true)
  const [temperature, setTemperature] = useState('0.1')

  // Load available models on component mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        setModelsLoading(true)
        const modelsData = await bedrockService.getAvailableModels()
        
        // Transform backend model data to frontend format
        const options: ModelOption[] = modelsData.text_models.map((model: any) => ({
          label: model.name,
          value: model.id,
          description: model.description
        }))
        
        setModelOptions(options)
        
        // Set default model (first one in the list)
        if (options.length > 0) {
          setSelectedModel(options[0])
        }
      } catch (error) {
        console.error('Failed to load models:', error)
        // Fallback to a basic model if API fails
        const fallbackOptions: ModelOption[] = [
          {
            label: 'Claude 3 Sonnet',
            value: 'anthropic.claude-3-sonnet-20240229-v1:0',
            description: 'Fallback model'
          }
        ]
        setModelOptions(fallbackOptions)
        setSelectedModel(fallbackOptions[0])
      } finally {
        setModelsLoading(false)
      }
    }

    loadModels()
  }, [])


  const handleExtractAll = async () => {
    if (!selectedFile) {
      alert('Please upload a spreadsheet file first')
      return
    }

    if (!selectedModel) {
      alert('Please select a model first')
      return
    }

    // Save both parts to session storage
    sessionStorage.setItem('spreadsheetDataReportInstructions', dataReportInstructions)
    sessionStorage.setItem('spreadsheetDataReportSchema', dataReportSchema)
    
    // Combine instructions and schema into a single prompt
    const combinedPrompt = `${dataReportInstructions}\n\n${dataReportSchema}`
    
    setLoading(true)
    setBedrockOutput('Processing spreadsheet with Bedrock...')

    try {
      // Prepare hyperparameters
      const hyperparameters = {
        temperature: parseFloat(temperature)
      }

      // Call the backend API using the service with combined prompt
      const result = await bedrockService.extractFromDocument(
        selectedFile,
        combinedPrompt,
        selectedModel.value,
        hyperparameters
      )
      
      // Display the extraction results
      const outputText = `${result.extracted_content}`
      setBedrockOutput(outputText)

      // Try to parse JSON for the viewer
      try {
        // Clean the content by removing markdown code blocks
        let cleanedContent = result.extracted_content.trim()
        
        // Remove markdown code blocks (```json, ```JSON, or just ```)
        cleanedContent = cleanedContent.replace(/^```(?:json|JSON)?\s*\n?/i, '')
        cleanedContent = cleanedContent.replace(/\n?```\s*$/i, '')
        cleanedContent = cleanedContent.trim()
        
        const jsonData = JSON.parse(cleanedContent)
        setParsedJsonOutput(jsonData)
        setHasExtracted(true) // Enable validation button
        console.log('Successfully parsed JSON from Bedrock output')
      } catch (parseError) {
        console.log('Could not parse as JSON, showing raw text:', parseError)
        setParsedJsonOutput(null)
        setHasExtracted(true) // Still enable validation even if JSON parsing failed
      }

    } catch (error) {
      console.error('Extraction failed:', error)
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
      setBedrockOutput(`Error: ${errorMessage}`)
      setParsedJsonOutput(null)
      setHasExtracted(false)
    } finally {
      setLoading(false)
    }
  }

  // Function to parse accuracy from validation result
  const parseValidationAccuracy = (validationText: string): 'High' | 'Medium' | 'Low' | null => {
    const lowerText = validationText.toLowerCase()
    if (lowerText.includes('overall accuracy: high') || lowerText.includes('accuracy: high')) {
      return 'High'
    } else if (lowerText.includes('overall accuracy: medium') || lowerText.includes('accuracy: medium')) {
      return 'Medium'
    } else if (lowerText.includes('overall accuracy: low') || lowerText.includes('accuracy: low')) {
      return 'Low'
    }
    // Try to infer from other indicators
    if (lowerText.includes('excellent') || lowerText.includes('accurate') || lowerText.includes('no discrepancies')) {
      return 'High'
    } else if (lowerText.includes('some discrepancies') || lowerText.includes('minor issues')) {
      return 'Medium'
    } else if (lowerText.includes('significant discrepancies') || lowerText.includes('major issues')) {
      return 'Low'
    }
    return null
  }

  const handleValidate = async () => {
    if (!selectedFile) {
      alert('Please upload a spreadsheet file first')
      return
    }

    if (!bedrockOutput) {
      alert('Please extract data first before validating')
      return
    }

    if (!selectedModel) {
      alert('Please select a model first')
      return
    }

    setValidating(true)
    setValidationResult('Validating extraction with Bedrock...')
    setShowValidationAlert(false)

    try {
      // Prepare hyperparameters
      const hyperparameters = {
        temperature: parseFloat(temperature)
      }

      // Call the validation API
      const result = await bedrockService.validateExtraction(
        selectedFile,
        bedrockOutput,
        selectedModel.value,
        hyperparameters
      )
      
      // Display the validation results
      setValidationResult(result.validation_result)
      
      // Parse accuracy and show alert
      const accuracy = parseValidationAccuracy(result.validation_result)
      setValidationAccuracy(accuracy)
      setShowValidationAlert(true)

    } catch (error) {
      console.error('Validation failed:', error)
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
      setValidationResult(`Validation Error: ${errorMessage}`)
      setValidationAccuracy('Low')
      setShowValidationAlert(true)
    } finally {
      setValidating(false)
    }
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    console.log('handleFileUpload called')
    const file = event.target.files?.[0]
    console.log('Selected file:', file)
    
    const allowedTypes = [
      'application/vnd.ms-excel', // .xls
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
      'text/csv' // .csv
    ]
    
    if (file && allowedTypes.includes(file.type)) {
      console.log('Starting file upload process...')
      setUploadingFile(true)
      
      // Force a re-render by waiting for the next tick
      await new Promise(resolve => setTimeout(resolve, 0))
      console.log('uploadingFile should now be true')
      
      try {
        // Add a longer delay to show loading state
        await new Promise(resolve => setTimeout(resolve, 1000))
        
        console.log('Processing file selection...')
        
        // Set the selected file
        setSelectedFile(file)
        await new Promise(resolve => setTimeout(resolve, 0))
        
        // Clear previous output
        setBedrockOutput('')
        setParsedJsonOutput(null)
        setHasExtracted(false)
        setValidationResult('')
        setValidationAccuracy(null)
        setShowValidationAlert(false)
        
        console.log('Spreadsheet file loaded:', file.name, `(${(file.size / 1024 / 1024).toFixed(2)} MB)`)
      } catch (error) {
        console.error('Failed to load spreadsheet:', error)
        alert('Failed to load spreadsheet file')
      } finally {
        console.log('Upload process complete, clearing uploadingFile state')
        setUploadingFile(false)
      }
    } else {
      alert('Please select a valid spreadsheet file (XLS, XLSX, or CSV)')
    }
  }

  const handleUploadClick = () => {
    console.log('Upload button clicked')
    console.log('File input ref:', fileInputRef.current)
    fileInputRef.current?.click()
  }





  return (
    <SpaceBetween size="l">
      <Header
        variant="h1"
        description="Generate comprehensive data reports from uploaded documents"
        actions={
          <SpaceBetween direction="horizontal" size="xs">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".xls,.xlsx,.csv"
              style={{ display: 'none' }}
            />
            <Button
              onClick={handleUploadClick}
              iconName="upload"
              loading={uploadingFile}
            >
              {uploadingFile ? 'Loading Spreadsheet...' : 'Upload Spreadsheet'}
            </Button>
            <Button
              variant="primary"
              onClick={handleExtractAll}
              loading={loading}
            >
              Generate Data Report
            </Button>
            <Button
              onClick={handleValidate}
              loading={validating}
              disabled={!hasExtracted}
              iconName="check"
            >
              {validating ? 'Validating...' : 'Validate'}
            </Button>
          </SpaceBetween>
        }
      >
        Data Report - All Extractions
      </Header>

      {loading && (
        <Container>
          <ProgressBar
            status="in-progress"
            value={65}
            label="Processing spreadsheet..."
            description="Generating comprehensive data report from spreadsheet data"
          />
        </Container>
      )}

      <Alert type="info">
        This page processes spreadsheet files (XLS, XLSX, CSV) and generates comprehensive data reports including detailed analysis, metrics, and structured information extraction.
      </Alert>

      {showValidationAlert && validationAccuracy && (
        <Alert
          type={
            validationAccuracy === 'High' ? 'success' : 
            validationAccuracy === 'Medium' ? 'warning' : 
            'error'
          }
          dismissible
          onDismiss={() => setShowValidationAlert(false)}
          header={`Validation Complete - ${validationAccuracy} Accuracy`}
        >
          <SpaceBetween size="s">
            <Box>
              {validationAccuracy === 'High' && 'The extracted data appears to be highly accurate and matches the document well.'}
              {validationAccuracy === 'Medium' && 'The extracted data is mostly accurate but may have some minor discrepancies.'}
              {validationAccuracy === 'Low' && 'The extracted data has significant discrepancies and should be reviewed carefully.'}
            </Box>
            <details>
              <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>
                View detailed validation report
              </summary>
              <Box margin={{ top: 's' }}>
                <Textarea
                  value={validationResult}
                  readOnly
                  rows={10}
                  placeholder="Detailed validation feedback will appear here..."
                />
              </Box>
            </details>
          </SpaceBetween>
        </Alert>
      )} 

      <ColumnLayout columns={2} variant="text-grid">
        <SpaceBetween size="m">
          <Container header={<Header variant="h2">Spreadsheet File Information</Header>}>
            {uploadingFile ? (
              <Box textAlign="center" padding="xxl">
                <ProgressBar
                  status="in-progress"
                  label="Loading Spreadsheet..."
                  description="Preparing spreadsheet for processing"
                />
              </Box>
            ) : selectedFile ? (
              <Box padding="l">
                <SpaceBetween size="s">
                  <Box>
                    <Box variant="strong">File Name:</Box> {selectedFile.name}
                  </Box>
                  <Box>
                    <Box variant="strong">File Size:</Box> {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </Box>
                  <Box>
                    <Box variant="strong">File Type:</Box> {selectedFile.type || 'Unknown'}
                  </Box>
                  <Box>
                    <Box variant="strong">Last Modified:</Box> {new Date(selectedFile.lastModified).toLocaleString()}
                  </Box>
                </SpaceBetween>
                <Alert type="info" header="File Processing Note">
                  Spreadsheet files cannot be previewed directly. Click "Generate Data Report" to process the file and extract structured data analysis.
                </Alert>
              </Box>
            ) : (
              <Box textAlign="center" padding="xxl" color="text-status-inactive">
                <Box variant="strong">No spreadsheet selected</Box>
                <Box variant="p">Upload a spreadsheet file (XLS, XLSX, or CSV) to process it</Box>
              </Box>
            )}
          </Container>

          <Container header={<Header variant="h2">Prompt Configuration</Header>}>
            <SpaceBetween size="m">
              <FormField
                label="Data Report Instructions"
                description="Enter the instructions for how to analyze the spreadsheet data and generate the report. This will be combined with the JSON schema below."
              >
                <Textarea
                  value={dataReportInstructions}
                  onChange={({ detail }) => setDataReportInstructions(detail.value)}
                  rows={6}
                  placeholder="Enter instructions for spreadsheet data analysis and reporting..."
                />
              </FormField>
              
              <FormField
                label="Data Report JSON Schema"
                description="Define the JSON schema that the model should follow for the data report output format."
              >
                <Textarea
                  value={dataReportSchema}
                  onChange={({ detail }) => setDataReportSchema(detail.value)}
                  rows={15}
                  placeholder="Enter the expected JSON schema for data reports..."
                />
              </FormField>
            </SpaceBetween>
          </Container>

          <Container header={<Header variant="h2">Model Configuration</Header>}>
            <SpaceBetween size="s">
              <FormField label="Model Selection" description="Choose the Bedrock model for spreadsheet data analysis">
                <Select
                  selectedOption={selectedModel}
                  onChange={({ detail }) => setSelectedModel(detail.selectedOption as ModelOption)}
                  options={modelOptions}
                  placeholder={modelsLoading ? "Loading models..." : "Select a model"}
                  disabled={modelsLoading}
                  loadingText="Loading available models..."
                />
              </FormField>
              

              
              <FormField label="Temperature" description="Controls randomness (0.0 = deterministic, 1.0 = creative)">
                <Input
                  value={temperature}
                  onChange={({ detail }) => setTemperature(detail.value)}
                  type="number"
                  step={0.1}
                  placeholder="0.1"
                />
              </FormField>
            </SpaceBetween>
          </Container>
        </SpaceBetween>

        <Container header={<Header variant="h2">Bedrock Data Report Output</Header>}>
          <FormField
            description="Results from Amazon Bedrock spreadsheet data analysis"
          >
            {parsedJsonOutput ? (
              <div style={{ 
                border: '1px solid #e1e4e8', 
                borderRadius: '4px', 
                padding: '12px',
                backgroundColor: '#f8f9fa',
                height: '1200px',
                overflow: 'auto'
              }}>
                <JsonViewer 
                  value={parsedJsonOutput}
                  theme="light"
                  displayDataTypes={false}
                  enableClipboard={true}
                  rootName="data_report_result"
                />
              </div>
            ) : bedrockOutput ? (
              <Textarea
                value={bedrockOutput}
                onChange={({ detail }) => setBedrockOutput(detail.value)}
                rows={30}
                placeholder="No data report results yet. Upload a spreadsheet and click Generate Data Report to see Bedrock output here."
              />
            ) : (
              <Box textAlign="center" padding="l" color="text-status-inactive">
                <Box variant="strong">No data report results yet</Box>
                <Box variant="p">Upload a spreadsheet and click Generate Data Report to see Bedrock output here.</Box>
              </Box>
            )}
          </FormField>
        </Container>
      </ColumnLayout>


    </SpaceBetween>
  )
}