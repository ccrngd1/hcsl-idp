import boto3
import base64
import json
import re
import pandas as pd
import io
from typing import Dict, Any, Optional
from fastapi import HTTPException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BedrockService:
    def __init__(self, region_name: str = "us-east-1"):
        """
        Initialize Bedrock service with EC2 IAM permissions
        """
        try:
            # Use EC2 instance IAM role - no explicit credentials needed
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=region_name
            )
            logger.info(f"Bedrock client initialized for region: {region_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize Bedrock client: {str(e)}")



    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to meet Bedrock requirements:
        - Only alphanumeric, whitespace, hyphens, parentheses, square brackets
        - No consecutive whitespace characters
        """
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9\s\-\(\)\[\]]', '_', filename)
        
        # Replace multiple consecutive whitespace with single space
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Trim whitespace from start and end
        sanitized = sanitized.strip()
        
        # Ensure we have a valid filename
        if not sanitized:
            sanitized = "document"
        
        return sanitized

    def prepare_document_for_bedrock(self, pdf_content: bytes, filename: str = "benefit_document.pdf") -> Dict[str, Any]:
        """
        Prepare document content for Bedrock API
        Returns base64 encoded PDF for multimodal models
        """
        try:
            # Encode PDF as base64 for document analysis
            pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
            
            # Sanitize the filename
            sanitized_filename = self.sanitize_filename(filename)
            
            return {
                "pdf_base64": pdf_base64,
                "document_type": "application/pdf",
                "sanitized_filename": sanitized_filename
            }
        except Exception as e:
            logger.error(f"Failed to prepare document: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to prepare document: {str(e)}")

    async def process_document_with_bedrock(
        self,
        pdf_content: bytes,
        prompt_template: str,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        hyperparameters: Optional[Dict[str, Any]] = None,
        filename: str = "benefit_document.pdf"
    ) -> Dict[str, Any]:
        """
        Process PDF document using AWS Bedrock Converse API
        
        Args:
            pdf_content: Raw PDF file content as bytes
            prompt_template: The prompt template for extraction
            model_id: Bedrock model identifier
            hyperparameters: Model configuration parameters
            
        Returns:
            Dictionary containing extraction results and metadata
        """
        try:
            # Default hyperparameters
            if hyperparameters is None:
                hyperparameters = {
                    "temperature": 0.1
                }

            # Prepare document content
            document_data = self.prepare_document_for_bedrock(pdf_content, filename)
            
            # Send PDF document directly to Bedrock
            message_content = [
                {
                    "text": prompt_template
                },
                {
                    "document": {
                        "format": "pdf",
                        "name": document_data['sanitized_filename'],
                        "source": {
                            "bytes": base64.b64decode(document_data['pdf_base64'])
                        }
                    }
                }
            ]

            # Prepare the converse API request
            converse_request = {
                "modelId": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                "inferenceConfig": {
                    "temperature": hyperparameters.get("temperature", 0.1)
                }
            }

            logger.info(f"Calling Bedrock Converse API with model: {model_id}")
            
            # Log the request details (excluding the actual PDF bytes for brevity)
            request_log = {
                "modelId": model_id,
                "message_role": "user",
                "prompt_text": prompt_template,
                "document_name": document_data['sanitized_filename'],
                "document_format": "pdf",
                "document_size_bytes": len(pdf_content),
                "inference_config": {
                    "temperature": hyperparameters.get("temperature", 0.1)
                }
            }
            print("=" * 80)
            print("BEDROCK REQUEST:")
            print(json.dumps(request_log, indent=2))
            print("=" * 80)
            
            # Call Bedrock Converse API
            response = self.bedrock_client.converse(**converse_request)
            
            # Log the full response
            print("BEDROCK RESPONSE:")
            print(json.dumps(response, indent=2, default=str))
            print("=" * 80)
            
            # Extract the response content
            output_message = response['output']['message']
            content = output_message['content'][0]['text']
            
            # Log just the extracted content for easy reading
            print("EXTRACTED CONTENT:")
            print(content)
            print("=" * 80)
            
            # Parse usage metrics
            usage = response.get('usage', {})
            
            result = {
                "success": True,
                "extracted_content": content,
                "model_used": model_id,
                "usage_metrics": {
                    "input_tokens": usage.get('inputTokens', 0),
                    "output_tokens": usage.get('outputTokens', 0),
                    "total_tokens": usage.get('totalTokens', 0)
                },
                "hyperparameters_used": hyperparameters,
                "document_info": {
                    "pdf_size_bytes": len(pdf_content),
                    "has_pdf_content": True
                }
            }
            
            logger.info(f"Successfully processed document. Output tokens: {usage.get('outputTokens', 0)}")
            return result

        except Exception as e:
            logger.error(f"Bedrock processing failed: {str(e)}")
            
            # Handle specific AWS errors
            if "ValidationException" in str(e):
                raise HTTPException(status_code=400, detail=f"Invalid request parameters: {str(e)}")
            elif "AccessDeniedException" in str(e):
                raise HTTPException(status_code=403, detail="Access denied. Check IAM permissions for Bedrock.")
            elif "ThrottlingException" in str(e):
                raise HTTPException(status_code=429, detail="Request throttled. Please try again later.")
            elif "ModelNotReadyException" in str(e):
                raise HTTPException(status_code=503, detail=f"Model {model_id} is not ready. Try again later.")
            else:
                raise HTTPException(status_code=500, detail=f"Bedrock processing failed: {str(e)}")

    async def validate_extraction_with_bedrock(
        self,
        pdf_content: bytes,
        extracted_json: str,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        hyperparameters: Optional[Dict[str, Any]] = None,
        filename: str = "benefit_document.pdf"
    ) -> Dict[str, Any]:
        """
        Validate extracted JSON against the original PDF document using AWS Bedrock
        
        Args:
            pdf_content: Raw PDF file content as bytes
            extracted_json: The JSON string to validate
            model_id: Bedrock model identifier
            hyperparameters: Model configuration parameters
            filename: Original filename for reference
            
        Returns:
            Dictionary containing validation results and metadata
        """
        try:
            # Default hyperparameters
            if hyperparameters is None:
                hyperparameters = {
                    "temperature": 0.1
                }

            # Prepare document content
            document_data = self.prepare_document_for_bedrock(pdf_content, filename)
            
            # Create validation prompt
            validation_prompt = f"""Validate that the information in the JSON matches the provided document.

Please carefully review the document and the extracted JSON data below, then provide a detailed validation report.

Extracted JSON to validate:
{extracted_json}

Instructions:
1. Compare each piece of information in the JSON against what you can see in the document
2. Identify any discrepancies, missing information, or incorrect values
3. Note any information in the document that wasn't captured in the JSON
4. Provide an overall accuracy assessment (must be exactly "High", "Medium", or "Low")
5. Give specific recommendations for corrections if needed

Please provide your validation in the following format:
- Overall Accuracy: [High/Medium/Low] (use exactly one of these three words)
- Discrepancies Found: [List any incorrect information]
- Missing Information: [List any important information not captured]
- Recommendations: [Specific suggestions for improvement]
- Validation Summary: [Brief overall assessment]

IMPORTANT: Start your response with "Overall Accuracy: High", "Overall Accuracy: Medium", or "Overall Accuracy: Low" so the system can properly categorize the results."""

            # Send PDF document and JSON to Bedrock for validation
            message_content = [
                {
                    "text": validation_prompt
                },
                {
                    "document": {
                        "format": "pdf",
                        "name": document_data['sanitized_filename'],
                        "source": {
                            "bytes": base64.b64decode(document_data['pdf_base64'])
                        }
                    }
                }
            ]

            # Prepare the converse API request
            converse_request = {
                "modelId": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                "inferenceConfig": {
                    "temperature": hyperparameters.get("temperature", 0.1)
                }
            }

            logger.info(f"Calling Bedrock Converse API for validation with model: {model_id}")
            
            # Log the request details (excluding the actual PDF bytes for brevity)
            request_log = {
                "modelId": model_id,
                "message_role": "user",
                "validation_prompt": "Validate JSON against document",
                "document_name": document_data['sanitized_filename'],
                "document_format": "pdf",
                "document_size_bytes": len(pdf_content),
                "json_length": len(extracted_json),
                "inference_config": {
                    "temperature": hyperparameters.get("temperature", 0.1)
                }
            }
            print("=" * 80)
            print("BEDROCK VALIDATION REQUEST:")
            print(json.dumps(request_log, indent=2))
            print("=" * 80)
            
            # Call Bedrock Converse API
            response = self.bedrock_client.converse(**converse_request)
            
            # Log the full response
            print("BEDROCK VALIDATION RESPONSE:")
            print(json.dumps(response, indent=2, default=str))
            print("=" * 80)
            
            # Extract the response content
            output_message = response['output']['message']
            validation_content = output_message['content'][0]['text']
            
            # Log just the validation content for easy reading
            print("VALIDATION CONTENT:")
            print(validation_content)
            print("=" * 80)
            
            # Parse usage metrics
            usage = response.get('usage', {})
            
            result = {
                "success": True,
                "validation_result": validation_content,
                "model_used": model_id,
                "usage_metrics": {
                    "input_tokens": usage.get('inputTokens', 0),
                    "output_tokens": usage.get('outputTokens', 0),
                    "total_tokens": usage.get('totalTokens', 0)
                },
                "hyperparameters_used": hyperparameters,
                "document_info": {
                    "pdf_size_bytes": len(pdf_content),
                    "json_length": len(extracted_json),
                    "has_pdf_content": True
                }
            }
            
            logger.info(f"Successfully validated extraction. Output tokens: {usage.get('outputTokens', 0)}")
            return result

        except Exception as e:
            logger.error(f"Bedrock validation failed: {str(e)}")
            
            # Handle specific AWS errors
            if "ValidationException" in str(e):
                raise HTTPException(status_code=400, detail=f"Invalid request parameters: {str(e)}")
            elif "AccessDeniedException" in str(e):
                raise HTTPException(status_code=403, detail="Access denied. Check IAM permissions for Bedrock.")
            elif "ThrottlingException" in str(e):
                raise HTTPException(status_code=429, detail="Request throttled. Please try again later.")
            elif "ModelNotReadyException" in str(e):
                raise HTTPException(status_code=503, detail=f"Model {model_id} is not ready. Try again later.")
            else:
                raise HTTPException(status_code=500, detail=f"Bedrock validation failed: {str(e)}")

    async def chat_with_document(
        self,
        pdf_content: bytes,
        message: str,
        chat_history: list,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        hyperparameters: Optional[Dict[str, Any]] = None,
        filename: str = "benefit_document.pdf"
    ) -> Dict[str, Any]:
        """
        Chat with PDF document using AWS Bedrock Converse API
        
        Args:
            pdf_content: Raw PDF file content as bytes
            message: User's current message
            chat_history: List of previous chat messages
            model_id: Bedrock model identifier
            hyperparameters: Model configuration parameters
            filename: Original filename for reference
            
        Returns:
            Dictionary containing chat response and metadata
        """
        try:
            # Default hyperparameters
            if hyperparameters is None:
                hyperparameters = {
                    "temperature": 0.7  # Slightly higher for conversational responses
                }

            # Prepare document content
            document_data = self.prepare_document_for_bedrock(pdf_content, filename)
            
            # Build conversation messages
            messages = []
            
            # Add chat history
            for chat_msg in chat_history:
                messages.append({
                    "role": chat_msg["role"],
                    "content": [{"text": chat_msg["content"]}]
                })
            
            # Add current user message with document context
            current_message_content = [
                {
                    "text": f"""You are an AI assistant helping users understand and analyze documents. The user has uploaded a document and wants to chat about it.

Current user question: {message}

Please provide a helpful, accurate response based on the document content. If the question cannot be answered from the document, let the user know that the information is not available in the provided document."""
                },
                {
                    "document": {
                        "format": "pdf",
                        "name": document_data['sanitized_filename'],
                        "source": {
                            "bytes": base64.b64decode(document_data['pdf_base64'])
                        }
                    }
                }
            ]
            
            messages.append({
                "role": "user",
                "content": current_message_content
            })

            # Prepare the converse API request
            converse_request = {
                "modelId": model_id,
                "messages": messages,
                "inferenceConfig": {
                    "temperature": hyperparameters.get("temperature", 0.7)
                }
            }

            logger.info(f"Calling Bedrock Converse API for chat with model: {model_id}")
            
            # Log the request details (excluding the actual PDF bytes for brevity)
            request_log = {
                "modelId": model_id,
                "messages_count": len(messages),
                "current_message": message,
                "chat_history_length": len(chat_history),
                "document_name": document_data['sanitized_filename'],
                "document_format": "pdf",
                "document_size_bytes": len(pdf_content),
                "inference_config": {
                    "temperature": hyperparameters.get("temperature", 0.7)
                }
            }
            print("=" * 80)
            print("BEDROCK CHAT REQUEST:")
            print(json.dumps(request_log, indent=2))
            print("=" * 80)
            
            # Call Bedrock Converse API
            response = self.bedrock_client.converse(**converse_request)
            
            # Log the full response
            print("BEDROCK CHAT RESPONSE:")
            print(json.dumps(response, indent=2, default=str))
            print("=" * 80)
            
            # Extract the response content
            output_message = response['output']['message']
            response_content = output_message['content'][0]['text']
            
            # Log just the response content for easy reading
            print("CHAT RESPONSE CONTENT:")
            print(response_content)
            print("=" * 80)
            
            # Parse usage metrics
            usage = response.get('usage', {})
            
            result = {
                "success": True,
                "response": response_content,
                "model_used": model_id,
                "usage_metrics": {
                    "input_tokens": usage.get('inputTokens', 0),
                    "output_tokens": usage.get('outputTokens', 0),
                    "total_tokens": usage.get('totalTokens', 0)
                },
                "hyperparameters_used": hyperparameters
            }
            
            logger.info(f"Successfully processed chat message. Output tokens: {usage.get('outputTokens', 0)}")
            return result

        except Exception as e:
            logger.error(f"Bedrock chat processing failed: {str(e)}")
            
            # Handle specific AWS errors
            if "ValidationException" in str(e):
                raise HTTPException(status_code=400, detail=f"Invalid request parameters: {str(e)}")
            elif "AccessDeniedException" in str(e):
                raise HTTPException(status_code=403, detail="Access denied. Check IAM permissions for Bedrock.")
            elif "ThrottlingException" in str(e):
                raise HTTPException(status_code=429, detail="Request throttled. Please try again later.")
            elif "ModelNotReadyException" in str(e):
                raise HTTPException(status_code=503, detail=f"Model {model_id} is not ready. Try again later.")
            else:
                raise HTTPException(status_code=500, detail=f"Bedrock chat processing failed: {str(e)}")

    def convert_spreadsheet_to_text(self, spreadsheet_content: bytes, content_type: str, filename: str) -> str:
        """
        Convert spreadsheet content to text format for processing
        """
        try:
            if content_type == "text/csv":
                # Handle CSV files
                df = pd.read_csv(io.BytesIO(spreadsheet_content))
            elif content_type == "application/vnd.ms-excel":
                # Handle .xls files
                df = pd.read_excel(io.BytesIO(spreadsheet_content), engine='xlrd')
            elif content_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                # Handle .xlsx files
                df = pd.read_excel(io.BytesIO(spreadsheet_content), engine='openpyxl')
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            # Convert DataFrame to a structured text representation
            text_content = f"Spreadsheet Data from file: {filename}\n"
            text_content += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
            
            # Add column information
            text_content += "Columns:\n"
            for i, col in enumerate(df.columns):
                text_content += f"{i+1}. {col} (dtype: {df[col].dtype})\n"
            text_content += "\n"
            
            # Add basic statistics for numerical columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                text_content += "Numerical Column Statistics:\n"
                stats = df[numeric_cols].describe()
                text_content += stats.to_string() + "\n\n"
            
            # Add sample data (first 10 rows)
            text_content += "Sample Data (first 10 rows):\n"
            text_content += df.head(10).to_string(index=True) + "\n\n"
            
            # Add data types and null counts
            text_content += "Data Info:\n"
            for col in df.columns:
                null_count = df[col].isnull().sum()
                unique_count = df[col].nunique()
                text_content += f"{col}: {null_count} nulls, {unique_count} unique values\n"
            
            return text_content
            
        except Exception as e:
            logger.error(f"Failed to convert spreadsheet to text: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to process spreadsheet: {str(e)}")

    async def process_spreadsheet_with_bedrock(
        self,
        spreadsheet_content: bytes,
        prompt_template: str,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        hyperparameters: Optional[Dict[str, Any]] = None,
        filename: str = "spreadsheet",
        content_type: str = "text/csv"
    ) -> Dict[str, Any]:
        """
        Process spreadsheet data using AWS Bedrock
        
        Args:
            spreadsheet_content: Raw spreadsheet file content as bytes
            prompt_template: The prompt template for analysis
            model_id: Bedrock model identifier
            hyperparameters: Model configuration parameters
            filename: Original filename for reference
            content_type: MIME type of the spreadsheet
            
        Returns:
            Dictionary containing analysis results and metadata
        """
        try:
            # Default hyperparameters
            if hyperparameters is None:
                hyperparameters = {
                    "temperature": 0.1
                }

            # Convert spreadsheet to text
            text_content = self.convert_spreadsheet_to_text(spreadsheet_content, content_type, filename)
            
            # Combine prompt with spreadsheet data
            full_prompt = f"{prompt_template}\n\nSpreadsheet Data:\n{text_content}"

            # Prepare the converse API request
            converse_request = {
                "modelId": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": full_prompt
                            }
                        ]
                    }
                ],
                "inferenceConfig": {
                    "temperature": hyperparameters.get("temperature", 0.1)
                }
            }

            logger.info(f"Calling Bedrock Converse API for spreadsheet analysis with model: {model_id}")
            
            # Log the request details
            request_log = {
                "modelId": model_id,
                "message_role": "user",
                "prompt_template": prompt_template,
                "filename": filename,
                "content_type": content_type,
                "spreadsheet_size_bytes": len(spreadsheet_content),
                "text_content_length": len(text_content),
                "inference_config": {
                    "temperature": hyperparameters.get("temperature", 0.1)
                }
            }
            print("=" * 80)
            print("BEDROCK SPREADSHEET REQUEST:")
            print(json.dumps(request_log, indent=2))
            print("=" * 80)
            
            # Call Bedrock Converse API
            response = self.bedrock_client.converse(**converse_request)
            
            # Log the full response
            print("BEDROCK SPREADSHEET RESPONSE:")
            print(json.dumps(response, indent=2, default=str))
            print("=" * 80)
            
            # Extract the response content
            output_message = response['output']['message']
            content = output_message['content'][0]['text']
            
            # Log just the extracted content for easy reading
            print("EXTRACTED SPREADSHEET ANALYSIS:")
            print(content)
            print("=" * 80)
            
            # Parse usage metrics
            usage = response.get('usage', {})
            
            result = {
                "success": True,
                "extracted_content": content,
                "model_used": model_id,
                "usage_metrics": {
                    "input_tokens": usage.get('inputTokens', 0),
                    "output_tokens": usage.get('outputTokens', 0),
                    "total_tokens": usage.get('totalTokens', 0)
                },
                "hyperparameters_used": hyperparameters,
                "document_info": {
                    "spreadsheet_size_bytes": len(spreadsheet_content),
                    "text_content_length": len(text_content),
                    "content_type": content_type,
                    "filename": filename,
                    "has_spreadsheet_content": True
                }
            }
            
            logger.info(f"Successfully processed spreadsheet. Output tokens: {usage.get('outputTokens', 0)}")
            return result

        except Exception as e:
            logger.error(f"Bedrock spreadsheet processing failed: {str(e)}")
            
            # Handle specific AWS errors
            if "ValidationException" in str(e):
                raise HTTPException(status_code=400, detail=f"Invalid request parameters: {str(e)}")
            elif "AccessDeniedException" in str(e):
                raise HTTPException(status_code=403, detail="Access denied. Check IAM permissions for Bedrock.")
            elif "ThrottlingException" in str(e):
                raise HTTPException(status_code=429, detail="Request throttled. Please try again later.")
            elif "ModelNotReadyException" in str(e):
                raise HTTPException(status_code=503, detail=f"Model {model_id} is not ready. Try again later.")
            else:
                raise HTTPException(status_code=500, detail=f"Bedrock spreadsheet processing failed: {str(e)}")

    async def validate_spreadsheet_extraction_with_bedrock(
        self,
        spreadsheet_content: bytes,
        extracted_json: str,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        hyperparameters: Optional[Dict[str, Any]] = None,
        filename: str = "spreadsheet",
        content_type: str = "text/csv"
    ) -> Dict[str, Any]:
        """
        Validate extracted JSON against the original spreadsheet data using AWS Bedrock
        
        Args:
            spreadsheet_content: Raw spreadsheet file content as bytes
            extracted_json: The JSON string to validate
            model_id: Bedrock model identifier
            hyperparameters: Model configuration parameters
            filename: Original filename for reference
            content_type: MIME type of the spreadsheet
            
        Returns:
            Dictionary containing validation results and metadata
        """
        try:
            # Default hyperparameters
            if hyperparameters is None:
                hyperparameters = {
                    "temperature": 0.1
                }

            # Convert spreadsheet to text
            text_content = self.convert_spreadsheet_to_text(spreadsheet_content, content_type, filename)
            
            # Create validation prompt
            validation_prompt = f"""Validate that the information in the JSON matches the provided spreadsheet data.

Please carefully review the spreadsheet data and the extracted JSON data below, then provide a detailed validation report.

Spreadsheet Data:
{text_content}

Extracted JSON to validate:
{extracted_json}

Instructions:
1. Compare each piece of information in the JSON against what you can see in the spreadsheet data
2. Verify statistical calculations and data summaries
3. Check if patterns and insights are supported by the actual data
4. Identify any discrepancies, missing information, or incorrect values
5. Note any important information in the spreadsheet that wasn't captured in the JSON
6. Provide an overall accuracy assessment (must be exactly "High", "Medium", or "Low")
7. Give specific recommendations for corrections if needed

Please provide your validation in the following format:
- Overall Accuracy: [High/Medium/Low] (use exactly one of these three words)
- Data Accuracy: [Assessment of statistical accuracy]
- Insights Validation: [Assessment of patterns and insights]
- Missing Information: [List any important information not captured]
- Recommendations: [Specific suggestions for improvement]
- Validation Summary: [Brief overall assessment]

IMPORTANT: Start your response with "Overall Accuracy: High", "Overall Accuracy: Medium", or "Overall Accuracy: Low" so the system can properly categorize the results."""

            # Prepare the converse API request
            converse_request = {
                "modelId": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": validation_prompt
                            }
                        ]
                    }
                ],
                "inferenceConfig": {
                    "temperature": hyperparameters.get("temperature", 0.1)
                }
            }

            logger.info(f"Calling Bedrock Converse API for spreadsheet validation with model: {model_id}")
            
            # Log the request details
            request_log = {
                "modelId": model_id,
                "message_role": "user",
                "validation_prompt": "Validate JSON against spreadsheet data",
                "filename": filename,
                "content_type": content_type,
                "spreadsheet_size_bytes": len(spreadsheet_content),
                "json_length": len(extracted_json),
                "text_content_length": len(text_content),
                "inference_config": {
                    "temperature": hyperparameters.get("temperature", 0.1)
                }
            }
            print("=" * 80)
            print("BEDROCK SPREADSHEET VALIDATION REQUEST:")
            print(json.dumps(request_log, indent=2))
            print("=" * 80)
            
            # Call Bedrock Converse API
            response = self.bedrock_client.converse(**converse_request)
            
            # Log the full response
            print("BEDROCK SPREADSHEET VALIDATION RESPONSE:")
            print(json.dumps(response, indent=2, default=str))
            print("=" * 80)
            
            # Extract the response content
            output_message = response['output']['message']
            validation_content = output_message['content'][0]['text']
            
            # Log just the validation content for easy reading
            print("SPREADSHEET VALIDATION CONTENT:")
            print(validation_content)
            print("=" * 80)
            
            # Parse usage metrics
            usage = response.get('usage', {})
            
            result = {
                "success": True,
                "validation_result": validation_content,
                "model_used": model_id,
                "usage_metrics": {
                    "input_tokens": usage.get('inputTokens', 0),
                    "output_tokens": usage.get('outputTokens', 0),
                    "total_tokens": usage.get('totalTokens', 0)
                },
                "hyperparameters_used": hyperparameters,
                "document_info": {
                    "spreadsheet_size_bytes": len(spreadsheet_content),
                    "json_length": len(extracted_json),
                    "text_content_length": len(text_content),
                    "content_type": content_type,
                    "filename": filename,
                    "has_spreadsheet_content": True
                }
            }
            
            logger.info(f"Successfully validated spreadsheet extraction. Output tokens: {usage.get('outputTokens', 0)}")
            return result

        except Exception as e:
            logger.error(f"Bedrock spreadsheet validation failed: {str(e)}")
            
            # Handle specific AWS errors
            if "ValidationException" in str(e):
                raise HTTPException(status_code=400, detail=f"Invalid request parameters: {str(e)}")
            elif "AccessDeniedException" in str(e):
                raise HTTPException(status_code=403, detail="Access denied. Check IAM permissions for Bedrock.")
            elif "ThrottlingException" in str(e):
                raise HTTPException(status_code=429, detail="Request throttled. Please try again later.")
            elif "ModelNotReadyException" in str(e):
                raise HTTPException(status_code=503, detail=f"Model {model_id} is not ready. Try again later.")
            else:
                raise HTTPException(status_code=500, detail=f"Bedrock spreadsheet validation failed: {str(e)}")

    def get_available_models(self) -> Dict[str, Any]:
        """
        Get list of available Bedrock models for document processing
        """
        return {
            "text_models": [
                {
                    "id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                    "name": "Claude 3.7 Sonnet", 
                    "description": "Claude 3.7 with improved performance",
                    "supports_documents": True
                }, 
                {
                    "id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                    "name": "Claude 4 Sonnet",
                    "description": "Claude 4 generation with enhanced reasoning",
                    "supports_documents": True
                },
                {
                    "id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    "name": "Claude 4.5 Sonnet", 
                    "description": "Latest Claude 4.5 generation with advanced capabilities",
                    "supports_documents": True
                },
                {
                    "id": "us.anthropic.claude-opus-4-1-20250805-v1:0",
                    "name": "Claude 4.1 Opus", 
                    "description": "Claude 4.1 Opus with enhanced reasoning",
                    "supports_documents": True
                },
                {
                    "id": "us.anthropic.claude-opus-4-5-20250514-v1:0", 
                    "name": "Claude 4.5 Opus", 
                    "description": "Most capable Claude 4.5 model",
                    "supports_documents": True
                },
                {
                    "id": "us.amazon.nova-pro-v1:0", 
                    "name": "Nova 1 Pro", 
                    "description": "Amazon's professional multimodal model",
                    "supports_documents": True
                },
                {
                    "id": "us.amazon.nova-premier-v1:0", 
                    "name": "Nova 1 Premier", 
                    "description": "Amazon's premier multimodal model",
                    "supports_documents": True
                },
                {
                    "id": "us.amazon.nova-2-pro-v1:0", 
                    "name": "Nova 2 Pro", 
                    "description": "Amazon's advanced Nova 2 model",
                    "supports_documents": True
                },
                {
                    "id": "us.amazon.nova-2-premier-v1:0", 
                    "name": "Nova 2 Premier", 
                    "description": "Amazon's most capable Nova 2 model",
                    "supports_documents": True
                }
            ],
            "default_hyperparameters": {
                "temperature": 0.1
            }
        }

# Create a singleton instance
bedrock_service = BedrockService()