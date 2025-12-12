import boto3
import base64
import json
import re
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