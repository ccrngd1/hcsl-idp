#!/usr/bin/env python3
"""
Test script for Bedrock integration
Run this to verify your AWS credentials and Bedrock access
"""

import asyncio
import sys
from bedrock_service import bedrock_service

async def test_bedrock_setup():
    """Test Bedrock service setup and permissions"""
    
    print("🔍 Testing Bedrock Service Setup...")
    print("=" * 50)
    
    try:
        # Test 1: Check available models
        print("1. Testing model availability...")
        models = bedrock_service.get_available_models()
        print(f"✅ Found {len(models['text_models'])} available models")
        
        for model in models['text_models']:
            print(f"   - {model['name']} ({model['id']})")
        
        # Test 2: Create a simple test document
        print("\n2. Testing document processing...")
        test_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Test Document) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000206 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n299\n%%EOF"
        
        # Simple test prompt
        test_prompt = "Extract any text content from this document and summarize it briefly."
        
        print("   Creating test extraction request...")
        
        # Note: This will fail if IAM permissions aren't set up correctly
        result = await bedrock_service.process_document_with_bedrock(
            pdf_content=test_pdf_content,
            prompt_template=test_prompt,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",  # Use faster model for testing
            hyperparameters={"max_tokens": 100, "temperature": 0.1}
        )
        
        print("✅ Document processing successful!")
        print(f"   Model used: {result['model_used']}")
        print(f"   Tokens used: {result['usage_metrics']['total_tokens']}")
        print(f"   Response preview: {result['extracted_content'][:100]}...")
        
        print("\n🎉 All tests passed! Bedrock integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure your EC2 instance has an IAM role with Bedrock permissions")
        print("2. Check that the IAM role includes these permissions:")
        print("   - bedrock:InvokeModel")
        print("   - bedrock:InvokeModelWithResponseStream")
        print("3. Verify the AWS region supports Bedrock (us-east-1, us-west-2, etc.)")
        print("4. Make sure the Claude models are enabled in your AWS account")
        
        return False

if __name__ == "__main__":
    success = asyncio.run(test_bedrock_setup())
    sys.exit(0 if success else 1)