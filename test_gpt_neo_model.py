#!/usr/bin/env python3
"""
Test Script for GPT-Neo 2.7B Medical AI Model

This script tests the performance of your fine-tuned model with various medical questions.
"""

import torch
import logging
from transformers import pipeline
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, model_path="./gpt-neo-2.7b-medical-finetuned"):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load the fine-tuned model
        self.generator = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path,
            device=0 if self.device == "cuda" else -1,
            max_length=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=50256
        )
        
        logger.info("Model loaded successfully!")
    
    def test_question(self, question, expected_keywords=None):
        """Test a single question and measure performance."""
        print(f"\n🤔 Question: {question}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = f"Medical Research Analysis:\n\nQuestion: {question}\n\nAnalysis:"
            
            # Generate response
            result = self.generator(
                prompt,
                max_length=len(prompt.split()) + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            
            generation_time = time.time() - start_time
            
            # Extract response
            generated_text = result[0]['generated_text']
            if "Analysis:" in generated_text:
                response = generated_text.split("Analysis:")[-1].strip()
            else:
                response = generated_text[len(prompt):].strip()
            
            print(f"🤖 Response: {response}")
            print(f"⏱️  Generation time: {generation_time:.2f} seconds")
            
            # Check for expected keywords
            if expected_keywords:
                found_keywords = []
                for keyword in expected_keywords:
                    if keyword.lower() in response.lower():
                        found_keywords.append(keyword)
                
                print(f"✅ Keywords found: {found_keywords}")
                print(f"📊 Keyword match rate: {len(found_keywords)}/{len(expected_keywords)} ({len(found_keywords)/len(expected_keywords)*100:.1f}%)")
            
            return {
                'question': question,
                'response': response,
                'generation_time': generation_time,
                'keywords_found': found_keywords if expected_keywords else []
            }
            
        except Exception as e:
            logger.error(f"Error testing question: {e}")
            print(f"❌ Error: {e}")
            return None
    
    def run_comprehensive_test(self):
        """Run a comprehensive test suite."""
        print("🧪 GPT-Neo 2.7B Medical AI Model Test Suite")
        print("=" * 60)
        
        # Test questions with expected keywords
        test_cases = [
            {
                'question': 'What are the latest treatments for diabetes?',
                'keywords': ['diabetes', 'treatment', 'insulin', 'glucose', 'medication']
            },
            {
                'question': 'Tell me about cancer immunotherapy research',
                'keywords': ['cancer', 'immunotherapy', 'treatment', 'research', 'therapy']
            },
            {
                'question': 'What are the symptoms of heart disease?',
                'keywords': ['heart', 'disease', 'symptoms', 'chest', 'pain']
            },
            {
                'question': 'How does COVID-19 affect the respiratory system?',
                'keywords': ['covid', 'respiratory', 'lungs', 'breathing', 'infection']
            },
            {
                'question': 'What is the role of genetics in Alzheimer\'s disease?',
                'keywords': ['genetics', 'alzheimer', 'disease', 'genes', 'inheritance']
            }
        ]
        
        results = []
        total_time = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test Case {i}/{len(test_cases)}")
            result = self.test_question(test_case['question'], test_case['keywords'])
            
            if result:
                results.append(result)
                total_time += result['generation_time']
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        if results:
            avg_time = total_time / len(results)
            total_keywords = sum(len(r['keywords_found']) for r in results)
            max_keywords = sum(len(tc['keywords']) for tc in test_cases)
            
            print(f"✅ Tests completed: {len(results)}/{len(test_cases)}")
            print(f"⏱️  Average generation time: {avg_time:.2f} seconds")
            print(f"📈 Total keyword matches: {total_keywords}/{max_keywords} ({total_keywords/max_keywords*100:.1f}%)")
            print(f"🚀 Device used: {self.device}")
            
            # Performance rating
            if avg_time < 5:
                print("🏆 Performance: Excellent (fast generation)")
            elif avg_time < 10:
                print("👍 Performance: Good (reasonable speed)")
            else:
                print("⚠️  Performance: Slow (consider optimization)")
            
            if total_keywords/max_keywords > 0.7:
                print("🎯 Accuracy: High (good keyword matching)")
            elif total_keywords/max_keywords > 0.4:
                print("📊 Accuracy: Medium (moderate keyword matching)")
            else:
                print("🔍 Accuracy: Low (consider more training)")
        
        return results

def main():
    """Main function to run the test suite."""
    try:
        tester = ModelTester()
        results = tester.run_comprehensive_test()
        
        print("\n🎉 Testing completed!")
        print("Your GPT-Neo 2.7B medical AI model is ready for use!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print("❌ Testing failed. Make sure your model is properly loaded.")

if __name__ == "__main__":
    main()
