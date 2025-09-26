#!/usr/bin/env python3
"""
Medical AI Assistant using GPT-Neo 2.7B Fine-tuned Model

This assistant uses your fine-tuned GPT-Neo 2.7B model to answer medical questions
and search your PubMed database.
"""

import torch
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from database import PubMedDatabase
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalAIAssistant:
    def __init__(self, model_path="./gpt-neo-2.7b-medical-finetuned", db_path="pubmed_data.db"):
        self.db = PubMedDatabase(db_path)
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        logger.info(f"Loading medical AI model from {model_path}")
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
        
        logger.info("Medical AI Assistant ready!")
    
    def search_database(self, query, limit=5):
        """Search the medical database for relevant articles and trials."""
        try:
            # Search articles
            articles = self.db.search_articles(query, limit=limit//2)
            trials = self.db.search_trials(query, limit=limit//2)
            
            results = []
            for article in articles:
                results.append({
                    'type': 'article',
                    'title': article['title'],
                    'abstract': article['abstract'][:200] + "..." if article['abstract'] and len(article['abstract']) > 200 else article['abstract'],
                    'pmid': article['pmid'],
                    'journal': article['journal'],
                    'date': article['publication_date']
                })
            
            for trial in trials:
                results.append({
                    'type': 'trial',
                    'title': trial['title'],
                    'conditions': trial['conditions'],
                    'status': trial['overall_status'],
                    'phase': trial['phase'],
                    'nct_id': trial['nct_id']
                })
            
            return results
        except Exception as e:
            logger.error(f"Database search error: {e}")
            return []
    
    def generate_response(self, question, context=""):
        """Generate a response using the fine-tuned model."""
        try:
            # Create a medical analysis prompt
            prompt = f"Medical Research Analysis:\n\nQuestion: {question}\n\n"
            
            if context:
                prompt += f"Context: {context}\n\n"
            
            prompt += "Analysis:"
            
            # Generate response
            result = self.generator(
                prompt,
                max_length=len(prompt.split()) + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            
            # Extract the generated text
            generated_text = result[0]['generated_text']
            
            # Extract just the analysis part
            if "Analysis:" in generated_text:
                analysis = generated_text.split("Analysis:")[-1].strip()
            else:
                analysis = generated_text[len(prompt):].strip()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def answer_question(self, question):
        """Answer a medical question using both database search and AI generation."""
        logger.info(f"Processing question: {question}")
        
        # Search database for relevant information
        search_results = self.search_database(question, limit=3)
        
        # Create context from search results
        context = ""
        if search_results:
            context = "Relevant research found:\n"
            for result in search_results[:2]:  # Use top 2 results
                if result['type'] == 'article':
                    context += f"- {result['title']} (PMID: {result['pmid']})\n"
                    if result['abstract']:
                        context += f"  {result['abstract']}\n"
                else:
                    context += f"- Clinical Trial: {result['title']} (NCT: {result['nct_id']})\n"
                    if result['conditions']:
                        context += f"  Conditions: {result['conditions']}\n"
        
        # Generate AI response
        ai_response = self.generate_response(question, context)
        
        return {
            'question': question,
            'ai_response': ai_response,
            'search_results': search_results,
            'context_used': bool(context)
        }
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        print("\n🏥 Medical AI Assistant - GPT-Neo 2.7B")
        print("=" * 50)
        print("Ask me any medical question! Type 'quit' to exit.")
        print("Examples:")
        print("- What are the latest treatments for diabetes?")
        print("- Tell me about cancer immunotherapy research")
        print("- What clinical trials are available for Alzheimer's?")
        print()
        
        while True:
            try:
                question = input("🤔 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Stay healthy!")
                    break
                
                if not question:
                    continue
                
                print("\n🔍 Searching medical database and generating response...")
                
                # Get answer
                result = self.answer_question(question)
                
                # Display results
                print(f"\n📋 Question: {result['question']}")
                print(f"\n🤖 AI Analysis:")
                print(result['ai_response'])
                
                if result['search_results']:
                    print(f"\n📚 Relevant Research Found:")
                    for i, res in enumerate(result['search_results'][:3], 1):
                        if res['type'] == 'article':
                            print(f"{i}. Article: {res['title']}")
                            print(f"   PMID: {res['pmid']}")
                            if res['journal']:
                                print(f"   Journal: {res['journal']}")
                        else:
                            print(f"{i}. Clinical Trial: {res['title']}")
                            print(f"   NCT ID: {res['nct_id']}")
                            if res['status']:
                                print(f"   Status: {res['status']}")
                        print()
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Stay healthy!")
                break
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                print("❌ Sorry, I encountered an error. Please try again.")

def main():
    """Main function to run the medical AI assistant."""
    try:
        # Initialize the assistant
        assistant = MedicalAIAssistant()
        
        # Start interactive chat
        assistant.interactive_chat()
        
    except Exception as e:
        logger.error(f"Failed to start assistant: {e}")
        print("❌ Failed to start the Medical AI Assistant.")
        print("Make sure your fine-tuned model is available at ./gpt-neo-2.7b-medical-finetuned/")

if __name__ == "__main__":
    main()
