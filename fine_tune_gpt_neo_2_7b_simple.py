#!/usr/bin/env python3
"""
GPT-Neo 2.7B Medical AI Fine-tuning Script (Simple)

This script fine-tunes EleutherAI GPT-Neo 2.7B on your medical research database.
Simple approach without LoRA to avoid gradient issues.
"""

import torch
import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from database import PubMedDatabase

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPTNeo27BSimpleFineTuner:
    def __init__(self, db_path="pubmed_data.db"):
        self.db = PubMedDatabase(db_path)
        self.device = self._detect_device()
        logger.info(f"Detected device: {self.device}")
    
    def _detect_device(self):
        """Detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def prepare_training_data(self, max_samples: Optional[int] = None):
        """Prepare training data from the medical research database."""
        logger.info("Preparing training data from medical research database...")
        
        # Get articles and trials
        articles = self.db.get_all_articles()
        trials = self.db.get_all_trials()
        
        if max_samples:
            # Use a representative sample
            articles = articles[:max_samples//2] if len(articles) > max_samples//2 else articles
            trials = trials[:max_samples//2] if len(trials) > max_samples//2 else trials
        
        training_texts = []
        
        # Process articles with medical format
        for article in articles:
            text = f"Medical Research Analysis:\n\n"
            text += f"Title: {article['title']}\n\n"
            
            if article['abstract']:
                text += f"Abstract: {article['abstract']}\n\n"
            
            if article.get('categories'):
                text += f"Categories: {', '.join(article['categories'])}\n\n"
            
            if article['authors']:
                text += f"Authors: {article['authors']}\n\n"
            
            if article['journal']:
                text += f"Journal: {article['journal']}\n\n"
            
            if article['publication_date']:
                text += f"Publication Date: {article['publication_date']}\n\n"
            
            text += f"PMID: {article['pmid']}\n\n"
            text += f"Analysis: This research article presents important findings in medical research. "
            if article['abstract']:
                text += f"The study focuses on {article['abstract'][:150]}... "
            text += "The research contributes to our understanding of medical science and may have implications for clinical practice. The findings could potentially inform treatment strategies and improve patient care. This study adds valuable knowledge to the medical field and may guide future research directions."
            
            training_texts.append(text)
        
        # Process clinical trials with medical format
        for trial in trials:
            text = f"Clinical Trial Analysis:\n\n"
            text += f"Title: {trial['title']}\n\n"
            
            if trial['brief_title']:
                text += f"Brief Title: {trial['brief_title']}\n\n"
            
            if trial['conditions']:
                text += f"Conditions: {trial['conditions']}\n\n"
            
            if trial['overall_status']:
                text += f"Status: {trial['overall_status']}\n\n"
            
            if trial['phase']:
                text += f"Phase: {trial['phase']}\n\n"
            
            if trial['interventions']:
                text += f"Interventions: {trial['interventions']}\n\n"
            
            if trial['primary_outcomes']:
                text += f"Primary Outcomes: {trial['primary_outcomes']}\n\n"
            
            if trial['start_date']:
                text += f"Start Date: {trial['start_date']}\n\n"
            
            text += f"NCT ID: {trial['nct_id']}\n\n"
            text += f"Analysis: This clinical trial is designed to investigate important medical interventions and outcomes. "
            if trial['conditions']:
                text += f"The study focuses on {trial['conditions'][:150]}... "
            text += "The trial aims to contribute to medical knowledge and potentially improve patient care. The results could inform clinical practice and treatment guidelines. This study represents an important step in advancing medical research and patient outcomes."
            
            training_texts.append(text)
        
        logger.info(f"Prepared {len(training_texts)} training examples")
        return training_texts
    
    def create_dataset(self, texts: List[str], tokenizer, max_length: int = 1024):
        """Create a dataset for training."""
        def tokenize_function(examples):
            # Tokenize the texts
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors=None
            )
            
            # For causal language modeling, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def setup_model_and_tokenizer(self, model_name: str):
        """Setup model and tokenizer."""
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for MPS compatibility
        )
        
        # Move to appropriate device
        if self.device == "mps":
            model = model.to("mps")
            logger.info("Model moved to MPS")
        elif self.device == "cuda":
            model = model.to("cuda")
            logger.info("Model moved to CUDA")
        else:
            logger.info("Model loaded on CPU")
        
        return model, tokenizer
    
    def fine_tune(self, model_name: str, max_samples: int = 1200):
        """Fine-tune the model."""
        logger.info(f"Starting GPT-Neo 2.7B fine-tuning...")
        
        try:
            # Load model and tokenizer
            model, tokenizer = self.setup_model_and_tokenizer(model_name)
            
            # Prepare training data
            training_texts = self.prepare_training_data(max_samples=max_samples)
            
            if not training_texts:
                logger.error("No training data found.")
                return False
            
            # Create dataset
            dataset = self.create_dataset(training_texts, tokenizer, max_length=1024)
            
            # Training arguments optimized for GPT-Neo 2.7B
            training_args = TrainingArguments(
                output_dir="./gpt-neo-2.7b-medical-finetuned",
                num_train_epochs=2,  # 2 epochs for good results
                per_device_train_batch_size=1,  # Small batch size for 2.7B model
                gradient_accumulation_steps=16,  # Large accumulation
                warmup_steps=100,
                learning_rate=2e-5,  # Lower learning rate for large model
                fp16=False,  # Disable fp16 for MPS
                bf16=False,  # Disable bf16 for MPS
                logging_steps=10,
                save_steps=200,
                save_total_limit=2,
                remove_unused_columns=False,
                eval_strategy="no",
                dataloader_pin_memory=False,
                gradient_checkpointing=True,
                dataloader_num_workers=0,
                report_to=None,
                max_grad_norm=1.0,
                lr_scheduler_type="cosine",
                save_strategy="steps",
                load_best_model_at_end=False,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )
            
            # Start training
            logger.info("Starting training...")
            trainer.train()
            
            # Save the fine-tuned model
            trainer.save_model()
            tokenizer.save_pretrained("./gpt-neo-2.7b-medical-finetuned")
            
            logger.info("Fine-tuning completed! Model saved to ./gpt-neo-2.7b-medical-finetuned")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            return False

def main():
    """Main function to fine-tune GPT-Neo 2.7B."""
    logger.info("🏥 GPT-Neo 2.7B Medical AI Fine-tuning")
    logger.info("=" * 60)
    
    # Check system capabilities
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    if device == "cpu":
        logger.warning("⚠️ CPU detected. GPT-Neo 2.7B will be slow on CPU.")
    elif device == "mps":
        logger.info("🍎 Apple Silicon detected. Using MPS acceleration.")
        logger.info("GPT-Neo 2.7B should work well with your 96GB RAM.")
    else:
        logger.info("🚀 CUDA detected. Using GPU acceleration.")
    
    # Use GPT-Neo 2.7B model
    model_name = "EleutherAI/gpt-neo-2.7B"
    logger.info(f"Selected model: {model_name}")
    
    # Initialize fine-tuner
    fine_tuner = GPTNeo27BSimpleFineTuner()
    
    # Determine training parameters
    max_samples = 1200  # Use 1200 samples for good results
    
    logger.info(f"Using {max_samples} training samples")
    
    # Start fine-tuning
    success = fine_tuner.fine_tune(
        model_name=model_name,
        max_samples=max_samples
    )
    
    if success:
        logger.info("🎉 Fine-tuning completed successfully!")
        logger.info("Your GPT-Neo 2.7B medical AI model is ready!")
        logger.info("Model saved to: ./gpt-neo-2.7b-medical-finetuned/")
        
        # Test the model
        logger.info("\n🧪 Testing the fine-tuned model...")
        try:
            from transformers import pipeline
            
            # Load the fine-tuned model
            generator = pipeline(
                "text-generation",
                model="./gpt-neo-2.7b-medical-finetuned",
                tokenizer="./gpt-neo-2.7b-medical-finetuned",
                device=0 if device == "cuda" else -1
            )
            
            # Test with a medical question
            test_prompt = "Medical Research Analysis:\n\nQuestion: What are the latest treatments for diabetes?\n\nAnalysis:"
            
            result = generator(
                test_prompt,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            logger.info("Test result:")
            logger.info(result[0]['generated_text'])
            
        except Exception as e:
            logger.warning(f"Could not test model: {e}")
        
    else:
        logger.error("❌ Fine-tuning failed. Check the logs above for details.")

if __name__ == "__main__":
    main()
