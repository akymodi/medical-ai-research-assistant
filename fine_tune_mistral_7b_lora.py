#!/usr/bin/env python3
"""
Mistral 7B Medical AI Fine-tuning Script with LoRA

This script fine-tunes Mistral 7B using LoRA (Low-Rank Adaptation) on your medical research database.
LoRA allows efficient fine-tuning with much lower memory requirements.
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
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from database import PubMedDatabase

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Mistral7BLoRAFineTuner:
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
            text = f"<s>[INST] Analyze this medical research article: [/INST]\n\n"
            text += f"**Title:** {article['title']}\n\n"
            
            if article['abstract']:
                text += f"**Abstract:** {article['abstract']}\n\n"
            
            if article.get('categories'):
                text += f"**Categories:** {', '.join(article['categories'])}\n\n"
            
            if article['authors']:
                text += f"**Authors:** {article['authors']}\n\n"
            
            if article['journal']:
                text += f"**Journal:** {article['journal']}\n\n"
            
            if article['publication_date']:
                text += f"**Publication Date:** {article['publication_date']}\n\n"
            
            text += f"**PMID:** {article['pmid']}\n\n"
            text += f"**Analysis:** This research article presents important findings in medical research. "
            if article['abstract']:
                text += f"The study focuses on {article['abstract'][:150]}... "
            text += "The research contributes to our understanding of medical science and may have implications for clinical practice. The findings could potentially inform treatment strategies and improve patient care. This study adds valuable knowledge to the medical field and may guide future research directions.</s>"
            
            training_texts.append(text)
        
        # Process clinical trials with medical format
        for trial in trials:
            text = f"<s>[INST] Analyze this clinical trial: [/INST]\n\n"
            text += f"**Title:** {trial['title']}\n\n"
            
            if trial['brief_title']:
                text += f"**Brief Title:** {trial['brief_title']}\n\n"
            
            if trial['conditions']:
                text += f"**Conditions:** {trial['conditions']}\n\n"
            
            if trial['overall_status']:
                text += f"**Status:** {trial['overall_status']}\n\n"
            
            if trial['phase']:
                text += f"**Phase:** {trial['phase']}\n\n"
            
            if trial['interventions']:
                text += f"**Interventions:** {trial['interventions']}\n\n"
            
            if trial['primary_outcomes']:
                text += f"**Primary Outcomes:** {trial['primary_outcomes']}\n\n"
            
            if trial['start_date']:
                text += f"**Start Date:** {trial['start_date']}\n\n"
            
            text += f"**NCT ID:** {trial['nct_id']}\n\n"
            text += f"**Analysis:** This clinical trial is designed to investigate important medical interventions and outcomes. "
            if trial['conditions']:
                text += f"The study focuses on {trial['conditions'][:150]}... "
            text += "The trial aims to contribute to medical knowledge and potentially improve patient care. The results could inform clinical practice and treatment guidelines. This study represents an important step in advancing medical research and patient outcomes.</s>"
            
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
                padding=False,  # No padding for LoRA
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
    
    def setup_lora_config(self):
        """Setup LoRA configuration for Mistral 7B."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Rank of adaptation
            lora_alpha=32,  # LoRA scaling parameter
            lora_dropout=0.1,  # LoRA dropout
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Mistral attention modules
            bias="none",
        )
        return lora_config
    
    def setup_model_and_tokenizer(self, model_name: str):
        """Setup model and tokenizer with LoRA."""
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantization for memory efficiency (only for CUDA)
        quantization_config = None
        if self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float32,  # Use float32 for MPS compatibility
            device_map="auto" if self.device == "cuda" else None,
        )
        
        # Move to appropriate device if not using device_map
        if self.device == "mps" and quantization_config is None:
            model = model.to("mps")
            logger.info("Model moved to MPS")
        elif self.device == "cpu" and quantization_config is None:
            logger.info("Model loaded on CPU")
        
        # Setup LoRA
        lora_config = self.setup_lora_config()
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model, tokenizer
    
    def fine_tune(self, model_name: str, max_samples: int = 1000):
        """Fine-tune the model using LoRA."""
        logger.info(f"Starting Mistral 7B LoRA fine-tuning...")
        
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
            
            # Training arguments optimized for LoRA
            training_args = TrainingArguments(
                output_dir="./mistral-7b-medical-lora",
                num_train_epochs=3,  # More epochs for LoRA
                per_device_train_batch_size=2,  # Small batch size
                gradient_accumulation_steps=8,  # Gradient accumulation
                warmup_steps=50,
                learning_rate=2e-4,  # Higher learning rate for LoRA
                fp16=False,  # Disable fp16 for MPS
                bf16=False,  # Disable bf16 for MPS
                logging_steps=10,
                save_steps=100,
                save_total_limit=3,
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
                optim="adamw_torch",
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )
            
            # Start training
            logger.info("Starting LoRA training...")
            trainer.train()
            
            # Save the fine-tuned model
            trainer.save_model()
            tokenizer.save_pretrained("./mistral-7b-medical-lora")
            
            logger.info("LoRA fine-tuning completed! Model saved to ./mistral-7b-medical-lora")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during LoRA fine-tuning: {e}")
            return False

def main():
    """Main function to fine-tune Mistral 7B with LoRA."""
    logger.info("🏥 Mistral 7B Medical AI LoRA Fine-tuning")
    logger.info("=" * 60)
    
    # Check system capabilities
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    if device == "cpu":
        logger.warning("⚠️ CPU detected. Mistral 7B will be very slow on CPU.")
    elif device == "mps":
        logger.info("🍎 Apple Silicon detected. Using MPS acceleration.")
        logger.info("Mistral 7B with LoRA should work well with your 96GB RAM.")
    else:
        logger.info("🚀 CUDA detected. Using GPU acceleration with quantization.")
    
    # Use Mistral 7B model
    model_name = "mistralai/Mistral-7B-v0.1"
    logger.info(f"Selected model: {model_name}")
    
    # Initialize fine-tuner
    fine_tuner = Mistral7BLoRAFineTuner()
    
    # Determine training parameters
    max_samples = 1000  # Use 1000 samples for LoRA training
    
    logger.info(f"Using {max_samples} training samples with LoRA")
    
    # Start fine-tuning
    success = fine_tuner.fine_tune(
        model_name=model_name,
        max_samples=max_samples
    )
    
    if success:
        logger.info("🎉 LoRA fine-tuning completed successfully!")
        logger.info("Your Mistral 7B medical AI model is ready!")
        logger.info("Model saved to: ./mistral-7b-medical-lora/")
        
        # Test the model
        logger.info("\n🧪 Testing the fine-tuned model...")
        try:
            from transformers import pipeline
            
            # Load the fine-tuned model
            generator = pipeline(
                "text-generation",
                model="./mistral-7b-medical-lora",
                tokenizer="./mistral-7b-medical-lora",
                device=0 if device == "cuda" else -1
            )
            
            # Test with a medical question
            test_prompt = "<s>[INST] What are the latest treatments for diabetes? [/INST]\n\n"
            
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
        logger.error("❌ LoRA fine-tuning failed. Check the logs above for details.")

if __name__ == "__main__":
    main()
