#!/usr/bin/env python3
"""
Training script for iBERT with Devstral from Hugging Face
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.devstral_lora import create_ibert_model
from src.training.multi_task_trainer import MultiTaskTrainer, MultiTaskConfig

def main():
    print("=" * 60)
    print("iBERT TRAINING WITH DEVSTRAL")
    print("=" * 60)
    
    # Training configuration
    config = MultiTaskConfig(
        data_dir="data/corpus",
        output_dir="models/ibert_devstral_v1",
        num_epochs=1,  # Start with 1 epoch
        batch_size=2,  # Small batch size for 24B model
        gradient_accumulation_steps=4,  # Effective batch size of 8
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=200,
        eval_steps=100,
        fp16=False,  # MPS doesn't support fp16 well
        use_wandb=False,  # Disable for initial test
        gradient_checkpointing=True  # Enable to save memory
    )
    
    print("\nTraining Configuration:")
    print(f"  Model: mistralai/Mistral-7B-Instruct-v0.3")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Output dir: {config.output_dir}")
    
    print("\nAttempting to load Mistral-7B-v0.3 from Hugging Face...")
    print("This will download the model if not cached locally.")
    print("Model size: ~7B parameters")
    print("Note: Using Mistral-7B as Devstral requires vLLM, not Transformers")
    
    try:
        # Create model with Mistral-7B v0.3 (latest version with access)
        # Note: Devstral requires vLLM, using Mistral-7B for Transformers compatibility
        model = create_ibert_model(
            base_model="mistralai/Mistral-7B-Instruct-v0.3",
            lora_rank=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Mistral attention layers
        )
        
        print("\n✅ Model loaded successfully!")
        
        # Create trainer
        trainer = MultiTaskTrainer(
            model=model,
            tokenizer=model.tokenizer,
            config=config
        )
        
        # Run training
        print("\nStarting training...")
        trainer.train()
        
        print("\n✅ Training complete!")
        print(f"Model saved to: {config.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nPossible issues:")
        print("1. Model requires authentication - log in with: huggingface-cli login")
        print("2. Model requires accepting terms at: https://huggingface.co/mistralai/Devstral-Small-2505")
        print("3. Insufficient memory (24B model requires ~32GB+ RAM)")
        print("4. Network issues downloading the model")
        raise

if __name__ == "__main__":
    main()