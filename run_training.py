#!/usr/bin/env python3
"""
Quick training script for initial iBERT model run
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.devstral_lora import create_ibert_model
from src.training.multi_task_trainer import MultiTaskTrainer, MultiTaskConfig

def main():
    print("=" * 60)
    print("iBERT INITIAL TRAINING RUN")
    print("=" * 60)
    
    # Use minimal config for quick test
    config = MultiTaskConfig(
        data_dir="data/corpus",
        output_dir="models/ibert_v1",
        num_epochs=1,  # Just 1 epoch for initial test
        batch_size=4,  # Small batch for memory
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        warmup_steps=50,
        logging_steps=5,
        save_steps=100,
        eval_steps=50,
        fp16=False,  # MPS doesn't support fp16 well
        use_wandb=False,  # Skip W&B for initial test
        gradient_checkpointing=False  # Disable for speed
    )
    
    print("\nTraining Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Output dir: {config.output_dir}")
    
    # Note: Devstral model would require API access or local download
    # For testing, we'll use a smaller model
    print("\n⚠️  Note: Using GPT2 for testing instead of Devstral")
    print("  (Devstral requires Mistral API access or manual download)")
    
    # Create model with smaller test config
    # GPT2 uses different layer names: c_attn for attention, c_proj for projection
    model = create_ibert_model(
        base_model="gpt2",  # Use GPT2 for testing
        lora_rank=16,  # Smaller rank for quick test
        target_modules=["c_attn", "c_proj"]  # GPT2 attention layers
    )
    
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

if __name__ == "__main__":
    main()