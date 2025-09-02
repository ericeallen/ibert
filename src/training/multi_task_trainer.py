#!/usr/bin/env python3
"""
Multi-task training pipeline for iBERT model
"""
import json
import torch
import wandb
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from tqdm import tqdm
import numpy as np


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task training"""
    data_dir: str = "data/corpus"
    output_dir: str = "models/ibert"
    task_weights: Dict[str, float] = None
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 500
    max_seq_length: int = 512
    fp16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    use_wandb: bool = True
    wandb_project: str = "ibert"
    wandb_run_name: Optional[str] = None
    
    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                "ibis_to_sql": 2.0,
                "code_completion": 1.0,
                "qa_pairs": 1.0,
                "error_solutions": 1.5,
                "function_docs": 1.0
            }


class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        task_weights: Optional[Dict[str, float]] = None,
        max_seq_length: int = 512
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_seq_length = max_seq_length
        self.task_weights = task_weights or {}
        
        # Load all task data
        self.data = []
        self.task_labels = []
        self._load_data()
    
    def _load_data(self):
        """Load data for all tasks"""
        task_files = {
            "ibis_to_sql": f"ibis_to_sql_{self.split}.jsonl",
            "code_completion": f"code_completion_{self.split}.jsonl",
            "qa_pairs": f"qa_pairs_{self.split}.jsonl",
            "error_solutions": f"error_solutions_{self.split}.jsonl",
            "function_docs": f"function_docs_{self.split}.jsonl"
        }
        
        for task_name, filename in task_files.items():
            file_path = self.data_dir / filename
            if not file_path.exists():
                print(f"Warning: {file_path} not found")
                continue
            
            with open(file_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    formatted = self._format_item(item, task_name)
                    if formatted:
                        self.data.append(formatted)
                        self.task_labels.append(task_name)
            
            print(f"Loaded {len([l for l in self.task_labels if l == task_name])} examples for {task_name}")
        
        # Apply task weights by duplicating examples
        if self.task_weights and self.split == "train":
            self._apply_task_weights()
    
    def _format_item(self, item: Dict, task_name: str) -> Optional[Dict]:
        """Format item based on task type"""
        if task_name == "ibis_to_sql":
            if "ibis_code" in item and "sql_code" in item:
                return {
                    "input": f"Convert this Ibis code to SQL:\n{item['ibis_code']}",
                    "output": item["sql_code"],
                    "task": task_name
                }
        
        elif task_name == "code_completion":
            if "code" in item:
                # Split code for completion task
                lines = item["code"].split("\n")
                if len(lines) > 2:
                    split_point = len(lines) // 2
                    input_code = "\n".join(lines[:split_point])
                    output_code = "\n".join(lines[split_point:])
                    return {
                        "input": f"Complete this Ibis code:\n{input_code}",
                        "output": output_code,
                        "task": task_name
                    }
        
        elif task_name == "qa_pairs":
            if "question" in item and "answer" in item:
                return {
                    "input": item["question"],
                    "output": item["answer"],
                    "task": task_name
                }
        
        elif task_name == "error_solutions":
            if "error_message" in item and "solution" in item:
                return {
                    "input": f"Fix this Ibis error:\n{item['error_type']}: {item['error_message']}",
                    "output": item["solution"],
                    "task": task_name
                }
        
        elif task_name == "function_docs":
            if "function" in item and "docstring" in item:
                sig = item.get("signature", {})
                args = ", ".join([a["name"] for a in sig.get("args", [])])
                return {
                    "input": f"Document this function:\ndef {item['function']}({args}):",
                    "output": item["docstring"],
                    "task": task_name
                }
        
        return None
    
    def _apply_task_weights(self):
        """Apply task weights by duplicating examples"""
        weighted_data = []
        weighted_labels = []
        
        for task_name, weight in self.task_weights.items():
            task_indices = [i for i, l in enumerate(self.task_labels) if l == task_name]
            
            # Duplicate based on weight
            repeat_count = int(weight)
            for _ in range(repeat_count):
                for idx in task_indices:
                    weighted_data.append(self.data[idx])
                    weighted_labels.append(self.task_labels[idx])
            
            # Handle fractional weights
            if weight % 1 > 0:
                sample_size = int(len(task_indices) * (weight % 1))
                sampled_indices = np.random.choice(task_indices, sample_size, replace=False)
                for idx in sampled_indices:
                    weighted_data.append(self.data[idx])
                    weighted_labels.append(self.task_labels[idx])
        
        self.data = weighted_data
        self.task_labels = weighted_labels
        
        # Shuffle
        indices = np.random.permutation(len(self.data))
        self.data = [self.data[i] for i in indices]
        self.task_labels = [self.task_labels[i] for i in indices]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class MultiTaskTrainer:
    """Trainer for multi-task learning"""
    
    def __init__(
        self,
        model,
        tokenizer,
        config: MultiTaskConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Initialize W&B if requested
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config)
            )
    
    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets"""
        train_dataset = MultiTaskDataset(
            self.config.data_dir,
            split="train",
            task_weights=self.config.task_weights,
            max_seq_length=self.config.max_seq_length
        )
        
        val_dataset = MultiTaskDataset(
            self.config.data_dir,
            split="val",
            max_seq_length=self.config.max_seq_length
        )
        
        return train_dataset, val_dataset
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function for multi-task batches"""
        inputs = []
        labels = []
        task_labels = []
        
        for item in batch:
            # Combine input and output
            text = f"{item['input']}\n\nAnswer: {item['output']}"
            inputs.append(text)
            labels.append(item['output'])
            task_labels.append(item['task'])
        
        # Tokenize
        encodings = self.tokenizer(
            inputs,
            truncation=True,
            padding=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )
        
        # Create labels for language modeling
        label_encodings = self.tokenizer(
            labels,
            truncation=True,
            padding=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings.input_ids,
            "attention_mask": encodings.attention_mask,
            "labels": label_encodings.input_ids,
            "task_labels": task_labels
        }
    
    def compute_metrics(self, eval_pred) -> Dict:
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        
        # Simple accuracy for now
        predictions = np.argmax(predictions, axis=-1)
        accuracy = np.mean(predictions == labels)
        
        metrics = {
            "accuracy": accuracy,
            "perplexity": np.exp(eval_pred.loss) if hasattr(eval_pred, 'loss') else 0
        }
        
        return metrics
    
    def train(self):
        """Run training"""
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if self.config.use_wandb else "none",
            run_name=self.config.wandb_run_name
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model.model,  # Access the underlying model
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save final model
        print(f"Saving model to {self.config.output_dir}")
        self.model.save_adapter(self.config.output_dir)
        
        # Log final metrics
        if self.config.use_wandb:
            wandb.log({"training_complete": True})
            wandb.finish()
        
        return trainer


def main():
    """Main training script"""
    from src.models.devstral_lora import create_ibert_model
    
    # Configuration
    config = MultiTaskConfig(
        data_dir="data/corpus",
        output_dir="models/ibert_v1",
        num_epochs=3,
        use_wandb=True,
        wandb_run_name="ibert_multi_task_v1"
    )
    
    # Create model
    print("Initializing model...")
    model = create_ibert_model()
    
    # Create trainer
    trainer = MultiTaskTrainer(
        model=model,
        tokenizer=model.tokenizer,
        config=config
    )
    
    # Train
    trainer.train()
    
    print("Training complete!")


if __name__ == "__main__":
    main()