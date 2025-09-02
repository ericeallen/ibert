#!/usr/bin/env python3
"""
iBERT model implementation using Devstral as base with LoRA fine-tuning
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    GenerationConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)


@dataclass
class IBertConfig:
    """Configuration for iBERT model"""
    base_model: str = "mistralai/Devstral-Small-2505"  # Updated to available model
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    task_type: str = "CAUSAL_LM"
    use_compiler_feedback: bool = True
    max_refinement_iterations: int = 3
    beam_size: int = 5
    temperature: float = 0.7
    max_length: int = 512
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


class IBertModel:
    """iBERT model with Devstral base and LoRA adapters"""
    
    def __init__(self, config: IBertConfig):
        self.config = config
        # Prioritize MPS (Metal) on Mac, then CUDA, then CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Metal Performance Shaders (MPS) for acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA GPU for acceleration")
        else:
            self.device = torch.device("cpu")
            print("Using CPU (no GPU acceleration available)")
        
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize base model and apply LoRA"""
        print(f"Loading base model: {self.config.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with appropriate dtype and device mapping
        if torch.backends.mps.is_available():
            # MPS-specific configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float32,  # MPS works better with float32
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)
        elif torch.cuda.is_available():
            # CUDA configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # CPU configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        
        # Configure LoRA
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            inference_mode=False
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, self.peft_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def generate(
        self,
        prompt: str,
        task_type: str = "code_completion",
        context: Optional[Dict] = None,
        validate_with_compiler: bool = True
    ) -> Dict:
        """
        Generate code with optional compiler validation
        
        Args:
            prompt: Input prompt/question
            task_type: Type of task (code_completion, ibis_to_sql, etc.)
            context: Additional context (schema, examples, etc.)
            validate_with_compiler: Whether to validate with Ibis compiler
            
        Returns:
            Dictionary with generated code and metadata
        """
        # Prepare input
        formatted_prompt = self._format_prompt(prompt, task_type, context)
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        # Generate with beam search
        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_length,
            num_beams=self.config.beam_size,
            temperature=self.config.temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode candidates
        candidates = []
        for seq in outputs.sequences:
            decoded = self.tokenizer.decode(
                seq[inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            candidates.append(decoded)
        
        # Validate with compiler if requested
        if validate_with_compiler and self.config.use_compiler_feedback:
            validated_candidates = self._validate_with_compiler(
                candidates,
                task_type
            )
            
            # Iterative refinement if all fail
            if not any(c["valid"] for c in validated_candidates):
                for iteration in range(self.config.max_refinement_iterations):
                    refined = self._refine_with_feedback(
                        prompt,
                        validated_candidates[0],
                        task_type,
                        context
                    )
                    if refined["valid"]:
                        return refined
            else:
                # Return first valid candidate
                for candidate in validated_candidates:
                    if candidate["valid"]:
                        return candidate
        
        # Return best candidate without validation
        return {
            "code": candidates[0],
            "candidates": candidates,
            "task_type": task_type,
            "validated": False
        }
    
    def _format_prompt(
        self,
        prompt: str,
        task_type: str,
        context: Optional[Dict]
    ) -> str:
        """Format prompt based on task type"""
        if task_type == "ibis_to_sql":
            template = "Convert the following Ibis expression to SQL:\n{prompt}"
        elif task_type == "code_completion":
            template = "Complete the following Ibis code:\n{prompt}"
        elif task_type == "error_resolution":
            template = "Fix the following Ibis error:\n{prompt}"
        elif task_type == "qa":
            template = "Answer the following question about Ibis:\n{prompt}"
        else:
            template = "{prompt}"
        
        formatted = template.format(prompt=prompt)
        
        # Add context if provided
        if context:
            if "schema" in context:
                formatted = f"Schema:\n{context['schema']}\n\n{formatted}"
            if "examples" in context:
                examples = "\n".join(context["examples"])
                formatted = f"Examples:\n{examples}\n\n{formatted}"
        
        return formatted
    
    def _validate_with_compiler(
        self,
        candidates: List[str],
        task_type: str
    ) -> List[Dict]:
        """Validate candidates with Ibis compiler"""
        import ibis
        
        validated = []
        for code in candidates:
            try:
                # Attempt to parse/compile the code
                if task_type in ["code_completion", "ibis_to_sql"]:
                    # Try to execute as Ibis expression
                    exec_globals = {"ibis": ibis}
                    exec(code, exec_globals)
                    validated.append({
                        "code": code,
                        "valid": True,
                        "error": None
                    })
                else:
                    # For other tasks, just check syntax
                    compile(code, "<string>", "exec")
                    validated.append({
                        "code": code,
                        "valid": True,
                        "error": None
                    })
            except Exception as e:
                validated.append({
                    "code": code,
                    "valid": False,
                    "error": str(e)
                })
        
        return validated
    
    def _refine_with_feedback(
        self,
        original_prompt: str,
        failed_attempt: Dict,
        task_type: str,
        context: Optional[Dict]
    ) -> Dict:
        """Refine generation based on compiler feedback"""
        error_msg = failed_attempt.get("error", "Unknown error")
        refinement_prompt = (
            f"{original_prompt}\n\n"
            f"Previous attempt failed with error:\n{error_msg}\n\n"
            f"Please fix the code:\n{failed_attempt['code']}"
        )
        
        return self.generate(
            refinement_prompt,
            task_type,
            context,
            validate_with_compiler=True
        )
    
    def save_adapter(self, save_path: str):
        """Save LoRA adapter weights"""
        if self.model:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"Adapter saved to {save_path}")
    
    def load_adapter(self, adapter_path: str):
        """Load LoRA adapter weights"""
        self.model = PeftModel.from_pretrained(
            self.model.base_model,
            adapter_path
        )
        print(f"Adapter loaded from {adapter_path}")


def create_ibert_model(
    base_model: str = "mistralai/Devstral-Small-1.1",
    lora_rank: int = 32,
    **kwargs
) -> IBertModel:
    """Factory function to create iBERT model"""
    config = IBertConfig(
        base_model=base_model,
        lora_rank=lora_rank,
        **kwargs
    )
    return IBertModel(config)


if __name__ == "__main__":
    # Test model initialization
    print("Initializing iBERT model with Devstral base...")
    model = create_ibert_model()
    
    # Test generation
    test_prompt = "Create an Ibis query to select all columns from a table called 'orders'"
    result = model.generate(
        test_prompt,
        task_type="code_completion",
        validate_with_compiler=False  # Skip validation for initial test
    )
    
    print(f"\nPrompt: {test_prompt}")
    print(f"Generated: {result['code']}")