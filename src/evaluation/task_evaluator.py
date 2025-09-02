#!/usr/bin/env python3
"""
Task-based evaluation framework for iBERT model
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import numpy as np
from tqdm import tqdm
import wandb
import ibis
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu


@dataclass
class EvaluationMetrics:
    """Metrics for evaluation"""
    task_name: str
    total_examples: int = 0
    compilation_success: int = 0
    first_pass_success: int = 0
    avg_iterations: float = 0.0
    execution_success: int = 0
    exact_match: int = 0
    bleu_score: float = 0.0
    rouge_l: float = 0.0
    avg_generation_time: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @property
    def compilation_rate(self) -> float:
        return self.compilation_success / max(self.total_examples, 1)
    
    @property
    def first_pass_rate(self) -> float:
        return self.first_pass_success / max(self.total_examples, 1)
    
    @property
    def execution_rate(self) -> float:
        return self.execution_success / max(self.total_examples, 1)
    
    @property
    def exact_match_rate(self) -> float:
        return self.exact_match / max(self.total_examples, 1)


class TaskEvaluator:
    """Evaluator for task-based model assessment"""
    
    def __init__(
        self,
        model,
        data_dir: str = "data/corpus",
        use_compiler_validation: bool = True,
        use_wandb: bool = True,
        wandb_project: str = "ibert-eval"
    ):
        self.model = model
        self.data_dir = Path(data_dir)
        self.use_compiler_validation = use_compiler_validation
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.init(project=wandb_project)
        
        # Initialize metrics
        self.metrics = {}
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    
    def evaluate_all_tasks(self, split: str = "test") -> Dict[str, EvaluationMetrics]:
        """Evaluate model on all tasks"""
        tasks = [
            "ibis_to_sql",
            "code_completion",
            "qa_pairs",
            "error_solutions",
            "function_docs"
        ]
        
        results = {}
        for task in tasks:
            print(f"\nEvaluating {task}...")
            metrics = self.evaluate_task(task, split)
            results[task] = metrics
            
            # Log to W&B
            if self.use_wandb:
                wandb.log({f"{task}_{k}": v for k, v in metrics.to_dict().items()})
        
        # Generate summary
        self._generate_summary(results)
        
        return results
    
    def evaluate_task(
        self,
        task_name: str,
        split: str = "test"
    ) -> EvaluationMetrics:
        """Evaluate model on a specific task"""
        # Load test data
        test_data = self._load_task_data(task_name, split)
        if not test_data:
            print(f"No test data found for {task_name}")
            return EvaluationMetrics(task_name=task_name)
        
        metrics = EvaluationMetrics(
            task_name=task_name,
            total_examples=len(test_data)
        )
        
        # Evaluate based on task type
        if task_name == "ibis_to_sql":
            metrics = self._evaluate_ibis_to_sql(test_data, metrics)
        elif task_name == "code_completion":
            metrics = self._evaluate_code_completion(test_data, metrics)
        elif task_name == "qa_pairs":
            metrics = self._evaluate_qa(test_data, metrics)
        elif task_name == "error_solutions":
            metrics = self._evaluate_error_resolution(test_data, metrics)
        elif task_name == "function_docs":
            metrics = self._evaluate_function_docs(test_data, metrics)
        
        return metrics
    
    def _load_task_data(self, task_name: str, split: str) -> List[Dict]:
        """Load test data for a task"""
        file_path = self.data_dir / f"{task_name}_{split}.jsonl"
        if not file_path.exists():
            return []
        
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        return data
    
    def _evaluate_ibis_to_sql(
        self,
        test_data: List[Dict],
        metrics: EvaluationMetrics
    ) -> EvaluationMetrics:
        """Evaluate Ibis to SQL translation"""
        predictions = []
        references = []
        
        for item in tqdm(test_data, desc="Evaluating Ibis→SQL"):
            start_time = time.time()
            
            # Generate SQL
            result = self.model.generate(
                item.get("ibis_code", ""),
                task_type="ibis_to_sql",
                validate_with_compiler=self.use_compiler_validation
            )
            
            generation_time = time.time() - start_time
            metrics.avg_generation_time += generation_time
            
            generated_sql = result.get("code", "")
            expected_sql = item.get("sql_code", "")
            
            predictions.append(generated_sql)
            references.append(expected_sql)
            
            # Check exact match (normalized)
            if self._normalize_sql(generated_sql) == self._normalize_sql(expected_sql):
                metrics.exact_match += 1
            
            # Check compilation
            if self._validate_sql(generated_sql):
                metrics.compilation_success += 1
                if result.get("iterations", 1) == 1:
                    metrics.first_pass_success += 1
            
            # Track iterations
            metrics.avg_iterations += result.get("iterations", 1)
        
        # Calculate BLEU score
        if predictions and references:
            metrics.bleu_score = corpus_bleu(predictions, [references]).score
        
        # Calculate averages
        metrics.avg_generation_time /= max(len(test_data), 1)
        metrics.avg_iterations /= max(len(test_data), 1)
        
        return metrics
    
    def _evaluate_code_completion(
        self,
        test_data: List[Dict],
        metrics: EvaluationMetrics
    ) -> EvaluationMetrics:
        """Evaluate code completion task"""
        for item in tqdm(test_data, desc="Evaluating code completion"):
            start_time = time.time()
            
            # Generate completion
            result = self.model.generate(
                item.get("code", "")[:len(item.get("code", ""))//2],  # Use first half
                task_type="code_completion",
                validate_with_compiler=self.use_compiler_validation
            )
            
            generation_time = time.time() - start_time
            metrics.avg_generation_time += generation_time
            
            # Validate generated code
            if self.use_compiler_validation:
                validation = self._validate_ibis_code(result.get("code", ""))
                if validation["valid"]:
                    metrics.compilation_success += 1
                    if result.get("iterations", 1) == 1:
                        metrics.first_pass_success += 1
                else:
                    # Track error types
                    error_type = validation.get("error_type", "unknown")
                    metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1
            
            metrics.avg_iterations += result.get("iterations", 1)
        
        metrics.avg_generation_time /= max(len(test_data), 1)
        metrics.avg_iterations /= max(len(test_data), 1)
        
        return metrics
    
    def _evaluate_qa(
        self,
        test_data: List[Dict],
        metrics: EvaluationMetrics
    ) -> EvaluationMetrics:
        """Evaluate Q&A task"""
        rouge_scores = []
        
        for item in tqdm(test_data, desc="Evaluating Q&A"):
            start_time = time.time()
            
            # Generate answer
            result = self.model.generate(
                item.get("question", ""),
                task_type="qa",
                validate_with_compiler=False
            )
            
            generation_time = time.time() - start_time
            metrics.avg_generation_time += generation_time
            
            generated_answer = result.get("code", "")
            expected_answer = item.get("answer", "")
            
            # Calculate ROUGE score
            scores = self.rouge_scorer.score(expected_answer, generated_answer)
            rouge_scores.append(scores['rougeL'].fmeasure)
        
        # Average ROUGE-L score
        if rouge_scores:
            metrics.rouge_l = np.mean(rouge_scores)
        
        metrics.avg_generation_time /= max(len(test_data), 1)
        
        return metrics
    
    def _evaluate_error_resolution(
        self,
        test_data: List[Dict],
        metrics: EvaluationMetrics
    ) -> EvaluationMetrics:
        """Evaluate error resolution task"""
        for item in tqdm(test_data, desc="Evaluating error resolution"):
            start_time = time.time()
            
            # Generate fix
            error_prompt = f"{item.get('error_type', '')}: {item.get('error_message', '')}"
            result = self.model.generate(
                error_prompt,
                task_type="error_resolution",
                validate_with_compiler=self.use_compiler_validation
            )
            
            generation_time = time.time() - start_time
            metrics.avg_generation_time += generation_time
            
            # Check if fix resolves the error
            if self.use_compiler_validation:
                # Simulate applying fix and checking if error is resolved
                fix_valid = self._validate_error_fix(
                    item.get("context", ""),
                    result.get("code", ""),
                    item.get("error_type", "")
                )
                if fix_valid:
                    metrics.execution_success += 1
                    if result.get("iterations", 1) == 1:
                        metrics.first_pass_success += 1
            
            metrics.avg_iterations += result.get("iterations", 1)
        
        metrics.avg_generation_time /= max(len(test_data), 1)
        metrics.avg_iterations /= max(len(test_data), 1)
        
        return metrics
    
    def _evaluate_function_docs(
        self,
        test_data: List[Dict],
        metrics: EvaluationMetrics
    ) -> EvaluationMetrics:
        """Evaluate function documentation task"""
        rouge_scores = []
        
        for item in tqdm(test_data, desc="Evaluating documentation"):
            start_time = time.time()
            
            # Generate documentation
            func_signature = f"def {item.get('function', 'func')}():"
            result = self.model.generate(
                func_signature,
                task_type="qa",  # Treat as Q&A task
                validate_with_compiler=False
            )
            
            generation_time = time.time() - start_time
            metrics.avg_generation_time += generation_time
            
            generated_doc = result.get("code", "")
            expected_doc = item.get("docstring", "")
            
            # Calculate ROUGE score
            scores = self.rouge_scorer.score(expected_doc, generated_doc)
            rouge_scores.append(scores['rougeL'].fmeasure)
            
            # Check if docstring is valid Python
            if self._validate_docstring(generated_doc):
                metrics.compilation_success += 1
        
        # Average ROUGE-L score
        if rouge_scores:
            metrics.rouge_l = np.mean(rouge_scores)
        
        metrics.avg_generation_time /= max(len(test_data), 1)
        
        return metrics
    
    def _validate_ibis_code(self, code: str) -> Dict[str, Any]:
        """Validate Ibis code"""
        try:
            # Try to compile the code
            compile(code, "<string>", "exec")
            
            # Try to execute in Ibis context
            exec_globals = {"ibis": ibis}
            exec(code, exec_globals)
            
            return {"valid": True, "error": None, "error_type": None}
        except SyntaxError as e:
            return {"valid": False, "error": str(e), "error_type": "syntax"}
        except NameError as e:
            return {"valid": False, "error": str(e), "error_type": "name"}
        except Exception as e:
            return {"valid": False, "error": str(e), "error_type": type(e).__name__}
    
    def _validate_sql(self, sql: str) -> bool:
        """Basic SQL validation"""
        try:
            # Basic syntax check
            sql = sql.strip()
            if not sql:
                return False
            
            # Check for common SQL keywords
            sql_upper = sql.upper()
            has_select = "SELECT" in sql_upper
            has_from = "FROM" in sql_upper
            
            return has_select or "INSERT" in sql_upper or "UPDATE" in sql_upper
        except:
            return False
    
    def _validate_error_fix(
        self,
        context: str,
        fix: str,
        error_type: str
    ) -> bool:
        """Validate if error fix resolves the issue"""
        try:
            # Combine context and fix
            combined_code = f"{context}\n{fix}"
            
            # Try to compile
            compile(combined_code, "<string>", "exec")
            
            # No error raised means fix might work
            return True
        except:
            return False
    
    def _validate_docstring(self, docstring: str) -> bool:
        """Validate if docstring is valid"""
        try:
            # Check if it's a valid string
            if not docstring or not isinstance(docstring, str):
                return False
            
            # Check basic docstring format
            lines = docstring.strip().split('\n')
            return len(lines) > 0 and len(docstring) > 10
        except:
            return False
    
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison"""
        if not sql:
            return ""
        
        # Basic normalization
        sql = sql.strip().upper()
        sql = ' '.join(sql.split())  # Normalize whitespace
        sql = sql.replace(';', '')  # Remove trailing semicolon
        
        return sql
    
    def _generate_summary(self, results: Dict[str, EvaluationMetrics]):
        """Generate evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        overall_metrics = {
            "total_examples": 0,
            "compilation_success": 0,
            "first_pass_success": 0,
            "avg_iterations": [],
            "avg_generation_time": []
        }
        
        for task_name, metrics in results.items():
            print(f"\n{task_name.upper()}:")
            print(f"  Total examples: {metrics.total_examples}")
            print(f"  Compilation rate: {metrics.compilation_rate:.2%}")
            print(f"  First-pass rate: {metrics.first_pass_rate:.2%}")
            print(f"  Avg iterations: {metrics.avg_iterations:.2f}")
            print(f"  Avg generation time: {metrics.avg_generation_time:.2f}s")
            
            if metrics.bleu_score > 0:
                print(f"  BLEU score: {metrics.bleu_score:.2f}")
            if metrics.rouge_l > 0:
                print(f"  ROUGE-L: {metrics.rouge_l:.3f}")
            if metrics.exact_match_rate > 0:
                print(f"  Exact match rate: {metrics.exact_match_rate:.2%}")
            
            # Aggregate metrics
            overall_metrics["total_examples"] += metrics.total_examples
            overall_metrics["compilation_success"] += metrics.compilation_success
            overall_metrics["first_pass_success"] += metrics.first_pass_success
            overall_metrics["avg_iterations"].append(metrics.avg_iterations)
            overall_metrics["avg_generation_time"].append(metrics.avg_generation_time)
        
        # Overall summary
        print("\n" + "-"*60)
        print("OVERALL:")
        print(f"  Total examples: {overall_metrics['total_examples']}")
        
        if overall_metrics["total_examples"] > 0:
            overall_compilation_rate = overall_metrics["compilation_success"] / overall_metrics["total_examples"]
            overall_first_pass_rate = overall_metrics["first_pass_success"] / overall_metrics["total_examples"]
            print(f"  Overall compilation rate: {overall_compilation_rate:.2%}")
            print(f"  Overall first-pass rate: {overall_first_pass_rate:.2%}")
        
        if overall_metrics["avg_iterations"]:
            print(f"  Average iterations: {np.mean(overall_metrics['avg_iterations']):.2f}")
        if overall_metrics["avg_generation_time"]:
            print(f"  Average generation time: {np.mean(overall_metrics['avg_generation_time']):.2f}s")
        
        print("="*60)
        
        # Log overall metrics to W&B
        if self.use_wandb:
            wandb.log({
                "overall_compilation_rate": overall_compilation_rate,
                "overall_first_pass_rate": overall_first_pass_rate,
                "overall_avg_iterations": np.mean(overall_metrics["avg_iterations"]),
                "overall_avg_generation_time": np.mean(overall_metrics["avg_generation_time"])
            })


def main():
    """Main evaluation script"""
    from src.models.devstral_lora import create_ibert_model
    
    # Load model
    print("Loading model...")
    model = create_ibert_model()
    
    # Load fine-tuned adapter if available
    adapter_path = "models/ibert_v1"
    if Path(adapter_path).exists():
        print(f"Loading adapter from {adapter_path}")
        model.load_adapter(adapter_path)
    
    # Create evaluator
    evaluator = TaskEvaluator(
        model=model,
        data_dir="data/corpus",
        use_compiler_validation=True,
        use_wandb=True,
        wandb_project="ibert-eval"
    )
    
    # Run evaluation
    results = evaluator.evaluate_all_tasks(split="test")
    
    # Save results
    results_file = Path("evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(
            {k: v.to_dict() for k, v in results.items()},
            f,
            indent=2
        )
    print(f"\nResults saved to {results_file}")
    
    if evaluator.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()