#!/usr/bin/env python3
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from tqdm import tqdm


class IbisRepoProcessor:
    def __init__(self, repo_path: str = "ibis", output_dir: str = "data/repo"):
        self.repo_path = Path(repo_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_data = {
            'functions': [],
            'classes': [],
            'docstrings': [],
            'test_examples': [],
            'error_patterns': []
        }
    
    def process(self):
        print(f"Processing Ibis repository at {self.repo_path}")
        
        python_files = list(self.repo_path.rglob("*.py"))
        notebook_files = list(self.repo_path.rglob("*.ipynb"))
        
        print(f"Found {len(python_files)} Python files and {len(notebook_files)} notebooks")
        
        for py_file in tqdm(python_files, desc="Processing Python files"):
            self._process_python_file(py_file)
        
        for nb_file in tqdm(notebook_files, desc="Processing notebooks"):
            self._process_notebook(nb_file)
        
        self._save_extracted_data()
        return self.extracted_data
    
    def _process_python_file(self, file_path: Path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            relative_path = file_path.relative_to(self.repo_path)
            is_test = 'test' in str(relative_path).lower()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._extract_function(node, str(relative_path), is_test)
                elif isinstance(node, ast.ClassDef):
                    self._extract_class(node, str(relative_path))
                elif isinstance(node, ast.Raise):
                    self._extract_error_pattern(node, str(relative_path))
        
        except Exception as e:
            pass
    
    def _extract_function(self, node: ast.FunctionDef, file_path: str, is_test: bool):
        func_data = {
            'name': node.name,
            'file': file_path,
            'type': 'test' if is_test else 'function',
            'docstring': ast.get_docstring(node),
            'signature': self._get_function_signature(node),
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
            'is_test': is_test
        }
        
        if is_test and node.name.startswith('test_'):
            func_data['test_name'] = node.name
            func_data['code'] = self._get_source_segment(node)
            self.extracted_data['test_examples'].append(func_data)
        else:
            self.extracted_data['functions'].append(func_data)
        
        if func_data['docstring']:
            self.extracted_data['docstrings'].append({
                'source': f"{file_path}::{node.name}",
                'docstring': func_data['docstring'],
                'type': 'function'
            })
    
    def _extract_class(self, node: ast.ClassDef, file_path: str):
        class_data = {
            'name': node.name,
            'file': file_path,
            'docstring': ast.get_docstring(node),
            'methods': [],
            'bases': [self._get_name(base) for base in node.bases]
        }
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_data = {
                    'name': item.name,
                    'docstring': ast.get_docstring(item),
                    'signature': self._get_function_signature(item),
                    'is_property': any(self._get_decorator_name(d) == 'property' 
                                     for d in item.decorator_list)
                }
                class_data['methods'].append(method_data)
        
        self.extracted_data['classes'].append(class_data)
        
        if class_data['docstring']:
            self.extracted_data['docstrings'].append({
                'source': f"{file_path}::{node.name}",
                'docstring': class_data['docstring'],
                'type': 'class'
            })
    
    def _extract_error_pattern(self, node: ast.Raise, file_path: str):
        if node.exc:
            error_type = None
            error_msg = None
            
            if isinstance(node.exc, ast.Call):
                error_type = self._get_name(node.exc.func)
                if node.exc.args and isinstance(node.exc.args[0], ast.Constant):
                    error_msg = node.exc.args[0].value
            elif isinstance(node.exc, ast.Name):
                error_type = node.exc.id
            
            if error_type:
                self.extracted_data['error_patterns'].append({
                    'file': file_path,
                    'error_type': error_type,
                    'message': error_msg,
                    'context': self._get_source_segment(node, lines=3)
                })
    
    def _get_function_signature(self, node: ast.FunctionDef) -> Dict:
        args = []
        for arg in node.args.args:
            arg_info = {'name': arg.arg}
            if arg.annotation:
                arg_info['type'] = self._get_annotation_string(arg.annotation)
            args.append(arg_info)
        
        return {
            'args': args,
            'returns': self._get_annotation_string(node.returns) if node.returns else None,
            'has_kwargs': node.args.kwarg is not None,
            'has_varargs': node.args.vararg is not None
        }
    
    def _get_annotation_string(self, annotation) -> str:
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif isinstance(annotation, ast.Subscript):
            base = self._get_name(annotation.value)
            if isinstance(annotation.slice, ast.Tuple):
                elts = [self._get_annotation_string(e) for e in annotation.slice.elts]
                return f"{base}[{', '.join(elts)}]"
            else:
                return f"{base}[{self._get_annotation_string(annotation.slice)}]"
        else:
            return 'Any'
    
    def _get_decorator_name(self, decorator) -> str:
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._get_name(decorator.value)}.{decorator.attr}"
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        return 'unknown'
    
    def _get_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return 'unknown'
    
    def _get_source_segment(self, node, lines: int = 10) -> str:
        return f"# Code from lines {node.lineno} to {node.end_lineno}"
    
    def _process_notebook(self, file_path: Path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            for cell in notebook.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    if 'ibis' in source.lower():
                        self.extracted_data['test_examples'].append({
                            'file': str(file_path.relative_to(self.repo_path)),
                            'type': 'notebook_example',
                            'code': source,
                            'outputs': cell.get('outputs', [])
                        })
        except Exception as e:
            pass
    
    def _save_extracted_data(self):
        functions_file = self.output_dir / 'functions.jsonl'
        with open(functions_file, 'w') as f:
            for item in self.extracted_data['functions']:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(self.extracted_data['functions'])} functions to {functions_file}")
        
        classes_file = self.output_dir / 'classes.jsonl'
        with open(classes_file, 'w') as f:
            for item in self.extracted_data['classes']:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(self.extracted_data['classes'])} classes to {classes_file}")
        
        tests_file = self.output_dir / 'test_examples.jsonl'
        with open(tests_file, 'w') as f:
            for item in self.extracted_data['test_examples']:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(self.extracted_data['test_examples'])} test examples to {tests_file}")
        
        docstrings_file = self.output_dir / 'docstrings.jsonl'
        with open(docstrings_file, 'w') as f:
            for item in self.extracted_data['docstrings']:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(self.extracted_data['docstrings'])} docstrings to {docstrings_file}")
        
        errors_file = self.output_dir / 'error_patterns.jsonl'
        with open(errors_file, 'w') as f:
            for item in self.extracted_data['error_patterns']:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(self.extracted_data['error_patterns'])} error patterns to {errors_file}")
        
        summary = {
            'total_functions': len(self.extracted_data['functions']),
            'total_classes': len(self.extracted_data['classes']),
            'total_test_examples': len(self.extracted_data['test_examples']),
            'total_docstrings': len(self.extracted_data['docstrings']),
            'total_error_patterns': len(self.extracted_data['error_patterns'])
        }
        
        summary_file = self.output_dir / 'processing_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_file}")


if __name__ == "__main__":
    processor = IbisRepoProcessor()
    data = processor.process()
    
    print(f"\nProcessing complete:")
    print(f"- Functions: {len(data['functions'])}")
    print(f"- Classes: {len(data['classes'])}")
    print(f"- Test examples: {len(data['test_examples'])}")
    print(f"- Docstrings: {len(data['docstrings'])}")
    print(f"- Error patterns: {len(data['error_patterns'])}")