#!/usr/bin/env python3
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from tqdm import tqdm


class TrainingCorpusBuilder:
    def __init__(self, data_dir: str = "data", output_dir: str = "data/corpus"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.corpus = {
            'ibis_to_sql': [],
            'qa_pairs': [],
            'error_solutions': [],
            'function_docs': [],
            'code_completion': []
        }
        
        random.seed(42)
    
    def build(self):
        print("Building training corpus...")
        
        self._process_documentation()
        self._process_repository_data()
        self._generate_synthetic_examples()
        self._create_splits()
        
        return self.corpus
    
    def _process_documentation(self):
        docs_file = self.data_dir / 'docs' / 'ibis_docs.jsonl'
        if not docs_file.exists():
            print(f"Warning: {docs_file} not found")
            return
        
        with open(docs_file, 'r') as f:
            docs = [json.loads(line) for line in f]
        
        for doc in tqdm(docs, desc="Processing documentation"):
            chunks = self._chunk_text(doc['content_markdown'])
            
            for chunk in chunks:
                if len(chunk.strip()) > 50:
                    self.corpus['qa_pairs'].append({
                        'question': f"Explain this Ibis concept: {doc['title']}",
                        'answer': chunk,
                        'source': doc['url'],
                        'type': 'documentation'
                    })
            
            for example in doc.get('code_examples', []):
                if example['language'] == 'python' and 'ibis' in example['code'].lower():
                    self.corpus['code_completion'].append({
                        'context': doc['title'],
                        'code': example['code'],
                        'source': doc['url']
                    })
    
    def _process_repository_data(self):
        functions_file = self.data_dir / 'repo' / 'functions.jsonl'
        if functions_file.exists():
            with open(functions_file, 'r') as f:
                functions = [json.loads(line) for line in f]
            
            for func in tqdm(functions[:1000], desc="Processing functions"):
                if func.get('docstring'):
                    self.corpus['function_docs'].append({
                        'function': func['name'],
                        'signature': func.get('signature', {}),
                        'docstring': func['docstring'],
                        'source': func['file']
                    })
        
        test_file = self.data_dir / 'repo' / 'test_examples.jsonl'
        if test_file.exists():
            with open(test_file, 'r') as f:
                tests = [json.loads(line) for line in f]
            
            for test in tqdm(tests[:500], desc="Processing test examples"):
                if test.get('code'):
                    self.corpus['code_completion'].append({
                        'context': f"Test: {test.get('test_name', 'unknown')}",
                        'code': test['code'],
                        'source': test['file'],
                        'type': 'test'
                    })
        
        errors_file = self.data_dir / 'repo' / 'error_patterns.jsonl'
        if errors_file.exists():
            with open(errors_file, 'r') as f:
                errors = [json.loads(line) for line in f]
            
            for error in errors[:200]:
                if error.get('message'):
                    self.corpus['error_solutions'].append({
                        'error_type': error['error_type'],
                        'error_message': error['message'],
                        'context': error.get('context', ''),
                        'solution': f"Handle {error['error_type']} by checking the conditions",
                        'source': error['file']
                    })
    
    def _generate_synthetic_examples(self):
        common_patterns = [
            {
                'ibis': 't.select(t.column1, t.column2)',
                'sql': 'SELECT column1, column2 FROM t',
                'description': 'Basic column selection'
            },
            {
                'ibis': 't.filter(t.column > 10)',
                'sql': 'SELECT * FROM t WHERE column > 10',
                'description': 'Simple filter condition'
            },
            {
                'ibis': 't.group_by("category").aggregate(count=t.count())',
                'sql': 'SELECT category, COUNT(*) as count FROM t GROUP BY category',
                'description': 'Group by with aggregation'
            },
            {
                'ibis': 't1.join(t2, t1.id == t2.id)',
                'sql': 'SELECT * FROM t1 JOIN t2 ON t1.id = t2.id',
                'description': 'Inner join on ID'
            },
            {
                'ibis': 't.order_by(t.date.desc()).limit(10)',
                'sql': 'SELECT * FROM t ORDER BY date DESC LIMIT 10',
                'description': 'Order by with limit'
            }
        ]
        
        for pattern in common_patterns:
            self.corpus['ibis_to_sql'].append({
                'ibis_code': pattern['ibis'],
                'sql_code': pattern['sql'],
                'description': pattern['description'],
                'type': 'synthetic'
            })
            
            self.corpus['qa_pairs'].append({
                'question': f"How do I {pattern['description'].lower()} in Ibis?",
                'answer': f"Use this pattern: {pattern['ibis']}",
                'type': 'synthetic'
            })
    
    def _chunk_text(self, text: str, chunk_size: int = 200) -> List[str]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def _create_splits(self):
        print("Creating train/val/test splits...")
        
        for task_type, examples in self.corpus.items():
            if not examples:
                continue
            
            random.shuffle(examples)
            
            n = len(examples)
            train_size = int(0.8 * n)
            val_size = int(0.1 * n)
            
            train_data = examples[:train_size]
            val_data = examples[train_size:train_size + val_size]
            test_data = examples[train_size + val_size:]
            
            train_file = self.output_dir / f'{task_type}_train.jsonl'
            with open(train_file, 'w') as f:
                for item in train_data:
                    f.write(json.dumps(item) + '\n')
            
            val_file = self.output_dir / f'{task_type}_val.jsonl'
            with open(val_file, 'w') as f:
                for item in val_data:
                    f.write(json.dumps(item) + '\n')
            
            test_file = self.output_dir / f'{task_type}_test.jsonl'
            with open(test_file, 'w') as f:
                for item in test_data:
                    f.write(json.dumps(item) + '\n')
            
            print(f"{task_type}: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        stats = {
            'total_examples': sum(len(v) for v in self.corpus.values()),
            'task_distribution': {k: len(v) for k, v in self.corpus.items()},
            'splits': {
                'train': 0.8,
                'validation': 0.1,
                'test': 0.1
            }
        }
        
        stats_file = self.output_dir / 'corpus_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nCorpus statistics saved to {stats_file}")
        print(f"Total examples: {stats['total_examples']}")


if __name__ == "__main__":
    builder = TrainingCorpusBuilder()
    corpus = builder.build()
    
    print("\nCorpus building complete!")
    for task_type, examples in corpus.items():
        print(f"- {task_type}: {len(examples)} examples")