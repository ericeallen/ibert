#!/usr/bin/env python3
"""
Cone-packing context assembly for focused code context
"""
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import networkx as nx


@dataclass
class Symbol:
    """Represents a code symbol"""
    name: str
    type: str  # function, class, method, variable
    file: str
    line: int
    definition: Optional[str] = None
    references: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ContextCone:
    """Represents a focused context cone"""
    target: Symbol
    upstream: List[Symbol]  # Dependencies
    downstream: List[Symbol]  # Usages
    lateral: List[Symbol]  # Related helpers
    examples: List[Dict]
    total_tokens: int = 0


class ConePacker:
    """
    Surgical context assembler that computes dependency cones
    for focused context instead of full repository dumps
    """
    
    def __init__(
        self,
        repo_path: str,
        symbol_index_path: Optional[str] = None,
        max_tokens: int = 30000
    ):
        self.repo_path = Path(repo_path)
        self.max_tokens = max_tokens
        self.symbol_graph = nx.DiGraph()
        self.symbol_index = {}
        
        if symbol_index_path:
            self.load_symbol_index(symbol_index_path)
    
    def load_symbol_index(self, index_path: str):
        """Load pre-computed symbol index from Serena/Semgrep"""
        with open(index_path, 'r') as f:
            for line in f:
                symbol_data = json.loads(line)
                symbol = Symbol(**symbol_data)
                self.symbol_index[symbol.name] = symbol
                
                # Build dependency graph
                for dep in symbol.dependencies:
                    self.symbol_graph.add_edge(dep, symbol.name)
                
                for ref in symbol.references:
                    self.symbol_graph.add_edge(symbol.name, ref)
    
    def compute_cone(
        self,
        target: str,
        depth: int = 2,
        include_tests: bool = True,
        include_docs: bool = True
    ) -> ContextCone:
        """
        Compute dependency cone for target symbol
        
        Args:
            target: Target symbol name or query
            depth: How many levels to traverse in the graph
            include_tests: Include related test files
            include_docs: Include documentation
            
        Returns:
            ContextCone with focused context
        """
        # Find target symbol
        target_symbol = self._find_symbol(target)
        if not target_symbol:
            raise ValueError(f"Symbol not found: {target}")
        
        # Compute upstream dependencies
        upstream = self._get_upstream_symbols(target_symbol.name, depth)
        
        # Compute downstream usages
        downstream = self._get_downstream_symbols(target_symbol.name, depth)
        
        # Find lateral/related symbols
        lateral = self._get_lateral_symbols(target_symbol.name)
        
        # Gather examples
        examples = self._gather_examples(
            target_symbol,
            include_tests,
            include_docs
        )
        
        # Create cone
        cone = ContextCone(
            target=target_symbol,
            upstream=upstream,
            downstream=downstream,
            lateral=lateral,
            examples=examples
        )
        
        # Calculate token count
        cone.total_tokens = self._estimate_tokens(cone)
        
        # Prune if exceeds max_tokens
        if cone.total_tokens > self.max_tokens:
            cone = self._prune_cone(cone)
        
        return cone
    
    def _find_symbol(self, query: str) -> Optional[Symbol]:
        """Find symbol by name or query"""
        # Direct match
        if query in self.symbol_index:
            return self.symbol_index[query]
        
        # Fuzzy match
        for name, symbol in self.symbol_index.items():
            if query.lower() in name.lower():
                return symbol
        
        return None
    
    def _get_upstream_symbols(
        self,
        target: str,
        depth: int
    ) -> List[Symbol]:
        """Get symbols that target depends on"""
        upstream = set()
        
        # BFS to find dependencies
        visited = set()
        queue = [(target, 0)]
        
        while queue:
            current, level = queue.pop(0)
            if level >= depth or current in visited:
                continue
            
            visited.add(current)
            
            # Get predecessors (dependencies)
            for pred in self.symbol_graph.predecessors(current):
                if pred in self.symbol_index:
                    upstream.add(pred)
                    queue.append((pred, level + 1))
        
        return [self.symbol_index[s] for s in upstream if s in self.symbol_index]
    
    def _get_downstream_symbols(
        self,
        target: str,
        depth: int
    ) -> List[Symbol]:
        """Get symbols that depend on target"""
        downstream = set()
        
        # BFS to find usages
        visited = set()
        queue = [(target, 0)]
        
        while queue:
            current, level = queue.pop(0)
            if level >= depth or current in visited:
                continue
            
            visited.add(current)
            
            # Get successors (usages)
            for succ in self.symbol_graph.successors(current):
                if succ in self.symbol_index:
                    downstream.add(succ)
                    queue.append((succ, level + 1))
        
        return [self.symbol_index[s] for s in downstream if s in self.symbol_index]
    
    def _get_lateral_symbols(self, target: str) -> List[Symbol]:
        """Get related helper symbols"""
        lateral = []
        
        if target not in self.symbol_index:
            return lateral
        
        target_symbol = self.symbol_index[target]
        target_file = target_symbol.file
        
        # Find symbols in same file
        for name, symbol in self.symbol_index.items():
            if symbol.file == target_file and name != target:
                lateral.append(symbol)
                if len(lateral) >= 5:  # Limit lateral symbols
                    break
        
        return lateral
    
    def _gather_examples(
        self,
        target: Symbol,
        include_tests: bool,
        include_docs: bool
    ) -> List[Dict]:
        """Gather relevant examples and tests"""
        examples = []
        
        # Find test files
        if include_tests:
            test_pattern = f"test_{target.name}"
            for name, symbol in self.symbol_index.items():
                if test_pattern in name.lower() or \
                   (symbol.type == "test" and target.name in str(symbol.references)):
                    examples.append({
                        "type": "test",
                        "name": name,
                        "code": symbol.definition or "",
                        "file": symbol.file
                    })
        
        # Find documentation
        if include_docs:
            # Look for docstrings or markdown files
            if target.definition and "\"\"\"" in target.definition:
                # Extract docstring
                lines = target.definition.split("\n")
                in_docstring = False
                docstring_lines = []
                for line in lines:
                    if '"""' in line:
                        in_docstring = not in_docstring
                        if not in_docstring:
                            break
                    elif in_docstring:
                        docstring_lines.append(line)
                
                if docstring_lines:
                    examples.append({
                        "type": "docstring",
                        "name": f"{target.name}_doc",
                        "code": "\n".join(docstring_lines),
                        "file": target.file
                    })
        
        return examples[:10]  # Limit examples
    
    def _estimate_tokens(self, cone: ContextCone) -> int:
        """Estimate token count for cone"""
        # Rough estimation: 1 token ≈ 4 characters
        total_chars = 0
        
        # Target symbol
        if cone.target.definition:
            total_chars += len(cone.target.definition)
        
        # Dependencies
        for symbol in cone.upstream + cone.downstream + cone.lateral:
            if symbol.definition:
                total_chars += len(symbol.definition)
        
        # Examples
        for example in cone.examples:
            total_chars += len(example.get("code", ""))
        
        return total_chars // 4
    
    def _prune_cone(self, cone: ContextCone) -> ContextCone:
        """Prune cone to fit within token limit"""
        # Priority: target > upstream > examples > downstream > lateral
        
        # Keep target
        pruned = ContextCone(
            target=cone.target,
            upstream=[],
            downstream=[],
            lateral=[],
            examples=[]
        )
        
        remaining_tokens = self.max_tokens
        
        # Add target
        if cone.target.definition:
            target_tokens = len(cone.target.definition) // 4
            remaining_tokens -= target_tokens
        
        # Add upstream dependencies
        for symbol in cone.upstream:
            if symbol.definition:
                symbol_tokens = len(symbol.definition) // 4
                if remaining_tokens >= symbol_tokens:
                    pruned.upstream.append(symbol)
                    remaining_tokens -= symbol_tokens
        
        # Add examples
        for example in cone.examples[:5]:  # Limit examples
            example_tokens = len(example.get("code", "")) // 4
            if remaining_tokens >= example_tokens:
                pruned.examples.append(example)
                remaining_tokens -= example_tokens
        
        # Add downstream if space
        for symbol in cone.downstream[:3]:  # Limit downstream
            if symbol.definition:
                symbol_tokens = len(symbol.definition) // 4
                if remaining_tokens >= symbol_tokens:
                    pruned.downstream.append(symbol)
                    remaining_tokens -= symbol_tokens
        
        pruned.total_tokens = self.max_tokens - remaining_tokens
        return pruned
    
    def format_cone(self, cone: ContextCone, format_type: str = "markdown") -> str:
        """Format cone for LLM consumption"""
        if format_type == "markdown":
            return self._format_markdown(cone)
        elif format_type == "json":
            return self._format_json(cone)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    def _format_markdown(self, cone: ContextCone) -> str:
        """Format cone as markdown"""
        sections = []
        
        # Target
        sections.append(f"## Target Symbol: {cone.target.name}")
        sections.append(f"Type: {cone.target.type}")
        sections.append(f"File: {cone.target.file}")
        if cone.target.definition:
            sections.append(f"```python\n{cone.target.definition}\n```")
        
        # Dependencies
        if cone.upstream:
            sections.append("\n## Dependencies")
            for symbol in cone.upstream:
                sections.append(f"### {symbol.name}")
                if symbol.definition:
                    sections.append(f"```python\n{symbol.definition}\n```")
        
        # Examples
        if cone.examples:
            sections.append("\n## Examples")
            for example in cone.examples:
                sections.append(f"### {example['name']} ({example['type']})")
                sections.append(f"```python\n{example['code']}\n```")
        
        # Usages
        if cone.downstream:
            sections.append("\n## Usages")
            for symbol in cone.downstream:
                sections.append(f"- {symbol.name} in {symbol.file}")
        
        return "\n".join(sections)
    
    def _format_json(self, cone: ContextCone) -> str:
        """Format cone as JSON"""
        data = {
            "target": {
                "name": cone.target.name,
                "type": cone.target.type,
                "file": cone.target.file,
                "definition": cone.target.definition
            },
            "upstream": [
                {
                    "name": s.name,
                    "type": s.type,
                    "file": s.file,
                    "definition": s.definition
                }
                for s in cone.upstream
            ],
            "downstream": [
                {"name": s.name, "file": s.file}
                for s in cone.downstream
            ],
            "examples": cone.examples,
            "total_tokens": cone.total_tokens
        }
        return json.dumps(data, indent=2)


if __name__ == "__main__":
    # Example usage
    packer = ConePacker(repo_path=".", max_tokens=10000)
    
    # Simulate loading some symbols
    packer.symbol_index = {
        "Table.select": Symbol(
            name="Table.select",
            type="method",
            file="ibis/table.py",
            line=100,
            definition="def select(self, *columns): ...",
            dependencies=["Table", "Column"],
            references=["test_select", "example_query"]
        ),
        "Table": Symbol(
            name="Table",
            type="class",
            file="ibis/table.py",
            line=10,
            definition="class Table: ...",
            dependencies=["BaseTable"],
            references=["Table.select", "Table.filter"]
        )
    }
    
    # Build graph
    packer.symbol_graph.add_edge("Table", "Table.select")
    packer.symbol_graph.add_edge("Column", "Table.select")
    
    # Compute cone
    cone = packer.compute_cone("Table.select", depth=2)
    
    # Format and print
    formatted = packer.format_cone(cone, "markdown")
    print(formatted)
    print(f"\nTotal tokens: {cone.total_tokens}")