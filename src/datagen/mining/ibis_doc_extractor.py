"""Extract SQL→Ibis examples from Ibis documentation and tutorials."""

import re
from pathlib import Path
from typing import List, Dict, Any
import json


def extract_from_markdown(md_file: Path) -> List[Dict[str, Any]]:
    """Extract SQL and Ibis code blocks from markdown documentation.

    Parameters
    ----------
    md_file : Path
        Markdown file to parse

    Returns
    -------
    list of dict
        Extracted code examples
    """
    try:
        content = md_file.read_text()
    except Exception:
        return []

    examples = []

    # Pattern: SQL code block followed by Ibis code block
    # ```sql
    # SELECT ...
    # ```
    # ```python
    # table.select(...)
    # ```

    # Find all code blocks
    code_block_pattern = r'```(\w+)\n(.+?)```'
    code_blocks = list(re.finditer(code_block_pattern, content, re.DOTALL))

    # Look for SQL→Python pairs
    for i, match in enumerate(code_blocks):
        lang = match.group(1).lower()
        code = match.group(2).strip()

        if lang == "sql" and i + 1 < len(code_blocks):
            next_match = code_blocks[i + 1]
            next_lang = next_match.group(1).lower()
            next_code = next_match.group(2).strip()

            if next_lang == "python" and "ibis" in next_code.lower():
                examples.append({
                    "source": "markdown_doc",
                    "file": str(md_file),
                    "sql": code,
                    "ibis": next_code,
                })

    return examples


def extract_from_jupyter(notebook_path: Path) -> List[Dict[str, Any]]:
    """Extract examples from Jupyter notebooks.

    Parameters
    ----------
    notebook_path : Path
        Jupyter notebook file

    Returns
    -------
    list of dict
        Extracted examples
    """
    try:
        with open(notebook_path) as f:
            notebook = json.load(f)
    except Exception:
        return []

    examples = []
    cells = notebook.get("cells", [])

    for i, cell in enumerate(cells):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))

        # Look for cells containing .sql()
        if cell_type == "code" and ".sql(" in source:
            # Try to extract SQL string
            sql_match = re.search(r'\.sql\(["\'](.+?)["\']\)', source, re.DOTALL)
            if sql_match:
                sql = sql_match.group(1)

                # Check next cells for Ibis equivalent
                if i + 1 < len(cells):
                    next_cell = cells[i + 1]
                    next_source = "".join(next_cell.get("source", []))

                    examples.append({
                        "source": "jupyter_notebook",
                        "file": str(notebook_path),
                        "sql": sql,
                        "ibis_context": next_source,
                    })

    return examples


def mine_ibis_documentation(ibis_repo_path: Path) -> List[Dict[str, Any]]:
    """Mine examples from Ibis documentation.

    Parameters
    ----------
    ibis_repo_path : Path
        Path to cloned Ibis repository

    Returns
    -------
    list of dict
        Mined examples
    """
    examples = []

    # Scan markdown documentation
    docs_dir = ibis_repo_path / "docs"
    if docs_dir.exists():
        md_files = list(docs_dir.rglob("*.md"))
        print(f"Scanning {len(md_files)} markdown files...")

        for md_file in md_files:
            examples.extend(extract_from_markdown(md_file))

    # Scan Jupyter notebooks
    notebooks = list(ibis_repo_path.rglob("*.ipynb"))
    if notebooks:
        print(f"Scanning {len(notebooks)} Jupyter notebooks...")

        for notebook in notebooks:
            examples.extend(extract_from_jupyter(notebook))

    return examples


if __name__ == "__main__":
    ibis_repo = Path("data/mining/repos/ibis")

    if not ibis_repo.exists():
        print("Ibis repository not found. Run github_miner.py first.")
    else:
        examples = mine_ibis_documentation(ibis_repo)
        print(f"Found {len(examples)} documentation examples")

        output_path = Path("data/mining/ibis_docs_mined.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

        print(f"Saved to {output_path}")
