"""Extract SQL→Ibis training examples from documentation and Jupyter notebooks.

This module parses various documentation formats to find SQL→Ibis code pairs:
- Markdown (.md) and Quarto (.qmd) documents
- Jupyter notebooks (.ipynb)
- Python code blocks with SQL strings
- Side-by-side SQL and Ibis examples

These extracted pairs serve as high-quality training data since they come
from official documentation and tutorials.
"""

import json
import re
from pathlib import Path
from typing import Any, NamedTuple

# Regex patterns for extracting code blocks
PATTERN_QUARTO_PYTHON = r"```\{python\}(.+?)```"
PATTERN_CODE_BLOCK = r"```(\w+)\n(.+?)```"
PATTERN_SQL_CALL = r'\.sql\(\s*["\'](.+?)["\']\s*\)'
PATTERN_SQL_STRING = r'["\']SELECT\s+.+?["\']'

# Keywords for validation
SQL_KEYWORD = "SELECT"
IBIS_KEYWORD = "ibis"


class DocumentExample(NamedTuple):
    """A code example extracted from documentation.

    Attributes
    ----------
    source_type : str
        Type of extraction (e.g., 'quarto_doc', 'markdown_doc')
    file_path : str
        Source documentation file path
    sql_code : str
        SQL query string
    ibis_code : str or None
        Corresponding Ibis code, if available
    context : str or None
        Additional context from surrounding code
    """

    source_type: str
    file_path: str
    sql_code: str
    ibis_code: str | None = None
    context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for JSON serialization.

        Returns
        -------
        dict
            Example as dictionary
        """
        result = {
            "source": self.source_type,
            "file": self.file_path,
            "sql": self.sql_code,
        }

        if self.ibis_code:
            result["ibis"] = self.ibis_code

        if self.context:
            context_key = "context" if self.ibis_code else "ibis_context"
            result[context_key] = self.context

        return result


class MarkdownExtractor:
    """Extracts SQL→Ibis examples from Markdown and Quarto documents."""

    def __init__(self, file_path: Path):
        """Initialize extractor for a documentation file.

        Parameters
        ----------
        file_path : Path
            Markdown or Quarto file to parse
        """
        self.file_path = file_path
        self.content = self._read_file_safely()

    def _read_file_safely(self) -> str | None:
        """Read file content with error handling.

        Returns
        -------
        str or None
            File content, or None if read failed
        """
        try:
            return self.file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def extract_examples(self) -> list[DocumentExample]:
        """Extract all SQL→Ibis examples from the document.

        Returns
        -------
        list of DocumentExample
            All discovered examples
        """
        if self.content is None:
            return []

        examples = []

        # Extract from different patterns
        examples.extend(self._extract_quarto_examples())
        examples.extend(self._extract_sequential_blocks())
        examples.extend(self._extract_python_sql_strings())

        return examples

    def _extract_quarto_examples(self) -> list[DocumentExample]:
        """Extract SQL from Quarto Python code blocks.

        Pattern: ```{python}
                 expr = t.sql("SELECT ...")
                 ```

        Returns
        -------
        list of DocumentExample
            Extracted examples
        """
        examples: list[DocumentExample] = []

        if self.content is None:
            return examples

        for match in re.finditer(PATTERN_QUARTO_PYTHON, self.content, re.DOTALL):
            code_block = match.group(1).strip()

            # Look for .sql() calls within the block
            for sql_match in re.finditer(PATTERN_SQL_CALL, code_block, re.DOTALL):
                sql_code = sql_match.group(1).strip()

                if self._is_valid_sql(sql_code):
                    examples.append(
                        DocumentExample(
                            source_type="quarto_doc",
                            file_path=str(self.file_path),
                            sql_code=sql_code,
                            context=code_block,
                        )
                    )

        return examples

    def _extract_sequential_blocks(self) -> list[DocumentExample]:
        """Extract SQL blocks followed by Python/Ibis blocks.

        Looks for sequential code blocks where SQL is followed by
        Python code containing 'ibis'.

        Returns
        -------
        list of DocumentExample
            Extracted examples
        """
        examples: list[DocumentExample] = []

        if self.content is None:
            return examples

        code_blocks = list(re.finditer(PATTERN_CODE_BLOCK, self.content, re.DOTALL))

        for index, match in enumerate(code_blocks):
            language = match.group(1).lower()
            code = match.group(2).strip()

            # Check if this is a SQL block followed by a Python block
            if language == "sql" and index + 1 < len(code_blocks):
                next_match = code_blocks[index + 1]
                next_language = next_match.group(1).lower()
                next_code = next_match.group(2).strip()

                if next_language == "python" and IBIS_KEYWORD in next_code.lower():
                    examples.append(
                        DocumentExample(
                            source_type="markdown_doc",
                            file_path=str(self.file_path),
                            sql_code=code,
                            ibis_code=next_code,
                        )
                    )

        return examples

    def _extract_python_sql_strings(self) -> list[DocumentExample]:
        """Extract SQL strings from Python code blocks.

        Returns
        -------
        list of DocumentExample
            Extracted examples
        """
        examples: list[DocumentExample] = []

        if self.content is None:
            return examples

        for match in re.finditer(PATTERN_CODE_BLOCK, self.content, re.DOTALL):
            language = match.group(1).lower()
            code = match.group(2).strip()

            if language in ["python", "py"]:
                # Search for SQL SELECT statements in strings
                for sql_match in re.finditer(PATTERN_SQL_STRING, code, re.DOTALL | re.IGNORECASE):
                    sql_code = sql_match.group(0).strip("\"'")

                    examples.append(
                        DocumentExample(
                            source_type="python_code_block",
                            file_path=str(self.file_path),
                            sql_code=sql_code,
                            context=code,
                        )
                    )

        return examples

    def _is_valid_sql(self, code: str) -> bool:
        """Check if code appears to be a SQL query.

        Parameters
        ----------
        code : str
            Code to validate

        Returns
        -------
        bool
            True if appears to be SQL
        """
        return SQL_KEYWORD in code.upper()


class JupyterExtractor:
    """Extracts SQL→Ibis examples from Jupyter notebooks."""

    def __init__(self, notebook_path: Path):
        """Initialize extractor for a Jupyter notebook.

        Parameters
        ----------
        notebook_path : Path
            Jupyter notebook file to parse
        """
        self.notebook_path = notebook_path
        self.notebook_data = self._load_notebook()

    def _load_notebook(self) -> dict[str, Any] | None:
        """Load and parse the Jupyter notebook JSON.

        Returns
        -------
        dict or None
            Notebook data, or None if load failed
        """
        try:
            with open(self.notebook_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                return None
        except (OSError, json.JSONDecodeError):
            return None

    def extract_examples(self) -> list[DocumentExample]:
        """Extract SQL→Ibis examples from notebook cells.

        Returns
        -------
        list of DocumentExample
            All discovered examples
        """
        if self.notebook_data is None:
            return []

        examples = []
        cells = self.notebook_data.get("cells", [])

        for index, cell in enumerate(cells):
            cell_examples = self._extract_from_cell(cell, index, cells)
            examples.extend(cell_examples)

        return examples

    def _extract_from_cell(
        self, cell: dict[str, Any], cell_index: int, all_cells: list[dict[str, Any]]
    ) -> list[DocumentExample]:
        """Extract examples from a single notebook cell.

        Parameters
        ----------
        cell : dict
            Cell data
        cell_index : int
            Index of this cell
        all_cells : list of dict
            All cells in notebook

        Returns
        -------
        list of DocumentExample
            Extracted examples from this cell
        """
        examples = []

        cell_type = cell.get("cell_type")
        source_lines = cell.get("source", [])
        source_text = "".join(source_lines)

        # Look for code cells containing .sql() calls
        if cell_type == "code" and ".sql(" in source_text:
            sql_code = self._extract_sql_from_source(source_text)

            if sql_code:
                # Get context from next cell if available
                context = self._get_next_cell_context(cell_index, all_cells)

                examples.append(
                    DocumentExample(
                        source_type="jupyter_notebook",
                        file_path=str(self.notebook_path),
                        sql_code=sql_code,
                        context=context,
                    )
                )

        return examples

    def _extract_sql_from_source(self, source: str) -> str | None:
        """Extract SQL code from cell source.

        Parameters
        ----------
        source : str
            Cell source code

        Returns
        -------
        str or None
            Extracted SQL code, or None if not found
        """
        sql_match = re.search(PATTERN_SQL_CALL, source, re.DOTALL)

        if sql_match:
            return sql_match.group(1)

        return None

    def _get_next_cell_context(
        self, current_index: int, all_cells: list[dict[str, Any]]
    ) -> str | None:
        """Get source from the next cell for context.

        Parameters
        ----------
        current_index : int
            Current cell index
        all_cells : list of dict
            All cells

        Returns
        -------
        str or None
            Next cell's source, or None if not available
        """
        if current_index + 1 < len(all_cells):
            next_cell = all_cells[current_index + 1]
            next_source_lines = next_cell.get("source", [])
            return "".join(next_source_lines)

        return None


class DocumentationMiner:
    """Coordinates mining of examples from documentation directories."""

    def __init__(self, repository_path: Path):
        """Initialize miner for a repository.

        Parameters
        ----------
        repository_path : Path
            Root path of repository to mine
        """
        self.repository_path = repository_path
        self.docs_directory = repository_path / "docs"

    def mine_all_documentation(self) -> list[dict[str, Any]]:
        """Mine examples from all documentation sources.

        Returns
        -------
        list of dict
            All extracted examples as dictionaries
        """
        examples = []

        # Mine markdown/Quarto documents
        examples.extend(self._mine_markdown_files())

        # Mine Jupyter notebooks
        examples.extend(self._mine_notebooks())

        return examples

    def _mine_markdown_files(self) -> list[dict[str, Any]]:
        """Mine examples from Markdown and Quarto files.

        Returns
        -------
        list of dict
            Extracted examples
        """
        examples: list[dict[str, Any]] = []

        if not self.docs_directory.exists():
            return examples

        # Find all markdown and Quarto files
        markdown_files = list(self.docs_directory.rglob("*.md")) + list(
            self.docs_directory.rglob("*.qmd")
        )

        print(f"Scanning {len(markdown_files)} markdown/quarto files...")

        for md_file in markdown_files:
            extractor = MarkdownExtractor(md_file)
            file_examples = extractor.extract_examples()

            if file_examples:
                print(f"  Found {len(file_examples)} examples in {md_file.name}")

            # Convert to dictionaries
            examples.extend(ex.to_dict() for ex in file_examples)

        return examples

    def _mine_notebooks(self) -> list[dict[str, Any]]:
        """Mine examples from Jupyter notebooks.

        Returns
        -------
        list of dict
            Extracted examples
        """
        examples: list[dict[str, Any]] = []

        notebooks = list(self.repository_path.rglob("*.ipynb"))

        if not notebooks:
            return examples

        print(f"Scanning {len(notebooks)} Jupyter notebooks...")

        for notebook in notebooks:
            extractor = JupyterExtractor(notebook)
            notebook_examples = extractor.extract_examples()

            if notebook_examples:
                print(f"  Found {len(notebook_examples)} in {notebook.name}")

            # Convert to dictionaries
            examples.extend(ex.to_dict() for ex in notebook_examples)

        return examples


def extract_from_markdown(markdown_file: Path) -> list[dict[str, Any]]:
    """Extract SQL→Ibis examples from a Markdown or Quarto file.

    This is a convenience function for the public API.

    Parameters
    ----------
    markdown_file : Path
        Markdown or Quarto file to parse

    Returns
    -------
    list of dict
        Extracted code examples
    """
    extractor = MarkdownExtractor(markdown_file)
    examples = extractor.extract_examples()
    return [ex.to_dict() for ex in examples]


def extract_from_jupyter(notebook_path: Path) -> list[dict[str, Any]]:
    """Extract SQL→Ibis examples from a Jupyter notebook.

    This is a convenience function for the public API.

    Parameters
    ----------
    notebook_path : Path
        Jupyter notebook file

    Returns
    -------
    list of dict
        Extracted examples
    """
    extractor = JupyterExtractor(notebook_path)
    examples = extractor.extract_examples()
    return [ex.to_dict() for ex in examples]


def save_examples_to_jsonl(examples: list[dict[str, Any]], output_path: Path) -> None:
    """Save extracted examples to JSONL file.

    Parameters
    ----------
    examples : list of dict
        Examples to save
    output_path : Path
        Destination file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"Saved {len(examples)} examples to {output_path}")


def main() -> None:
    """Main entry point for documentation mining."""
    ibis_repo_path = Path("data/mining/repos/ibis")

    if not ibis_repo_path.exists():
        print("Error: Ibis repository not found.")
        print("Run github_miner.py first to clone the repository.")
        return

    print("Mining Ibis documentation for SQL→Ibis examples...")

    miner = DocumentationMiner(ibis_repo_path)
    examples = miner.mine_all_documentation()

    print(f"\nFound {len(examples)} documentation examples")

    # Save results
    output_path = Path("data/mining/ibis_docs_mined.jsonl")
    save_examples_to_jsonl(examples, output_path)


if __name__ == "__main__":
    main()
