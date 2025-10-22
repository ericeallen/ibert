"""Comprehensive test suite for documentation extraction functionality.

This module tests the extraction of SQL→Ibis examples from documentation,
covering:
- Markdown and Quarto document parsing
- Jupyter notebook extraction
- Code block pattern matching
- Sequential SQL/Ibis block detection
"""

import json
from pathlib import Path
from typing import Any

import pytest

from src.datagen.mining.ibis_doc_extractor import (
    DocumentationMiner,
    DocumentExample,
    JupyterExtractor,
    MarkdownExtractor,
    extract_from_jupyter,
    extract_from_markdown,
)


class TestDocumentExample:
    """Test suite for the DocumentExample NamedTuple."""

    def test_creation_full(self):
        """Test creating DocumentExample with all fields."""
        example = DocumentExample(
            source_type="markdown_doc",
            file_path="/docs/example.md",
            sql_code="SELECT * FROM users",
            ibis_code="users.select()",
            context="Full code block",
        )

        assert example.source_type == "markdown_doc"
        assert example.file_path == "/docs/example.md"
        assert example.sql_code == "SELECT * FROM users"
        assert example.ibis_code == "users.select()"
        assert example.context == "Full code block"

    def test_creation_minimal(self):
        """Test creating DocumentExample with minimal fields."""
        example = DocumentExample(
            source_type="quarto_doc",
            file_path="/docs/example.qmd",
            sql_code="SELECT COUNT(*) FROM events",
        )

        assert example.ibis_code is None
        assert example.context is None

    def test_to_dict_with_ibis_code(self):
        """Test converting to dictionary with ibis_code."""
        example = DocumentExample(
            source_type="markdown_doc",
            file_path="/docs/example.md",
            sql_code="SELECT * FROM users",
            ibis_code="users.select()",
            context="code block",
        )

        result = example.to_dict()

        assert result["source"] == "markdown_doc"
        assert result["file"] == "/docs/example.md"
        assert result["sql"] == "SELECT * FROM users"
        assert result["ibis"] == "users.select()"
        assert result["context"] == "code block"

    def test_to_dict_without_ibis_code(self):
        """Test converting to dictionary with only context."""
        example = DocumentExample(
            source_type="jupyter_notebook",
            file_path="/notebooks/demo.ipynb",
            sql_code="SELECT id FROM products",
            context="next cell",
        )

        result = example.to_dict()

        assert "ibis" not in result
        assert result["ibis_context"] == "next cell"

    def test_to_dict_minimal(self):
        """Test converting minimal example to dictionary."""
        example = DocumentExample(
            source_type="quarto_doc", file_path="/docs/test.qmd", sql_code="SELECT 1"
        )

        result = example.to_dict()

        assert "ibis" not in result
        assert "context" not in result
        assert "ibis_context" not in result


class TestMarkdownExtractor:
    """Test suite for the MarkdownExtractor class."""

    @pytest.fixture
    def temp_md_file(self, tmp_path: Path) -> Path:
        """Create temporary markdown file.

        Parameters
        ----------
        tmp_path : Path
            Pytest's temporary directory

        Returns
        -------
        Path
            Markdown file path
        """
        return tmp_path / "test.md"

    def test_initialization_success(self, temp_md_file: Path):
        """Test successful initialization."""
        temp_md_file.write_text("# Test Document")

        extractor = MarkdownExtractor(temp_md_file)

        assert extractor.file_path == temp_md_file
        assert extractor.content == "# Test Document"

    def test_initialization_file_not_found(self):
        """Test initialization with non-existent file."""
        extractor = MarkdownExtractor(Path("/nonexistent/file.md"))

        assert extractor.content is None

    def test_read_file_safely_success(self, temp_md_file: Path):
        """Test reading file successfully."""
        temp_md_file.write_text("Content")

        extractor = MarkdownExtractor(temp_md_file)
        content = extractor._read_file_safely()

        assert content == "Content"

    def test_is_valid_sql(self, temp_md_file: Path):
        """Test SQL validation."""
        temp_md_file.write_text("")
        extractor = MarkdownExtractor(temp_md_file)

        assert extractor._is_valid_sql("SELECT * FROM users")
        assert extractor._is_valid_sql("select count(*) from events")
        assert not extractor._is_valid_sql("print('hello')")

    def test_extract_quarto_examples(self, temp_md_file: Path):
        """Test extracting from Quarto Python blocks."""
        content = """
# Example

```{python}
import ibis
result = t.sql("SELECT * FROM users WHERE age > 18")
```

```{python}
data = con.sql("SELECT COUNT(*) FROM events")
```
"""
        temp_md_file.write_text(content)
        extractor = MarkdownExtractor(temp_md_file)

        examples = extractor._extract_quarto_examples()

        assert len(examples) == 2
        assert all(ex.source_type == "quarto_doc" for ex in examples)
        assert "users" in examples[0].sql_code
        assert "events" in examples[1].sql_code

    def test_extract_sequential_blocks(self, temp_md_file: Path):
        """Test extracting sequential SQL→Python blocks."""
        content = """
Here's an example:

```sql
SELECT user_id, COUNT(*) as event_count
FROM events
GROUP BY user_id
```

```python
import ibis
result = events.group_by('user_id').agg(event_count=ibis._.count())
```
"""
        temp_md_file.write_text(content)
        extractor = MarkdownExtractor(temp_md_file)

        examples = extractor._extract_sequential_blocks()

        assert len(examples) == 1
        assert examples[0].source_type == "markdown_doc"
        assert "GROUP BY" in examples[0].sql_code
        assert "ibis" in examples[0].ibis_code

    def test_extract_sequential_blocks_no_ibis(self, temp_md_file: Path):
        """Test that SQL→Python blocks without ibis are skipped."""
        content = """
```sql
SELECT * FROM users
```

```python
print("hello")
```
"""
        temp_md_file.write_text(content)
        extractor = MarkdownExtractor(temp_md_file)

        examples = extractor._extract_sequential_blocks()

        assert len(examples) == 0

    def test_extract_python_sql_strings(self, temp_md_file: Path):
        """Test extracting SQL strings from Python blocks."""
        content = """
```python
query = "SELECT id, name FROM products WHERE price > 100"
con.execute(query)
```

```py
sql = "SELECT COUNT(*) FROM orders"
```
"""
        temp_md_file.write_text(content)
        extractor = MarkdownExtractor(temp_md_file)

        examples = extractor._extract_python_sql_strings()

        assert len(examples) == 2
        assert all(ex.source_type == "python_code_block" for ex in examples)

    def test_extract_examples_combined(self, temp_md_file: Path):
        """Test extracting all patterns from a document."""
        content = """
# SQL→Ibis Examples

## Quarto Style

```{python}
result = t.sql("SELECT * FROM users")
```

## Sequential Blocks

```sql
SELECT COUNT(*) FROM events
```

```python
import ibis
count = events.count()
```

## Python String

```python
query = "SELECT id FROM products"
```
"""
        temp_md_file.write_text(content)
        extractor = MarkdownExtractor(temp_md_file)

        examples = extractor.extract_examples()

        assert len(examples) >= 3

        source_types = {ex.source_type for ex in examples}
        assert "quarto_doc" in source_types
        assert "markdown_doc" in source_types
        assert "python_code_block" in source_types

    def test_extract_examples_empty_file(self, temp_md_file: Path):
        """Test extracting from empty file."""
        temp_md_file.write_text("")
        extractor = MarkdownExtractor(temp_md_file)

        examples = extractor.extract_examples()

        assert examples == []


class TestJupyterExtractor:
    """Test suite for the JupyterExtractor class."""

    @pytest.fixture
    def sample_notebook(self) -> dict[str, Any]:
        """Create sample notebook structure.

        Returns
        -------
        dict
            Notebook data
        """
        return {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["import ibis\n", 'result = t.sql("SELECT * FROM users")\n'],
                },
                {"cell_type": "code", "source": ["result.head()\n"]},
                {"cell_type": "markdown", "source": ["# Example\n"]},
                {"cell_type": "code", "source": ['con.sql("SELECT COUNT(*) FROM events")\n']},
            ]
        }

    @pytest.fixture
    def temp_notebook_file(self, tmp_path: Path, sample_notebook: dict[str, Any]) -> Path:
        """Create temporary notebook file.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory
        sample_notebook : dict
            Notebook data

        Returns
        -------
        Path
            Notebook file path
        """
        notebook_file = tmp_path / "test.ipynb"
        with open(notebook_file, "w") as f:
            json.dump(sample_notebook, f)
        return notebook_file

    def test_initialization_success(self, temp_notebook_file: Path):
        """Test successful initialization."""
        extractor = JupyterExtractor(temp_notebook_file)

        assert extractor.notebook_path == temp_notebook_file
        assert extractor.notebook_data is not None
        assert "cells" in extractor.notebook_data

    def test_initialization_file_not_found(self):
        """Test initialization with non-existent file."""
        extractor = JupyterExtractor(Path("/nonexistent/notebook.ipynb"))

        assert extractor.notebook_data is None

    def test_load_notebook_invalid_json(self, tmp_path: Path):
        """Test loading invalid JSON notebook."""
        notebook_file = tmp_path / "invalid.ipynb"
        notebook_file.write_text("not valid json")

        extractor = JupyterExtractor(notebook_file)

        assert extractor.notebook_data is None

    def test_extract_sql_from_source(self, temp_notebook_file: Path):
        """Test extracting SQL from cell source."""
        extractor = JupyterExtractor(temp_notebook_file)

        source = 'result = t.sql("SELECT * FROM users")'
        sql = extractor._extract_sql_from_source(source)

        assert sql == "SELECT * FROM users"

    def test_extract_sql_from_source_not_found(self, temp_notebook_file: Path):
        """Test when SQL not found in source."""
        extractor = JupyterExtractor(temp_notebook_file)

        source = 'print("hello")'
        sql = extractor._extract_sql_from_source(source)

        assert sql is None

    def test_get_next_cell_context(self, temp_notebook_file: Path, sample_notebook: dict[str, Any]):
        """Test getting next cell context."""
        extractor = JupyterExtractor(temp_notebook_file)

        context = extractor._get_next_cell_context(0, sample_notebook["cells"])

        assert context is not None
        assert "result.head()" in context

    def test_get_next_cell_context_last_cell(
        self, temp_notebook_file: Path, sample_notebook: dict[str, Any]
    ):
        """Test getting context when current cell is last."""
        extractor = JupyterExtractor(temp_notebook_file)

        last_index = len(sample_notebook["cells"]) - 1
        context = extractor._get_next_cell_context(last_index, sample_notebook["cells"])

        assert context is None

    def test_extract_from_cell(self, temp_notebook_file: Path, sample_notebook: dict[str, Any]):
        """Test extracting examples from a cell."""
        extractor = JupyterExtractor(temp_notebook_file)

        examples = extractor._extract_from_cell(
            sample_notebook["cells"][0], 0, sample_notebook["cells"]
        )

        assert len(examples) == 1
        assert examples[0].source_type == "jupyter_notebook"
        assert "users" in examples[0].sql_code
        assert examples[0].context is not None

    def test_extract_from_cell_no_sql(
        self, temp_notebook_file: Path, sample_notebook: dict[str, Any]
    ):
        """Test extracting from cell without SQL."""
        extractor = JupyterExtractor(temp_notebook_file)

        examples = extractor._extract_from_cell(
            sample_notebook["cells"][2],  # Markdown cell
            2,
            sample_notebook["cells"],
        )

        assert len(examples) == 0

    def test_extract_examples(self, temp_notebook_file: Path):
        """Test extracting all examples from notebook."""
        extractor = JupyterExtractor(temp_notebook_file)

        examples = extractor.extract_examples()

        assert len(examples) == 2
        assert all(ex.source_type == "jupyter_notebook" for ex in examples)

    def test_extract_examples_empty_notebook(self, tmp_path: Path):
        """Test extracting from empty notebook."""
        notebook_file = tmp_path / "empty.ipynb"
        with open(notebook_file, "w") as f:
            json.dump({"cells": []}, f)

        extractor = JupyterExtractor(notebook_file)
        examples = extractor.extract_examples()

        assert examples == []


class TestDocumentationMiner:
    """Test suite for the DocumentationMiner class."""

    @pytest.fixture
    def temp_repo(self, tmp_path: Path) -> Path:
        """Create temporary repository structure.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory

        Returns
        -------
        Path
            Repository path
        """
        repo = tmp_path / "repo"
        repo.mkdir()

        docs_dir = repo / "docs"
        docs_dir.mkdir()

        return repo

    @pytest.fixture
    def miner(self, temp_repo: Path) -> DocumentationMiner:
        """Create DocumentationMiner instance.

        Parameters
        ----------
        temp_repo : Path
            Temporary repository

        Returns
        -------
        DocumentationMiner
            Miner instance
        """
        return DocumentationMiner(temp_repo)

    def test_initialization(self, temp_repo: Path):
        """Test miner initialization."""
        miner = DocumentationMiner(temp_repo)

        assert miner.repository_path == temp_repo
        assert miner.docs_directory == temp_repo / "docs"

    def test_mine_markdown_files(self, miner: DocumentationMiner, temp_repo: Path):
        """Test mining markdown files."""
        docs_dir = temp_repo / "docs"

        # Create markdown files
        md_file = docs_dir / "example.md"
        md_file.write_text(
            """
```{python}
result = t.sql("SELECT * FROM users")
```
"""
        )

        qmd_file = docs_dir / "tutorial.qmd"
        qmd_file.write_text(
            """
```{python}
con.sql("SELECT COUNT(*) FROM events")
```
"""
        )

        examples = miner._mine_markdown_files()

        assert len(examples) == 2

    def test_mine_markdown_files_no_docs(self, tmp_path: Path):
        """Test mining when docs directory doesn't exist."""
        repo = tmp_path / "repo"
        repo.mkdir()

        miner = DocumentationMiner(repo)
        examples = miner._mine_markdown_files()

        assert examples == []

    def test_mine_notebooks(self, miner: DocumentationMiner, temp_repo: Path):
        """Test mining Jupyter notebooks."""
        notebook_data = {
            "cells": [{"cell_type": "code", "source": ['result = t.sql("SELECT * FROM users")']}]
        }

        notebook_file = temp_repo / "example.ipynb"
        with open(notebook_file, "w") as f:
            json.dump(notebook_data, f)

        examples = miner._mine_notebooks()

        assert len(examples) == 1
        assert examples[0]["source"] == "jupyter_notebook"

    def test_mine_notebooks_none_found(self, miner: DocumentationMiner):
        """Test mining when no notebooks exist."""
        examples = miner._mine_notebooks()

        assert examples == []

    def test_mine_all_documentation(self, miner: DocumentationMiner, temp_repo: Path):
        """Test mining all documentation sources."""
        docs_dir = temp_repo / "docs"

        # Create markdown file
        md_file = docs_dir / "guide.md"
        md_file.write_text(
            """
```{python}
t.sql("SELECT * FROM users")
```
"""
        )

        # Create notebook
        notebook_data = {
            "cells": [{"cell_type": "code", "source": ['con.sql("SELECT COUNT(*) FROM events")']}]
        }

        notebook_file = temp_repo / "tutorial.ipynb"
        with open(notebook_file, "w") as f:
            json.dump(notebook_data, f)

        examples = miner.mine_all_documentation()

        assert len(examples) == 2


class TestConvenienceFunctions:
    """Test suite for public API convenience functions."""

    def test_extract_from_markdown(self, tmp_path: Path):
        """Test extract_from_markdown function."""
        md_file = tmp_path / "test.md"
        md_file.write_text(
            """
```{python}
result = t.sql("SELECT * FROM users")
```
"""
        )

        examples = extract_from_markdown(md_file)

        assert len(examples) == 1
        assert isinstance(examples[0], dict)
        assert examples[0]["source"] == "quarto_doc"

    def test_extract_from_jupyter(self, tmp_path: Path):
        """Test extract_from_jupyter function."""
        notebook_data = {
            "cells": [{"cell_type": "code", "source": ['t.sql("SELECT * FROM users")']}]
        }

        notebook_file = tmp_path / "test.ipynb"
        with open(notebook_file, "w") as f:
            json.dump(notebook_data, f)

        examples = extract_from_jupyter(notebook_file)

        assert len(examples) == 1
        assert isinstance(examples[0], dict)
        assert examples[0]["source"] == "jupyter_notebook"


class TestIntegration:
    """Integration tests for complete documentation mining workflow."""

    def test_end_to_end_mining(self, tmp_path: Path):
        """Test complete end-to-end documentation mining."""
        # Setup repository structure
        repo = tmp_path / "ibis"
        repo.mkdir()

        docs_dir = repo / "docs"
        docs_dir.mkdir()

        tutorials_dir = docs_dir / "tutorials"
        tutorials_dir.mkdir()

        # Create markdown documentation
        md_content = """
# Tutorial

## SQL Method

```{python}
import ibis
users = t.sql("SELECT * FROM users WHERE active = true")
```

## Sequential Example

```sql
SELECT product_id, COUNT(*) as order_count
FROM orders
GROUP BY product_id
```

```python
import ibis
result = orders.group_by('product_id').agg(
    order_count=ibis._.count()
)
```
"""
        (tutorials_dir / "tutorial.md").write_text(md_content)

        # Create Jupyter notebook
        notebook_data = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Example Notebook"]},
                {
                    "cell_type": "code",
                    "source": [
                        "import ibis\n",
                        'result = con.sql("SELECT COUNT(*) FROM events")\n',
                    ],
                },
                {"cell_type": "code", "source": ["result.execute()"]},
            ]
        }

        notebook_file = repo / "examples" / "demo.ipynb"
        notebook_file.parent.mkdir(parents=True)
        with open(notebook_file, "w") as f:
            json.dump(notebook_data, f)

        # Run complete mining
        miner = DocumentationMiner(repo)
        all_examples = miner.mine_all_documentation()

        # Verify results
        assert len(all_examples) >= 3

        # Check different source types
        source_types = {ex["source"] for ex in all_examples}
        assert "quarto_doc" in source_types or "markdown_doc" in source_types
        assert "jupyter_notebook" in source_types

        # Verify SQL content
        sql_codes = [ex["sql"] for ex in all_examples]
        assert any("users" in sql for sql in sql_codes)
        assert any("orders" in sql for sql in sql_codes)
        assert any("events" in sql for sql in sql_codes)
