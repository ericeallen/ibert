"""Comprehensive test suite for GitHub repository mining functionality.

This module tests the mining of SQL→Ibis examples from GitHub repositories,
covering:
- Repository cloning
- SQL pattern extraction
- Python file scanning
- Jupyter notebook processing
- Configuration file parsing
"""

import pytest
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

from src.datagen.mining.github_miner import (
    GitHubRepositoryMiner,
    RepositoryScanner,
    RepositoryConfig,
    SQLExample,
    mine_repository,
    load_repository_config,
    _parse_config_line,
    SQL_KEYWORD,
    GIT_CLONE_DEPTH,
)


class TestSQLExample:
    """Test suite for the SQLExample NamedTuple."""

    def test_creation_with_all_fields(self):
        """Test creating SQLExample with all fields."""
        example = SQLExample(
            source_type="table.sql()",
            file_path="/path/to/file.py",
            sql_code="SELECT * FROM users",
            ibis_var="result"
        )

        assert example.source_type == "table.sql()"
        assert example.file_path == "/path/to/file.py"
        assert example.sql_code == "SELECT * FROM users"
        assert example.ibis_var == "result"

    def test_creation_without_ibis_var(self):
        """Test creating SQLExample without ibis_var."""
        example = SQLExample(
            source_type="direct_sql",
            file_path="/path/to/file.py",
            sql_code="SELECT COUNT(*) FROM events"
        )

        assert example.ibis_var is None

    def test_to_dict_with_ibis_var(self):
        """Test converting to dictionary with ibis_var."""
        example = SQLExample(
            source_type="table.sql()",
            file_path="/path/to/file.py",
            sql_code="SELECT * FROM users",
            ibis_var="result"
        )

        result = example.to_dict()

        assert result["source"] == "table.sql()"
        assert result["file"] == "/path/to/file.py"
        assert result["sql"] == "SELECT * FROM users"
        assert result["ibis_var"] == "result"

    def test_to_dict_without_ibis_var(self):
        """Test converting to dictionary without ibis_var."""
        example = SQLExample(
            source_type="direct_sql",
            file_path="/path/to/file.py",
            sql_code="SELECT COUNT(*) FROM events"
        )

        result = example.to_dict()

        assert "ibis_var" not in result
        assert result["source"] == "direct_sql"


class TestRepositoryConfig:
    """Test suite for the RepositoryConfig NamedTuple."""

    def test_creation_with_scan_dirs(self):
        """Test creating config with scan directories."""
        config = RepositoryConfig(
            url="https://github.com/user/repo.git",
            name="repo",
            scan_dirs=["src", "tests"]
        )

        assert config.url == "https://github.com/user/repo.git"
        assert config.name == "repo"
        assert config.scan_dirs == ["src", "tests"]

    def test_creation_without_scan_dirs(self):
        """Test creating config without scan directories."""
        config = RepositoryConfig(
            url="https://github.com/user/repo.git",
            name="repo",
            scan_dirs=None
        )

        assert config.scan_dirs is None


class TestGitHubRepositoryMiner:
    """Test suite for the GitHubRepositoryMiner class."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path: Path) -> Path:
        """Create temporary cache directory.

        Parameters
        ----------
        tmp_path : Path
            Pytest's temporary directory

        Returns
        -------
        Path
            Cache directory path
        """
        cache_dir = tmp_path / "repos"
        return cache_dir

    @pytest.fixture
    def miner(self, temp_cache_dir: Path) -> GitHubRepositoryMiner:
        """Create GitHubRepositoryMiner instance.

        Parameters
        ----------
        temp_cache_dir : Path
            Temporary cache directory

        Returns
        -------
        GitHubRepositoryMiner
            Miner instance
        """
        return GitHubRepositoryMiner(temp_cache_dir)

    def test_initialization(self, miner: GitHubRepositoryMiner, temp_cache_dir: Path):
        """Test miner initialization."""
        assert miner.cache_dir == temp_cache_dir
        assert miner.cache_dir.exists()

    @patch('subprocess.run')
    def test_clone_repository_new(self, mock_run, miner: GitHubRepositoryMiner):
        """Test cloning a new repository."""
        repo_url = "https://github.com/test/repo.git"
        local_name = "test-repo"

        mock_run.return_value = Mock()

        result = miner.clone_repository(repo_url, local_name)

        assert result == miner.cache_dir / local_name
        mock_run.assert_called_once()

        # Verify git clone command
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "git"
        assert call_args[1] == "clone"
        assert "--depth" in call_args
        assert str(GIT_CLONE_DEPTH) in call_args
        assert repo_url in call_args

    def test_clone_repository_already_exists(
        self,
        miner: GitHubRepositoryMiner,
        capsys
    ):
        """Test cloning when repository already exists."""
        local_name = "existing-repo"
        repo_path = miner.cache_dir / local_name
        repo_path.mkdir(parents=True)

        with patch('subprocess.run') as mock_run:
            result = miner.clone_repository("https://github.com/test/repo.git", local_name)

            assert result == repo_path
            mock_run.assert_not_called()

        captured = capsys.readouterr()
        assert "already cloned" in captured.out

    def test_find_python_files(self, miner: GitHubRepositoryMiner, tmp_path: Path):
        """Test finding Python files recursively."""
        # Create directory structure with Python files
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        (repo_dir / "file1.py").touch()
        (repo_dir / "file2.txt").touch()

        subdir = repo_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.py").touch()

        python_files = miner.find_python_files(repo_dir)

        assert len(python_files) == 2
        assert all(f.suffix == ".py" for f in python_files)

    def test_find_jupyter_notebooks(self, miner: GitHubRepositoryMiner, tmp_path: Path):
        """Test finding Jupyter notebooks."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        (repo_dir / "notebook1.ipynb").touch()
        (repo_dir / "file.py").touch()

        subdir = repo_dir / "subdir"
        subdir.mkdir()
        (subdir / "notebook2.ipynb").touch()

        notebooks = miner.find_jupyter_notebooks(repo_dir)

        assert len(notebooks) == 2
        assert all(f.suffix == ".ipynb" for f in notebooks)

    def test_read_file_safely_success(self, miner: GitHubRepositoryMiner, tmp_path: Path):
        """Test reading file successfully."""
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        content = miner._read_file_safely(test_file)

        assert content == "print('hello')"

    def test_read_file_safely_file_not_found(self, miner: GitHubRepositoryMiner):
        """Test reading non-existent file."""
        content = miner._read_file_safely(Path("/nonexistent/file.py"))

        assert content is None

    def test_is_valid_sql_query(self, miner: GitHubRepositoryMiner):
        """Test SQL query validation."""
        assert miner._is_valid_sql_query("SELECT * FROM users")
        assert miner._is_valid_sql_query("select count(*) from events")
        assert not miner._is_valid_sql_query("print('hello')")
        assert not miner._is_valid_sql_query("import ibis")

    def test_extract_sql_method_calls(self, miner: GitHubRepositoryMiner, tmp_path: Path):
        """Test extracting SQL from method calls."""
        code = '''
import ibis

result = t.sql("SELECT user_id, name FROM users WHERE age > 18")
other = con.sql("SELECT COUNT(*) FROM events")
'''
        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        examples = miner._extract_sql_method_calls(code, test_file)

        assert len(examples) == 2
        assert examples[0].source_type == "table.sql()"
        assert examples[0].ibis_var == "result"
        assert "SELECT user_id" in examples[0].sql_code
        assert examples[1].ibis_var == "other"

    def test_extract_direct_sql_calls(self, miner: GitHubRepositoryMiner, tmp_path: Path):
        """Test extracting direct SQL calls."""
        code = '''
con.sql("SELECT * FROM users")
backend.sql("SELECT id FROM products")
'''
        test_file = tmp_path / "test.py"

        examples = miner._extract_direct_sql_calls(code, test_file)

        assert len(examples) == 2
        assert all(ex.source_type == "direct_sql" for ex in examples)
        assert "users" in examples[0].sql_code
        assert "products" in examples[1].sql_code

    def test_extract_multiline_sql(self, miner: GitHubRepositoryMiner, tmp_path: Path):
        """Test extracting multiline SQL strings."""
        code = '''
sql = """
SELECT user_id, COUNT(*) as event_count
FROM events
GROUP BY user_id
"""
'''
        test_file = tmp_path / "test.py"

        examples = miner._extract_multiline_sql(code, test_file)

        assert len(examples) == 1
        assert examples[0].source_type == "multiline_sql"
        assert "GROUP BY" in examples[0].sql_code

    def test_extract_sql_examples_combined(
        self,
        miner: GitHubRepositoryMiner,
        tmp_path: Path
    ):
        """Test extracting all SQL patterns from a file."""
        code = '''
import ibis

# Pattern 1: Method call with assignment
result = t.sql("SELECT * FROM users")

# Pattern 2: Direct call
con.sql("SELECT COUNT(*) FROM events")

# Pattern 3: Multiline
sql = """
SELECT name, age
FROM users
WHERE age > 18
"""
'''
        test_file = tmp_path / "combined.py"
        test_file.write_text(code)

        examples = miner.extract_sql_examples(test_file)

        # Note: Pattern 1 also matches pattern 2, so we get 4 examples
        assert len(examples) >= 3

        # Check different pattern types are all found
        source_types = {ex.source_type for ex in examples}
        assert "table.sql()" in source_types
        assert "direct_sql" in source_types
        assert "multiline_sql" in source_types

    def test_extract_sql_examples_file_read_error(
        self,
        miner: GitHubRepositoryMiner
    ):
        """Test handling file read errors."""
        examples = miner.extract_sql_examples(Path("/nonexistent/file.py"))

        assert examples == []


class TestRepositoryScanner:
    """Test suite for the RepositoryScanner class."""

    @pytest.fixture
    def miner(self, tmp_path: Path) -> GitHubRepositoryMiner:
        """Create miner instance."""
        return GitHubRepositoryMiner(tmp_path / "repos")

    @pytest.fixture
    def scanner(self, miner: GitHubRepositoryMiner) -> RepositoryScanner:
        """Create scanner instance."""
        return RepositoryScanner(miner)

    def test_initialization(self, scanner: RepositoryScanner, miner: GitHubRepositoryMiner):
        """Test scanner initialization."""
        assert scanner.miner == miner

    def test_get_scan_paths_with_target_dirs(
        self,
        scanner: RepositoryScanner,
        tmp_path: Path
    ):
        """Test getting scan paths with specific directories."""
        repo_path = tmp_path / "repo"
        target_dirs = ["src", "tests"]

        paths = scanner._get_scan_paths(repo_path, target_dirs)

        assert len(paths) == 2
        assert paths[0] == repo_path / "src"
        assert paths[1] == repo_path / "tests"

    def test_get_scan_paths_without_target_dirs(
        self,
        scanner: RepositoryScanner,
        tmp_path: Path
    ):
        """Test getting scan paths without specific directories."""
        repo_path = tmp_path / "repo"

        paths = scanner._get_scan_paths(repo_path, None)

        assert len(paths) == 1
        assert paths[0] == repo_path

    def test_get_directory_label_repo_root(
        self,
        scanner: RepositoryScanner,
        tmp_path: Path
    ):
        """Test directory label for repo root."""
        repo_path = tmp_path / "repo"

        label = scanner._get_directory_label(repo_path, repo_path, "test-repo")

        assert label == "test-repo"

    def test_get_directory_label_subdirectory(
        self,
        scanner: RepositoryScanner,
        tmp_path: Path
    ):
        """Test directory label for subdirectory."""
        repo_path = tmp_path / "repo"
        subdir = repo_path / "src" / "tests"

        label = scanner._get_directory_label(subdir, repo_path, "test-repo")

        assert "src" in label
        assert "tests" in label

    def test_scan_python_files(
        self,
        scanner: RepositoryScanner,
        tmp_path: Path
    ):
        """Test scanning Python files."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create Python file with SQL
        py_file = repo_path / "example.py"
        py_file.write_text('result = t.sql("SELECT * FROM users")')

        scan_paths = [repo_path]

        examples = scanner._scan_python_files(scan_paths, repo_path, "test-repo")

        # May match both table.sql() and direct_sql patterns
        assert len(examples) >= 1
        assert any(ex["source"] == "table.sql()" for ex in examples)
        assert any("users" in ex["sql"] for ex in examples)

    def test_scan_repository_full(self, scanner: RepositoryScanner, tmp_path: Path):
        """Test full repository scanning."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create Python file
        py_file = repo_path / "code.py"
        py_file.write_text('con.sql("SELECT id FROM products")')

        examples = scanner.scan_repository(repo_path, "test-repo", None)

        assert len(examples) >= 1
        assert any("products" in ex["sql"] for ex in examples)


class TestConfigParsing:
    """Test suite for configuration file parsing."""

    def test_parse_config_line_full(self):
        """Test parsing complete config line."""
        line = "https://github.com/user/repo.git|repo-name|src,tests,docs"

        config = _parse_config_line(line)

        assert config is not None
        assert config.url == "https://github.com/user/repo.git"
        assert config.name == "repo-name"
        assert config.scan_dirs == ["src", "tests", "docs"]

    def test_parse_config_line_no_scan_dirs(self):
        """Test parsing config line without scan directories."""
        line = "https://github.com/user/repo.git|repo-name|"

        config = _parse_config_line(line)

        assert config is not None
        assert config.scan_dirs is None

    def test_parse_config_line_minimal(self):
        """Test parsing minimal config line."""
        line = "https://github.com/user/repo.git|repo-name"

        config = _parse_config_line(line)

        assert config is not None
        assert config.url == "https://github.com/user/repo.git"
        assert config.name == "repo-name"
        assert config.scan_dirs is None

    def test_parse_config_line_invalid(self, capsys):
        """Test parsing invalid config line."""
        line = "invalid-line"

        config = _parse_config_line(line)

        assert config is None

        captured = capsys.readouterr()
        assert "Skipping invalid config line" in captured.out

    def test_load_repository_config(self, tmp_path: Path):
        """Test loading full repository configuration."""
        config_file = tmp_path / "repos.txt"
        config_file.write_text("""
# Comment line
https://github.com/user/repo1.git|repo1|src,tests

# Another comment
https://github.com/user/repo2.git|repo2|

https://github.com/user/repo3.git|repo3
""")

        configs = load_repository_config(config_file)

        assert len(configs) == 3
        assert configs[0].name == "repo1"
        assert configs[0].scan_dirs == ["src", "tests"]
        assert configs[1].name == "repo2"
        assert configs[1].scan_dirs is None
        assert configs[2].name == "repo3"

    def test_load_repository_config_with_empty_lines(self, tmp_path: Path):
        """Test loading config with blank lines."""
        config_file = tmp_path / "repos.txt"
        config_file.write_text("""
https://github.com/user/repo1.git|repo1|src


https://github.com/user/repo2.git|repo2|
""")

        configs = load_repository_config(config_file)

        assert len(configs) == 2


class TestMineRepository:
    """Test suite for the mine_repository function."""

    @patch.object(GitHubRepositoryMiner, 'clone_repository')
    @patch.object(RepositoryScanner, 'scan_repository')
    def test_mine_repository_success(
        self,
        mock_scan,
        mock_clone,
        tmp_path: Path
    ):
        """Test successful repository mining."""
        repo_path = tmp_path / "repo"
        mock_clone.return_value = repo_path
        mock_scan.return_value = [{"sql": "SELECT 1"}]

        miner = GitHubRepositoryMiner(tmp_path / "cache")
        examples = mine_repository(
            "https://github.com/user/repo.git",
            "repo",
            ["src"],
            miner
        )

        assert len(examples) == 1
        mock_clone.assert_called_once()
        mock_scan.assert_called_once()

    @patch.object(GitHubRepositoryMiner, 'clone_repository')
    def test_mine_repository_clone_failure(
        self,
        mock_clone,
        tmp_path: Path,
        capsys
    ):
        """Test handling clone failure."""
        mock_clone.side_effect = subprocess.CalledProcessError(1, 'git')

        miner = GitHubRepositoryMiner(tmp_path / "cache")
        examples = mine_repository(
            "https://github.com/user/repo.git",
            "repo",
            None,
            miner
        )

        assert examples == []

        captured = capsys.readouterr()
        assert "Error cloning" in captured.out


class TestIntegration:
    """Integration tests for the complete mining workflow."""

    def test_end_to_end_mining(self, tmp_path: Path):
        """Test complete end-to-end mining workflow."""
        # Create mock repository
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        # Create Python files with different SQL patterns
        code_file1 = repo_dir / "users.py"
        code_file1.write_text('users_query = t.sql("SELECT user_id FROM users WHERE active = true")')

        code_file2 = repo_dir / "orders.py"
        code_file2.write_text('''
sql = """
SELECT product_id, SUM(quantity) as total
FROM orders
GROUP BY product_id
"""
''')

        # Create miner and scanner
        miner = GitHubRepositoryMiner(tmp_path / "cache")
        scanner = RepositoryScanner(miner)

        # Scan repository
        examples = scanner.scan_repository(repo_dir, "test-repo", None)

        # Verify results - should find examples from both files
        assert len(examples) >= 2

        # Verify different types of examples
        sql_codes = [ex["sql"] for ex in examples]
        assert any("users" in sql.lower() for sql in sql_codes)
        assert any("orders" in sql.lower() for sql in sql_codes)


