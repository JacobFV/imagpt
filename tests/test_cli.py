"""
Tests for the imgpt CLI functionality.
"""

import os
import tempfile
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from imgpt.cli import (
    read_prompt_file,
    find_prompt_files,
    get_output_path,
    main
)


def test_read_prompt_file_simple():
    """Test reading a simple prompt file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.prompt', delete=False) as f:
        f.write("A simple test prompt")
        f.flush()
        
        content = read_prompt_file(Path(f.name))
        assert content == "A simple test prompt"
        
        os.unlink(f.name)


def test_read_prompt_file_markdown():
    """Test reading a markdown prompt file with description."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("""# Test Markdown

**Description:**
A markdown test prompt with description section.

**Style:** Test
        """)
        f.flush()
        
        content = read_prompt_file(Path(f.name))
        assert "markdown test prompt" in content.lower()
        
        os.unlink(f.name)


def test_find_prompt_files():
    """Test finding prompt files in a directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        (temp_path / "test.prompt").write_text("prompt 1")
        (temp_path / "test.txt").write_text("prompt 2")
        (temp_path / "test.md").write_text("prompt 3")
        (temp_path / "ignore.py").write_text("not a prompt")
        
        files = find_prompt_files(temp_path)
        assert len(files) == 3
        
        extensions = {f.suffix for f in files}
        assert extensions == {'.prompt', '.txt', '.md'}


def test_get_output_path():
    """Test generating output paths."""
    prompt_file = Path("test.prompt")
    output_dir = Path("output")
    
    # Test default PNG format
    output_path = get_output_path(prompt_file, output_dir)
    assert output_path == Path("output/test.png")
    
    # Test different formats
    output_path_jpeg = get_output_path(prompt_file, output_dir, "jpeg")
    assert output_path_jpeg == Path("output/test.jpeg")
    
    output_path_webp = get_output_path(prompt_file, output_dir, "webp")
    assert output_path_webp == Path("output/test.webp")


def test_cli_help():
    """Test CLI help command."""
    result = subprocess.run(
        [sys.executable, "-m", "imgpt.cli", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "imgpt" in result.stdout
    assert "Generate images using OpenAI API" in result.stdout


def test_cli_version():
    """Test CLI version command."""
    result = subprocess.run(
        [sys.executable, "-m", "imgpt.cli", "--version"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_missing_api_key():
    """Test behavior when API key is missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(SystemExit):
            from imgpt.cli import get_api_key
            get_api_key()


def test_directory_validation():
    """Test directory validation."""
    result = subprocess.run(
        [sys.executable, "-m", "imgpt.cli", "--dir", "/nonexistent/directory"],
        capture_output=True,
        text=True,
        env={"OPENAI_API_KEY": "test-key"}
    )
    assert result.returncode == 1
    assert "does not exist" in result.stdout 