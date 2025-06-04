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
from typer.testing import CliRunner

from imgpt.cli import (
    read_prompt_file,
    find_prompt_files,
    get_output_path,
    app
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
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AI Image Generator" in result.stdout
    assert "Generate images using OpenAI API" in result.stdout


def test_cli_version():
    """Test CLI version command."""
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "v0.3.0" in result.stdout


def test_config_show():
    """Test config show command."""
    runner = CliRunner()
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "Current Configuration" in result.stdout
    assert "API Settings" in result.stdout


def test_config_set():
    """Test config set command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('imgpt.config.ConfigManager._get_config_dir', return_value=Path(temp_dir)):
            runner = CliRunner()
            result = runner.invoke(app, ["config", "set", "default_model", "dall-e-2"])
            assert result.exit_code == 0
            assert "Set default_model = dall-e-2" in result.stdout


def test_generate_help():
    """Test generate command help."""
    runner = CliRunner()
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "Generate images from prompts" in result.stdout


def test_generate_missing_input():
    """Test generate command with missing input."""
    runner = CliRunner()
    result = runner.invoke(app, ["generate"])
    assert result.exit_code == 1
    assert "Must provide either a prompt or a directory" in result.stdout


def test_generate_invalid_directory():
    """Test generate command with invalid directory."""
    runner = CliRunner()
    result = runner.invoke(app, ["generate", "--dir", "/nonexistent/directory"])
    assert result.exit_code == 1
    assert "does not exist" in result.stdout


def test_generate_invalid_model_size():
    """Test generate command with invalid model/size combination."""
    runner = CliRunner()
    result = runner.invoke(app, ["generate", "test prompt", "--model", "dall-e-2", "--size", "2048x2048"])
    assert result.exit_code == 1
    assert "not valid for model" in result.stdout 