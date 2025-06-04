#!/usr/bin/env python3
"""
Tests for the configuration system.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from imgpt.config import ConfigManager, ImageptConfig


class TestImageptConfig:
    """Test the ImageptConfig model."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ImageptConfig()
        
        assert config.openai_api_key is None
        assert config.default_model == "gpt-image-1"
        assert config.default_size is None
        assert config.default_quality == "high"
        assert config.default_style is None
        assert config.default_format == "png"
        assert config.default_prompts_dir is None
        assert config.default_output_dir is None
        assert config.default_delay == 2.0
        assert config.skip_existing is False
    
    def test_valid_size_validation(self):
        """Test valid size formats."""
        config = ImageptConfig(default_size="1024x1024")
        assert config.default_size == "1024x1024"
        
        config = ImageptConfig(default_size="1536x1024")
        assert config.default_size == "1536x1024"
    
    def test_invalid_size_validation(self):
        """Test invalid size formats."""
        with pytest.raises(ValueError):
            ImageptConfig(default_size="invalid")
        
        with pytest.raises(ValueError):
            ImageptConfig(default_size="1024")
        
        with pytest.raises(ValueError):
            ImageptConfig(default_size="1024x")
        
        with pytest.raises(ValueError):
            ImageptConfig(default_size="0x1024")


class TestConfigManager:
    """Test the ConfigManager class."""
    
    def test_config_dir_creation(self):
        """Test that config directory is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config_dir = Path(temp_dir) / "test_config"
            
            def mock_get_config_dir():
                test_config_dir.mkdir(parents=True, exist_ok=True)
                return test_config_dir
            
            with patch.object(ConfigManager, '_get_config_dir', side_effect=mock_get_config_dir):
                manager = ConfigManager()
                # The constructor should have called _get_config_dir and created the directory
                assert manager.config_dir == test_config_dir
                assert manager.config_dir.exists()
                assert manager.config_dir.is_dir()
    
    def test_load_default_config(self):
        """Test loading default config when no file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(ConfigManager, '_get_config_dir', return_value=Path(temp_dir)):
                manager = ConfigManager()
                config = manager.load_config()
                
                assert isinstance(config, ImageptConfig)
                assert config.default_model == "gpt-image-1"
                assert config.default_quality == "high"
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(ConfigManager, '_get_config_dir', return_value=Path(temp_dir)):
                manager = ConfigManager()
                
                # Create a custom config
                config = ImageptConfig(
                    openai_api_key="test-key",
                    default_model="dall-e-3",
                    default_size="1024x1024",
                    default_quality="hd"
                )
                
                # Save config
                manager.save_config(config)
                
                # Create new manager and load config
                manager2 = ConfigManager()
                loaded_config = manager2.load_config()
                
                assert loaded_config.openai_api_key == "test-key"
                assert loaded_config.default_model == "dall-e-3"
                assert loaded_config.default_size == "1024x1024"
                assert loaded_config.default_quality == "hd"
    
    def test_update_config(self):
        """Test updating configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(ConfigManager, '_get_config_dir', return_value=Path(temp_dir)):
                manager = ConfigManager()
                
                # Update config
                updated_config = manager.update_config(
                    default_model="dall-e-2",
                    default_delay=1.5
                )
                
                assert updated_config.default_model == "dall-e-2"
                assert updated_config.default_delay == 1.5
                
                # Verify it's persisted
                manager2 = ConfigManager()
                loaded_config = manager2.load_config()
                assert loaded_config.default_model == "dall-e-2"
                assert loaded_config.default_delay == 1.5
    
    def test_reset_config(self):
        """Test resetting configuration to defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(ConfigManager, '_get_config_dir', return_value=Path(temp_dir)):
                manager = ConfigManager()
                
                # Set some custom values
                manager.update_config(
                    default_model="dall-e-3",
                    default_delay=5.0
                )
                
                # Reset to defaults
                reset_config = manager.reset_config()
                
                assert reset_config.default_model == "gpt-image-1"
                assert reset_config.default_delay == 2.0
    
    def test_get_api_key_from_config(self):
        """Test getting API key from config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(ConfigManager, '_get_config_dir', return_value=Path(temp_dir)):
                manager = ConfigManager()
                
                # Set API key in config
                manager.update_config(openai_api_key="config-key")
                
                # Should get key from config
                api_key = manager.get_api_key()
                assert api_key == "config-key"
    
    def test_get_api_key_from_env(self):
        """Test getting API key from environment when not in config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(ConfigManager, '_get_config_dir', return_value=Path(temp_dir)):
                with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
                    manager = ConfigManager()
                    
                    # Should get key from environment
                    api_key = manager.get_api_key()
                    assert api_key == "env-key"
    
    def test_get_api_key_config_priority(self):
        """Test that config API key takes priority over environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(ConfigManager, '_get_config_dir', return_value=Path(temp_dir)):
                with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
                    manager = ConfigManager()
                    
                    # Set API key in config
                    manager.update_config(openai_api_key="config-key")
                    
                    # Should get key from config, not environment
                    api_key = manager.get_api_key()
                    assert api_key == "config-key"
    
    def test_get_api_key_missing(self):
        """Test error when API key is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(ConfigManager, '_get_config_dir', return_value=Path(temp_dir)):
                with patch.dict(os.environ, {}, clear=True):
                    manager = ConfigManager()
                    
                    # Should exit with error
                    with pytest.raises(SystemExit):
                        manager.get_api_key() 