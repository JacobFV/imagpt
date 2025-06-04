#!/usr/bin/env python3
"""
Configuration management for imgpt.

Handles persistent configuration storage and validation using Pydantic.
Configuration is stored in a TOML file in the user's config directory.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Literal
import tomllib
import tomli_w
from pydantic import BaseModel, Field, field_validator


class ImageptConfig(BaseModel):
    """Configuration model for imgpt settings."""
    
    # API Configuration
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for image generation"
    )
    
    # Default Generation Settings
    default_model: Literal["gpt-image-1", "dall-e-2", "dall-e-3"] = Field(
        default="gpt-image-1",
        description="Default model to use for image generation"
    )
    
    default_size: Optional[str] = Field(
        default=None,
        description="Default image size (e.g., '1024x1024', '1536x1024')"
    )
    
    default_quality: Literal["auto", "high", "medium", "low", "hd", "standard"] = Field(
        default="high",
        description="Default image quality"
    )
    
    default_style: Optional[Literal["vivid", "natural"]] = Field(
        default=None,
        description="Default style for DALL-E 3 (vivid or natural)"
    )
    
    default_format: Literal["png", "jpeg", "webp"] = Field(
        default="png",
        description="Default output format"
    )
    
    # Directory Settings
    default_prompts_dir: Optional[str] = Field(
        default=None,
        description="Default directory to look for prompt files"
    )
    
    default_output_dir: Optional[str] = Field(
        default=None,
        description="Default directory to save generated images"
    )
    
    # Processing Settings
    default_delay: float = Field(
        default=2.0,
        ge=0.0,
        description="Default delay between API calls in seconds"
    )
    
    skip_existing: bool = Field(
        default=False,
        description="Default setting for skipping existing images"
    )
    
    @field_validator('default_size')
    @classmethod
    def validate_size(cls, v):
        """Validate image size format."""
        if v is None:
            return v
        
        if not isinstance(v, str):
            raise ValueError("Size must be a string")
        
        try:
            parts = v.split('x')
            if len(parts) != 2:
                raise ValueError("Size must be in format 'WIDTHxHEIGHT'")
            
            width, height = int(parts[0]), int(parts[1])
            if width <= 0 or height <= 0:
                raise ValueError("Width and height must be positive integers")
                
        except ValueError as e:
            raise ValueError(f"Invalid size format '{v}': {e}")
        
        return v
    
    @field_validator('default_prompts_dir', 'default_output_dir')
    @classmethod
    def validate_directories(cls, v):
        """Validate directory paths."""
        if v is None:
            return v
        
        path = Path(v).expanduser()
        if not path.exists():
            raise ValueError(f"Directory does not exist: {path}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        
        return str(path)


class ConfigManager:
    """Manages persistent configuration for imgpt."""
    
    def __init__(self):
        self.config_dir = self._get_config_dir()
        self.config_file = self.config_dir / "config.toml"
        self._config: Optional[ImageptConfig] = None
    
    def _get_config_dir(self) -> Path:
        """Get the configuration directory for the current platform."""
        if sys.platform == "win32":
            config_dir = Path(os.environ.get("APPDATA", "~")) / "imgpt"
        elif sys.platform == "darwin":
            config_dir = Path("~/Library/Application Support/imgpt").expanduser()
        else:
            # Linux and other Unix-like systems
            config_dir = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser() / "imgpt"
        
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir
    
    def load_config(self) -> ImageptConfig:
        """Load configuration from file or create default."""
        if self._config is not None:
            return self._config
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'rb') as f:
                    config_data = tomllib.load(f)
                self._config = ImageptConfig(**config_data)
            except Exception as e:
                print(f"⚠️  Warning: Failed to load config from {self.config_file}: {e}")
                print("Using default configuration.")
                self._config = ImageptConfig()
        else:
            self._config = ImageptConfig()
        
        return self._config
    
    def save_config(self, config: ImageptConfig) -> None:
        """Save configuration to file."""
        try:
            config_data = config.model_dump(exclude_none=True)
            with open(self.config_file, 'wb') as f:
                tomli_w.dump(config_data, f)
            self._config = config
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
            raise
    
    def update_config(self, **kwargs) -> ImageptConfig:
        """Update configuration with new values."""
        config = self.load_config()
        
        # Create a new config with updated values
        config_data = config.model_dump()
        config_data.update(kwargs)
        
        try:
            new_config = ImageptConfig(**config_data)
            self.save_config(new_config)
            return new_config
        except Exception as e:
            print(f"❌ Failed to update configuration: {e}")
            raise
    
    def reset_config(self) -> ImageptConfig:
        """Reset configuration to defaults."""
        config = ImageptConfig()
        self.save_config(config)
        return config
    
    def show_config(self) -> None:
        """Display current configuration."""
        config = self.load_config()
        
        print("🔧 Current Configuration")
        print("=" * 50)
        
        # API Settings
        print("\n📡 API Settings:")
        api_key = config.openai_api_key
        if api_key:
            masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
            print(f"  OpenAI API Key: {masked_key}")
        else:
            print("  OpenAI API Key: Not set")
        
        # Generation Settings
        print("\n🎨 Generation Settings:")
        print(f"  Default Model: {config.default_model}")
        print(f"  Default Size: {config.default_size or 'Auto (model-dependent)'}")
        print(f"  Default Quality: {config.default_quality}")
        print(f"  Default Style: {config.default_style or 'None'}")
        print(f"  Default Format: {config.default_format}")
        
        # Directory Settings
        print("\n📁 Directory Settings:")
        print(f"  Default Prompts Dir: {config.default_prompts_dir or 'None'}")
        print(f"  Default Output Dir: {config.default_output_dir or 'None'}")
        
        # Processing Settings
        print("\n⚙️  Processing Settings:")
        print(f"  Default Delay: {config.default_delay}s")
        print(f"  Skip Existing: {config.skip_existing}")
        
        print(f"\n📄 Config File: {self.config_file}")
    
    def get_api_key(self) -> str:
        """Get API key from config or environment."""
        config = self.load_config()
        
        # Try config first, then environment
        api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("❌ Error: OpenAI API key not found")
            print("Set it using one of these methods:")
            print("1. imgpt config set openai_api_key 'your-key-here'")
            print("2. export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)
        
        return api_key


# Global config manager instance
config_manager = ConfigManager() 