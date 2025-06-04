#!/usr/bin/env python3
"""
Image Generator from Prompt Files or Direct Prompts

This script generates images from text prompt files or direct prompt input using OpenAI's API.
Can be used as a CLI tool or installed via pipx.

Usage:
    # From directory of prompt files
    imgpt generate --dir <prompts_dir> [--output <output_dir>]
    
    # From direct prompt
    imgpt generate "A beautiful sunset over mountains"
    
    # Save to specific file
    imgpt generate "A robot in space" --output robot_space.png
    
    # Configuration management
    imgpt config show
    imgpt config set openai_api_key "your-key-here"
    imgpt config set default_model "dall-e-3"
"""

import os
import sys
import base64
import time
from pathlib import Path
from typing import List, Optional, Annotated

import openai
import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from .config import config_manager, ImageptConfig

# Initialize Typer app and console
app = typer.Typer(
    name="imgpt",
    help="🎨 AI Image Generator - Generate images using OpenAI API from prompt files or direct input",
    rich_markup_mode="rich"
)
config_app = typer.Typer(help="🔧 Configuration management")
app.add_typer(config_app, name="config")

console = Console()





def read_prompt_file(prompt_path: Path) -> str:
    """Read prompt content from a file."""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # If it's a markdown file, try to extract description
    if prompt_path.suffix.lower() == '.md':
        lines = content.split('\n')
        description_lines = []
        in_description = False
        
        for line in lines:
            if line.startswith('**Description:**'):
                in_description = True
                continue
            elif in_description and line.startswith('**'):
                break
            elif in_description and line.strip():
                description_lines.append(line.strip())
        
        # If description found, use it; otherwise use cleaned content
        if description_lines:
            return ' '.join(description_lines)
        else:
            # Remove markdown headers and formatting for a cleaner prompt
            clean_lines = []
            for line in lines:
                if not line.startswith('#') and not line.startswith('**') and line.strip():
                    clean_lines.append(line.strip())
            return ' '.join(clean_lines)
    
    return content


def generate_image(client: openai.OpenAI, prompt: str, filename: str, 
                  model: str = "gpt-image-1", size: str = "1536x1024", 
                  quality: str = "high", style: str = None, 
                  output_format: str = "png") -> bytes:
    """Generate an image using OpenAI's API."""
    rprint(f"🎨 Generating image for [bold]{filename}[/bold]...")
    rprint(f"📝 Prompt: [dim]{prompt[:100]}{'...' if len(prompt) > 100 else ''}[/dim]")
    rprint(f"🔧 Model: [cyan]{model}[/cyan], Size: [cyan]{size}[/cyan], Quality: [cyan]{quality}[/cyan]")
    
    try:
        # Validate and truncate prompt based on model
        max_lengths = {
            "gpt-image-1": 32000,
            "dall-e-2": 1000,
            "dall-e-3": 4000
        }
        
        max_length = max_lengths.get(model, 32000)
        if len(prompt) > max_length:
            prompt = prompt[:max_length] + "..."
            rprint(f"⚠️  Warning: Prompt truncated to {max_length} characters for {model}")
        
        # Build API parameters based on model
        api_params = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "quality": quality
        }
        
        # Add model-specific parameters
        if model == "gpt-image-1":
            if output_format in ["png", "jpeg", "webp"]:
                api_params["output_format"] = output_format
        elif model == "dall-e-3":
            if style in ["vivid", "natural"]:
                api_params["style"] = style
            # dall-e-3 uses response_format instead of output_format
            api_params["response_format"] = "b64_json"
        elif model == "dall-e-2":
            # dall-e-2 uses response_format
            api_params["response_format"] = "b64_json"
        
        # Generate image
        response = client.images.generate(**api_params)
        
        # Decode base64 image data
        image_data = base64.b64decode(response.data[0].b64_json)
        
        rprint(f"✅ Successfully generated image for [bold green]{filename}[/bold green]")
        return image_data
        
    except Exception as e:
        rprint(f"❌ Error generating image for {filename}: {str(e)}")
        raise


def save_image(image_data: bytes, output_path: Path):
    """Save image data to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(image_data)
    
    rprint(f"💾 Saved image to [bold blue]{output_path}[/bold blue]")


def find_prompt_files(prompts_dir: Path) -> List[Path]:
    """Find all prompt files in the directory."""
    prompt_files = []
    
    # Look for various prompt file extensions
    extensions = ['.prompt', '.txt', '.md']
    
    for ext in extensions:
        prompt_files.extend(prompts_dir.glob(f"*{ext}"))
    
    return sorted(prompt_files)


def get_output_path(prompt_file: Path, output_dir: Path, output_format: str = "png") -> Path:
    """Get the output path for a prompt file."""
    # Remove the prompt extension and add the output format extension
    base_name = prompt_file.stem
    return output_dir / f"{base_name}.{output_format}"


def validate_model_size(model: str, size: str) -> bool:
    """Validate that the size is compatible with the model."""
    valid_sizes = {
        "gpt-image-1": ["1024x1024", "1536x1024", "1024x1536"],
        "dall-e-2": ["256x256", "512x512", "1024x1024"],
        "dall-e-3": ["1024x1024", "1792x1024", "1024x1792"]
    }
    
    return size in valid_sizes.get(model, [])


def get_default_size(model: str) -> str:
    """Get default size for a model."""
    size_defaults = {
        "gpt-image-1": "1536x1024",
        "dall-e-2": "1024x1024", 
        "dall-e-3": "1024x1024"
    }
    return size_defaults.get(model, "1536x1024")


@app.command()
def generate(
    prompt: Annotated[Optional[str], typer.Argument(help="Direct prompt text for image generation")] = None,
    dir: Annotated[Optional[str], typer.Option("--dir", help="Directory containing prompt files")] = None,
    output: Annotated[Optional[str], typer.Option("--output", help="Output file path (for direct prompts) or directory (for prompt files)")] = None,
    model: Annotated[Optional[str], typer.Option("--model", help="Model to use for image generation")] = None,
    size: Annotated[Optional[str], typer.Option("--size", help="Image size (e.g., 1024x1024, 1536x1024, 1024x1536)")] = None,
    quality: Annotated[Optional[str], typer.Option("--quality", help="Image quality")] = None,
    style: Annotated[Optional[str], typer.Option("--style", help="Image style for DALL-E 3 (vivid or natural)")] = None,
    format: Annotated[Optional[str], typer.Option("--format", help="Output format")] = None,
    delay: Annotated[Optional[float], typer.Option("--delay", help="Delay between API calls in seconds")] = None,
    skip_existing: Annotated[bool, typer.Option("--skip-existing", help="Skip generating images that already exist")] = False,
):
    """🎨 Generate images from prompts or prompt files."""
    
    # Load configuration and apply defaults
    config = config_manager.load_config()
    
    # Apply configuration defaults if not specified
    model = model or config.default_model
    size = size or config.default_size or get_default_size(model)
    quality = quality or config.default_quality
    style = style or config.default_style
    format = format or config.default_format
    delay = delay if delay is not None else config.default_delay
    skip_existing = skip_existing or config.skip_existing
    
    # Validate inputs
    if not prompt and not dir:
        rprint("❌ Error: Must provide either a prompt or a directory with --dir")
        raise typer.Exit(1)
    
    if prompt and dir:
        rprint("❌ Error: Cannot use both direct prompt and directory mode")
        raise typer.Exit(1)
    
    # Validate model and size compatibility
    if not validate_model_size(model, size):
        valid_sizes = {
            "gpt-image-1": ["1024x1024", "1536x1024", "1024x1536"],
            "dall-e-2": ["256x256", "512x512", "1024x1024"],
            "dall-e-3": ["1024x1024", "1792x1024", "1024x1792"]
        }
        rprint(f"❌ Error: Size '{size}' is not valid for model '{model}'")
        rprint(f"Valid sizes for {model}: {', '.join(valid_sizes[model])}")
        raise typer.Exit(1)
    
    rprint("🤖 [bold]AI Image Generator[/bold]")
    rprint("=" * 50)
    
    # Initialize OpenAI client
    try:
        api_key = config_manager.get_api_key()
        client = openai.OpenAI(api_key=api_key)
        rprint("✅ OpenAI client initialized")
    except Exception as e:
        rprint(f"❌ Failed to initialize OpenAI client: {e}")
        raise typer.Exit(1)
    
    # Handle directory mode
    if dir:
        prompts_dir = Path(dir)
        output_dir = Path(output) if output else (Path(config.default_output_dir) if config.default_output_dir else prompts_dir)
        
        if not prompts_dir.exists():
            rprint(f"❌ Prompts directory does not exist: {prompts_dir}")
            raise typer.Exit(1)
        
        if not prompts_dir.is_dir():
            rprint(f"❌ Prompts path is not a directory: {prompts_dir}")
            raise typer.Exit(1)
        
        rprint(f"📁 Prompts directory: [bold blue]{prompts_dir}[/bold blue]")
        rprint(f"📂 Output directory: [bold blue]{output_dir}[/bold blue]")
        
        # Find all prompt files
        prompt_files = find_prompt_files(prompts_dir)
        if not prompt_files:
            rprint(f"❌ No prompt files found in {prompts_dir}")
            rprint("Looking for files with extensions: .prompt, .txt, .md")
            raise typer.Exit(1)
        
        rprint(f"📁 Found [bold]{len(prompt_files)}[/bold] prompt files")
        
        # Generate images for each prompt file
        success_count = 0
        for prompt_file in prompt_files:
            try:
                output_path = get_output_path(prompt_file, output_dir, format)
                
                # Skip if image already exists and --skip-existing is set
                if skip_existing and output_path.exists():
                    rprint(f"⏭️  Skipping [dim]{prompt_file.name}[/dim] (image already exists)")
                    continue
                
                # Read prompt
                file_prompt = read_prompt_file(prompt_file)
                if not file_prompt.strip():
                    rprint(f"⚠️  Skipping [dim]{prompt_file.name}[/dim] (empty prompt)")
                    continue
                
                # Generate image
                image_data = generate_image(client, file_prompt, prompt_file.name, 
                                          model, size, quality, style, format)
                
                # Save image
                save_image(image_data, output_path)
                success_count += 1
                
                # Rate limiting - be respectful to the API
                if delay > 0:
                    time.sleep(delay)
                
            except Exception as e:
                rprint(f"❌ Failed to process {prompt_file.name}: {e}")
                continue
        
        rprint(f"\n🎉 [bold green]Image generation complete![/bold green]")
        rprint(f"✅ Successfully generated [bold]{success_count}/{len(prompt_files)}[/bold] images")
        rprint(f"📂 Images saved to: [bold blue]{output_dir}[/bold blue]")
    
    # Handle direct prompt mode
    else:
        if output:
            output_path = Path(output)
            # Ensure correct extension for output format
            expected_ext = f".{format}"
            if output_path.suffix.lower() != expected_ext:
                output_path = output_path.with_suffix(expected_ext)
        else:
            # Use default output directory if configured
            output_dir = Path(config.default_output_dir) if config.default_output_dir else Path.cwd()
            # Generate filename from prompt
            safe_name = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_').lower()
            output_path = output_dir / f"{safe_name}.{format}"
        
        rprint(f"📝 Prompt: [dim]{prompt}[/dim]")
        rprint(f"📂 Output: [bold blue]{output_path}[/bold blue]")
        
        try:
            # Generate image
            image_data = generate_image(client, prompt, output_path.name, 
                                      model, size, quality, style, format)
            
            # Save image
            save_image(image_data, output_path)
            
            rprint(f"\n🎉 [bold green]Image generation complete![/bold green]")
            rprint(f"📂 Image saved to: [bold blue]{output_path}[/bold blue]")
            
        except Exception as e:
            rprint(f"❌ Failed to generate image: {e}")
            raise typer.Exit(1)


@config_app.command("show")
def config_show():
    """📋 Show current configuration."""
    config_manager.show_config()


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Configuration key to set")],
    value: Annotated[str, typer.Argument(help="Configuration value to set")]
):
    """⚙️ Set a configuration value."""
    try:
        # Convert string values to appropriate types
        if key in ["default_delay"]:
            value = float(value)
        elif key in ["skip_existing"]:
            value = value.lower() in ["true", "1", "yes", "on"]
        
        config_manager.update_config(**{key: value})
        rprint(f"✅ Set [bold]{key}[/bold] = [cyan]{value}[/cyan]")
    except Exception as e:
        rprint(f"❌ Failed to set configuration: {e}")
        raise typer.Exit(1)


@config_app.command("reset")
def config_reset():
    """🔄 Reset configuration to defaults."""
    if typer.confirm("Are you sure you want to reset all configuration to defaults?"):
        config_manager.reset_config()
        rprint("✅ Configuration reset to defaults")
    else:
        rprint("❌ Configuration reset cancelled")


@config_app.command("path")
def config_path():
    """📁 Show configuration file path."""
    rprint(f"📄 Configuration file: [bold blue]{config_manager.config_file}[/bold blue]")


@app.command()
def version():
    """📋 Show version information."""
    rprint("🎨 [bold]imgpt[/bold] v0.3.0")
    rprint("AI Image Generator with persistent configuration")
    rprint("Made with ❤️  by Jacob Valdez")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main() 