#!/usr/bin/env python3
"""
Image Generator from Prompt Files or Direct Prompts

This script generates images from text prompt files or direct prompt input using OpenAI's API.
Can be used as a CLI tool or installed via pipx.

Usage:
    # From directory of prompt files
    generate-images --dir <prompts_dir> [--output <output_dir>]
    
    # From direct prompt
    generate-images "A beautiful sunset over mountains"
    
    # Save to specific file
    generate-images "A robot in space" --output robot_space.png
"""

import os
import sys
import base64
import time
import argparse
from pathlib import Path
from typing import List, Optional

import openai


def get_api_key() -> str:
    """Get OpenAI API key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please export your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    return api_key


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


def generate_image(client: openai.OpenAI, prompt: str, filename: str) -> bytes:
    """Generate an image using OpenAI's API."""
    print(f"🎨 Generating image for {filename}...")
    print(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    
    try:
        # Truncate prompt if too long for gpt-image-1
        if len(prompt) > 32000:
            prompt = prompt[:32000] + "..."
            print(f"⚠️  Warning: Prompt truncated to 32000 characters")
        
        # Generate image using gpt-image-1
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size="1536x1024",  # Landscape format
            quality="high"
        )
        
        # Decode base64 image data
        image_data = base64.b64decode(response.data[0].b64_json)
        
        print(f"✅ Successfully generated image for {filename}")
        return image_data
        
    except Exception as e:
        print(f"❌ Error generating image for {filename}: {str(e)}")
        raise


def save_image(image_data: bytes, output_path: Path):
    """Save image data to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(image_data)
    
    print(f"💾 Saved image to {output_path}")


def find_prompt_files(prompts_dir: Path) -> List[Path]:
    """Find all prompt files in the directory."""
    prompt_files = []
    
    # Look for various prompt file extensions
    extensions = ['.prompt', '.txt', '.md']
    
    for ext in extensions:
        prompt_files.extend(prompts_dir.glob(f"*{ext}"))
    
    return sorted(prompt_files)


def get_output_path(prompt_file: Path, output_dir: Path) -> Path:
    """Get the output path for a prompt file."""
    # Remove the prompt extension and add .png
    base_name = prompt_file.stem
    return output_dir / f"{base_name}.png"


def generate_from_directory(client: openai.OpenAI, prompts_dir: Path, output_dir: Path, 
                          delay: float, skip_existing: bool) -> int:
    """Generate images from a directory of prompt files."""
    # Find all prompt files
    prompt_files = find_prompt_files(prompts_dir)
    if not prompt_files:
        print(f"❌ No prompt files found in {prompts_dir}")
        print("Looking for files with extensions: .prompt, .txt, .md")
        return 1
    
    print(f"📁 Found {len(prompt_files)} prompt files")
    
    # Generate images for each prompt file
    success_count = 0
    for prompt_file in prompt_files:
        try:
            output_path = get_output_path(prompt_file, output_dir)
            
            # Skip if image already exists and --skip-existing is set
            if skip_existing and output_path.exists():
                print(f"⏭️  Skipping {prompt_file.name} (image already exists)")
                continue
            
            # Read prompt
            prompt = read_prompt_file(prompt_file)
            if not prompt.strip():
                print(f"⚠️  Skipping {prompt_file.name} (empty prompt)")
                continue
            
            # Generate image
            image_data = generate_image(client, prompt, prompt_file.name)
            
            # Save image
            save_image(image_data, output_path)
            success_count += 1
            
            # Rate limiting - be respectful to the API
            if delay > 0:
                time.sleep(delay)
            
        except Exception as e:
            print(f"❌ Failed to process {prompt_file.name}: {e}")
            continue
    
    print(f"\n🎉 Image generation complete!")
    print(f"✅ Successfully generated {success_count}/{len(prompt_files)} images")
    print(f"📂 Images saved to: {output_dir}")
    return 0


def generate_from_prompt(client: openai.OpenAI, prompt: str, output_path: Path) -> int:
    """Generate a single image from a direct prompt."""
    try:
        # Generate image
        image_data = generate_image(client, prompt, output_path.name)
        
        # Save image
        save_image(image_data, output_path)
        
        print(f"\n🎉 Image generation complete!")
        print(f"📂 Image saved to: {output_path}")
        return 0
        
    except Exception as e:
        print(f"❌ Failed to generate image: {e}")
        return 1


def main():
    """Main function to generate images from prompt files or direct prompts."""
    parser = argparse.ArgumentParser(
        description="Generate images using OpenAI API from prompt files or direct input",
        prog="imgpt"
    )
    
    # Mutually exclusive group for input method
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "prompt", 
        nargs='?', 
        help="Direct prompt text for image generation"
    )
    input_group.add_argument(
        "--dir", 
        dest="prompts_dir",
        help="Directory containing prompt files"
    )
    
    parser.add_argument(
        "--output", 
        help="Output file path (for direct prompts) or directory (for prompt files)"
    )
    parser.add_argument(
        "--delay", 
        type=float, 
        default=2.0, 
        help="Delay between API calls in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--skip-existing", 
        action="store_true", 
        help="Skip generating images that already exist"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version="%(prog)s 0.1.0"
    )
    
    args = parser.parse_args()
    
    print("🤖 AI Image Generator")
    print("=" * 50)
    
    # Initialize OpenAI client
    try:
        api_key = get_api_key()
        client = openai.OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")
        return 1
    
    # Handle directory mode
    if args.prompts_dir:
        prompts_dir = Path(args.prompts_dir)
        output_dir = Path(args.output) if args.output else prompts_dir
        
        if not prompts_dir.exists():
            print(f"❌ Prompts directory does not exist: {prompts_dir}")
            return 1
        
        if not prompts_dir.is_dir():
            print(f"❌ Prompts path is not a directory: {prompts_dir}")
            return 1
        
        print(f"📁 Prompts directory: {prompts_dir}")
        print(f"📂 Output directory: {output_dir}")
        
        return generate_from_directory(
            client, prompts_dir, output_dir, args.delay, args.skip_existing
        )
    
    # Handle direct prompt mode
    else:
        if args.output:
            output_path = Path(args.output)
            # Ensure .png extension
            if output_path.suffix.lower() != '.png':
                output_path = output_path.with_suffix('.png')
        else:
            # Generate filename from prompt
            safe_name = "".join(c for c in args.prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_').lower()
            output_path = Path(f"{safe_name}.png")
        
        print(f"📝 Prompt: {args.prompt}")
        print(f"📂 Output: {output_path}")
        
        return generate_from_prompt(client, args.prompt, output_path)


if __name__ == "__main__":
    exit(main()) 