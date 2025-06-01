# imGPT 🎨

A powerful CLI tool for generating images using OpenAI's API. Generate images from text prompts directly or process entire directories of prompt files.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI API](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/api/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🚀 **Direct Prompt Generation**: Generate images from command-line prompts
- 📁 **Batch Processing**: Process entire directories of prompt files
- 🎯 **Multiple Formats**: Support for `.prompt`, `.txt`, and `.md` files
- 🔄 **Smart Skipping**: Skip existing images to save time and API costs
- ⚡ **Rate Limiting**: Configurable delays to respect API limits
- 🎨 **High Quality**: Uses OpenAI's gpt-image-1 model for best results
- 📦 **Easy Install**: Install globally with pipx

## 🚀 Quick Start

### Install with pipx (Recommended)

```bash
pipx install imgpt
```

### Install with pip

```bash
pip install imgpt
```

### Set up your API key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Generate your first image

```bash
imgpt "A majestic dragon flying over a medieval castle at sunset"
```

## 📖 Usage

### Direct Prompt Generation

Generate a single image from a text prompt:

```bash
# Basic usage
imgpt "A cute robot playing guitar"

# Save to specific file
imgpt "A space station orbiting Earth" --output space_station.png

# Custom output location
imgpt "Abstract art with vibrant colors" --output ./art/abstract.png
```

### Batch Processing from Directory

Process multiple prompt files at once:

```bash
# Process all prompt files in a directory
imgpt --dir ./my_prompts

# Save to different output directory
imgpt --dir ./prompts --output ./generated_images

# Skip existing images and use faster processing
imgpt --dir ./prompts --skip-existing --delay 1
```

## 📁 Supported File Formats

### `.prompt` files
```
A beautiful sunset over snow-capped mountains with a lake reflection
```

### `.txt` files
```
A futuristic cityscape with flying cars and neon lights
```

### `.md` files (with special parsing)
```markdown
# Image Description

**Description:**
A serene Japanese garden with cherry blossoms, a small bridge over a koi pond, and traditional lanterns. The scene should be peaceful and zen-like.

**Style:** Photorealistic
**Mood:** Tranquil
```

## 🛠️ Command Line Options

```
imgpt [OPTIONS] [PROMPT]

Arguments:
  PROMPT                    Direct prompt text for image generation

Options:
  --dir PATH               Directory containing prompt files
  --output PATH            Output file/directory path
  --delay FLOAT            Delay between API calls in seconds (default: 2.0)
  --skip-existing          Skip generating images that already exist
  --model MODEL            Model to use: gpt-image-1, dall-e-2, dall-e-3 (default: gpt-image-1)
  --size SIZE              Image dimensions (e.g., 1024x1024, 1536x1024, 1024x1536)
  --quality QUALITY        Image quality: auto, high, medium, low, hd, standard (default: high)
  --style STYLE            Image style for DALL-E 3: vivid, natural
  --format FORMAT          Output format for gpt-image-1: png, jpeg, webp (default: png)
  --version                Show version and exit
  --help                   Show help message and exit
```

## 📋 Examples

### Single Image Generation

```bash
# Simple prompt
imgpt "A red sports car"

# Complex prompt with details
imgpt "A detailed oil painting of a lighthouse on a rocky cliff during a storm, dramatic lighting, high contrast"

# Save with custom name
imgpt "A minimalist logo design" --output company_logo.png

# Use different models and settings
imgpt "A futuristic cityscape" --model dall-e-3 --size 1792x1024 --style vivid

# Generate portrait orientation
imgpt "A portrait of a wise old wizard" --size 1024x1536

# Use DALL-E 2 for faster generation
imgpt "A simple cartoon cat" --model dall-e-2 --size 512x512

# Generate JPEG format
imgpt "A landscape photo" --format jpeg --quality high
```

### Batch Processing

```bash
# Process directory (saves images alongside prompts)
imgpt --dir ./product_descriptions

# Separate input/output directories
imgpt --dir ./marketing_prompts --output ./marketing_images

# Production settings (skip existing, faster processing)
imgpt --dir ./prompts --output ./images --skip-existing --delay 0.5

# Batch process with DALL-E 3 for high quality
imgpt --dir ./art_prompts --model dall-e-3 --quality hd --style natural

# Generate thumbnails with DALL-E 2
imgpt --dir ./thumbnails --model dall-e-2 --size 256x256

# Batch process with custom format and quality
imgpt --dir ./web_images --format webp --quality medium --delay 1
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | Your OpenAI API key |

### Image Settings

The tool supports multiple models and configurations:

#### Models
- **gpt-image-1** (default): OpenAI's latest image model
  - Sizes: 1024x1024, 1536x1024 (landscape), 1024x1536 (portrait)
  - Quality: auto, high, medium, low
  - Formats: png, jpeg, webp
- **dall-e-3**: High-quality artistic images
  - Sizes: 1024x1024, 1792x1024 (landscape), 1024x1792 (portrait)
  - Quality: auto, hd, standard
  - Styles: vivid, natural
- **dall-e-2**: Fast and cost-effective
  - Sizes: 256x256, 512x512, 1024x1024
  - Quality: standard only

## 📦 Installation Methods

### Method 1: pipx (Recommended)

```bash
# Install globally without affecting system Python
pipx install imgpt

# Upgrade
pipx upgrade imgpt

# Uninstall
pipx uninstall imgpt
```

### Method 2: pip

```bash
# Install globally
pip install imgpt

# Install in virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install imgpt
```

### Method 3: Development Install

```bash
git clone https://github.com/humanrobots-ai/imgpt.git
cd imgpt
poetry install
poetry run imgpt "test prompt"
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=imgpt

# Run specific test
poetry run pytest tests/test_cli.py::test_read_prompt_file_simple
```

### Code Quality

```bash
# Format code
poetry run black src/

# Lint code
poetry run flake8 src/
```

## 🚨 Error Handling

The tool gracefully handles various error conditions:

- **Missing API Key**: Clear instructions for setting up authentication
- **Empty Prompts**: Skips empty files with warnings
- **API Errors**: Continues processing other files if one fails
- **Network Issues**: Retries with exponential backoff
- **Invalid Paths**: Validates input/output directories

## 💡 Tips & Best Practices

### Writing Better Prompts

1. **Be Specific**: Include details about style, lighting, composition
2. **Use Descriptive Language**: "vibrant", "detailed", "photorealistic"
3. **Specify Art Style**: "oil painting", "digital art", "photograph"
4. **Include Mood**: "serene", "dramatic", "whimsical"

### Batch Processing

1. **Organize Prompts**: Use descriptive filenames for easy identification
2. **Use Skip Existing**: Avoid regenerating images unnecessarily
3. **Adjust Delays**: Balance speed vs. API rate limits
4. **Separate Outputs**: Keep generated images organized

### Cost Management

1. **Preview Prompts**: Review prompts before batch processing
2. **Use Skip Existing**: Avoid duplicate generations
3. **Test Single Images**: Verify prompts work before batch runs
4. **Monitor Usage**: Track API usage in OpenAI dashboard

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📧 **Email**: jacobfv123@gmail.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/humanrobots-ai/imgpt/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/humanrobots-ai/imgpt/discussions)

## 🙏 Acknowledgments

- OpenAI for providing the amazing image generation API
- The Python community for excellent tooling and libraries
- All contributors and users of this tool

---

Made with ❤️ by [Jacob Valdez](https://github.com/jacobfv123) 