# Building the Transformer from Scratch — Interactive Presentation

A deployment-ready interactive presentation for the "Building the Transformer from Scratch" seminar. This project includes a custom-built single-page application with D3.js simulations and an AI image generation pipeline.

## Project Overview

- **Goal**: Visually explain the evolution of LLMs from historical statistics to modern Transformers.
- **Tech Stack**: HTML5, D3.js, MathJax, Google Gemini API (for imagery).
- **Deployment**: Design for GitHub Pages.

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Images**
   You need a Google Gemini API key.
   ```bash
   # Generate all images (may take a minute)
   python generate_images.py --api-key YOUR_API_KEY
   
   # Or just create placeholders for testing
   python generate_images.py --dry-run
   ```

3. **Run locally**
   Simply open `index.html` in your browser. No server required (though a local server is recommended for better font loading/caching behavior).
   ```bash
   # Optional
   python -m http.server
   ```

## Controls

- **Arrows / Space**: Navigate slides
- **F**: Toggle Fullscreen
- **S**: Toggle Speaker Notes
- **O**: Overview Mode
- **G**: Go to slide
- **Esc**: Close dialogs

## Deployment

1. Commit all files.
2. Push to GitHub.
3. Enable GitHub Pages on the repository (Settings -> Pages -> Source: main branch).
4. The `.nojekyll` file ensures `_` prefixed directories are served correctly (though we aren't using many).

## License

MIT
