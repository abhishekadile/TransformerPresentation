
import os
import argparse
import time
import json
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Style Suffix
STYLE_SUFFIX = """
Style: Modern digital illustration, clean vector-like aesthetic with depth. 
Color palette: deep navy (#0f172a) and charcoal backgrounds, sky blue (#38bdf8) 
and amber (#f59e0b) accents, white highlights. Professional and polished, 
suitable for a tech conference presentation. No text or labels in the image. 
Widescreen 16:9 aspect ratio composition.
"""

IMAGE_SPECS = [
    {
        "filename": "slide_01_title.png",
        "prompt": """A dramatic wide shot of a neural network visualization transforming 
        into a transformer architecture. Abstract geometric patterns of interconnected 
        nodes on the left morphing into organized parallel attention layers on the right. 
        Glowing data streams flowing through the structure. Deep space-like background 
        with constellation-like node connections. Epic, cinematic scale suggesting the 
        grandeur of the technology. A single bright focal point at the center where 
        old meets new architecture."""
    },
    {
        "filename": "slide_02_ancient_data.png",
        "prompt": """A montage composition showing ancient civilizations collecting data. 
        Left section: Roman census officials with scrolls and tablets in a marble forum. 
        Center: Chinese Han Dynasty administrators with population registers and 
        counting rods. Right: Indian scholars with Arthashastra manuscripts. 
        All three scenes connected by flowing golden data streams that suggest the 
        continuity of data collection through history. Ancient architecture elements 
        frame each scene. Warm, historical color palette transitioning to cooler 
        modern tones at the edges."""
    },
    {
        "filename": "slide_03_hollerith.png",
        "prompt": """A dramatic scene showing Herman Hollerith's punch card tabulating 
        machine from the 1890 US Census. The machine is rendered in detailed mechanical 
        glory with brass gears and electrical contacts. Punch cards are flowing through 
        the machine. In the background, a mountain of paper census forms contrasts with 
        the elegant efficiency of the machine. Subtle visual connection: the punch card 
        holes morph into binary digits (0s and 1s) at the edges, bridging to modern 
        computing. Warm industrial lighting, steampunk-adjacent aesthetic."""
    },
    {
        "filename": "slide_04_statistics_founders.png",  
        "prompt": """An elegant composition showing the key figures of statistical 
        history arranged around a central bell curve visualization. Silhouettes or 
        stylized portraits of Gauss, Bayes, Bernoulli, and Laplace positioned at 
        cardinal points. Mathematical formulas and probability distributions float 
        between them like constellations. The bell curve at center is rendered as a 
        luminous 3D surface. Historical manuscript textures blend with modern data 
        visualization aesthetics. Each figure emanates their key contribution as 
        glowing mathematical notation."""
    },
    {
        "filename": "slide_05_pascal_dice.png",
        "prompt": """A richly detailed scene of 17th century French gambling parlor. 
        Ornate dice and playing cards on a polished wooden table. Two figures in 
        period clothing (representing Pascal and Fermat) lean over the table in 
        animated discussion, with mathematical probability trees and equations floating 
        above them as translucent overlays. Candlelight illumination. The dice faces 
        show probability distributions instead of dots. A letter being written 
        between them glows with mathematical formulas. Rich, warm Baroque palette."""
    },
    {
        "filename": "slide_06_markov_nekrasov.png",
        "prompt": """A dramatic split-panel composition. Left side: an Orthodox church 
        with golden onion domes in winter St. Petersburg, representing Nekrasov the 
        theologian. Right side: a mathematics lecture hall with chalkboards covered 
        in equations, representing Markov the mathematician. In the center where they 
        meet: a clash of ideas visualized as colliding wave patterns — one organic 
        and flowing (theology/free will), one structured and mathematical (probability 
        chains). The year 1913 subtly embedded in the architecture. Dramatic lighting 
        from both sides meeting at the center."""
    },
    {
        "filename": "slide_07_eugene_onegin.png",
        "prompt": """An artistic rendering of Pushkin's Eugene Onegin manuscript pages 
        with Russian Cyrillic text. Letters are being analyzed — vowels glow blue, 
        consonants glow amber. A transition probability matrix emerges from the text 
        like a holographic overlay, showing the patterns Markov discovered. Flowing 
        arrows connect letters showing dependencies. The manuscript pages curve and 
        transform into a mathematical chain diagram at the bottom. Russian literary 
        elegance meets mathematical precision. Atmospheric, scholarly lighting."""
    },
    {
        "filename": "slide_09_perceptron.png",
        "prompt": """A retro-futuristic visualization of the Mark I Perceptron from 1958. 
        The machine is rendered as a beautiful piece of mid-century modern technology — 
        all chrome, knobs, and patch cables. A single artificial neuron diagram 
        is overlaid, showing inputs converging through weights to a summation point 
        and threshold activation. Newspaper headlines about the Perceptron float 
        around it (but text is abstracted/blurred). 1950s optimistic space-age 
        aesthetic. The neuron diagram glows with potential energy."""
    },
    {
        "filename": "slide_10_xor_ai_winter.png",
        "prompt": """A dramatic scene showing the XOR problem as a battlefield. Four 
        data points arranged in the classic XOR pattern (two classes that can't be 
        linearly separated) hover in a dark space. A single glowing line tries to 
        separate them but fails — shown breaking apart. In the background, a 
        winter storm approaches: neural network diagrams freezing over with ice 
        crystals, research papers covered in snow. The 'AI Winter' visualized as 
        literal frozen technology. A book titled 'Perceptrons' casts a long shadow. 
        Cold blue palette with warm fragments dying out."""
    },
    {
        "filename": "slide_11_backprop.png",
        "prompt": """A beautiful visualization of backpropagation through a neural 
        network. A multi-layer network is rendered as a crystal-like structure with 
        nodes as luminous spheres connected by fiber-optic-like edges. On the forward 
        pass (left to right), warm golden light flows through the network. On the 
        backward pass (right to left), cool blue gradient signals flow back, each 
        node pulsing as it receives error information. The chain rule is visualized 
        as the gradient signal splitting and propagating. A clear before/after: 
        wrong output on the right correcting as gradients flow. Dynamic, flowing 
        energy aesthetic."""
    },
    {
        "filename": "slide_13_rnn_memory.png",
        "prompt": """A visual metaphor for RNN sequential processing and memory loss. 
        A long sentence 'The cat sat on the mat' is shown as a pathway of connected 
        chambers. A glowing orb (the hidden state) travels from left to right through 
        each chamber. With each step, the orb dims and loses color — by the last 
        word, it's barely visible. The first words are vibrant and detailed, the 
        last words are faded. Time flows left to right with clock-like arrows. 
        Each chamber has a loop-back arrow suggesting recurrence. Watercolor-fade 
        effect showing information degradation."""
    },
    {
        "filename": "slide_14_lstm_gates.png",
        "prompt": """A detailed architectural diagram of an LSTM cell rendered as 
        a sophisticated machine. Three gates are shown as distinct mechanisms: 
        the forget gate as a valve releasing old information (red), the input 
        gate as a funnel accepting new information (green), the output gate as 
        a lens focusing the output (blue). The cell state runs through the center 
        as a highway/conveyor belt. Sigmoid and tanh activations are shown as 
        transformation chambers. Clean, technical illustration style with subtle 
        3D depth. Blueprint-meets-infographic aesthetic."""
    },
    {
        "filename": "slide_18_attention_paper.png",
        "prompt": """A dramatic 'movie poster' style composition for the 2017 paper 
        'Attention Is All You Need'. The iconic transformer architecture diagram 
        is rendered as a towering monumental structure, like a futuristic building 
        or spacecraft. Eight attention heads are visualized as spotlights emanating 
        from the structure, each illuminating a different aspect of a sentence below. 
        The old sequential RNN architecture crumbles in the background. Lightning 
        bolts of parallel computation replace the slow chain of sequential processing. 
        Epic, revolutionary tone. The dawn of a new era in AI."""
    },
    {
        "filename": "slide_30_architectures.png",
        "prompt": """A clean architectural comparison of three transformer variants 
        side by side. Left: Encoder-only (BERT) shown as a bidirectional structure 
        with arrows going both ways, glowing symmetrically. Center: Decoder-only 
        (GPT) shown as a unidirectional structure with a causal mask creating a 
        triangular pattern, autoregressive arrows flowing one direction. Right: 
        Encoder-Decoder (T5/original Transformer) showing both components connected 
        by cross-attention bridges. Each variant is rendered as a distinct tower 
        with its own character. Clean, technical, but visually striking. 
        Each labeled with its key use case as an icon (search, writing, translation)."""
    },
    {
        "filename": "slide_33_bert_vs_gpt.png",
        "prompt": """A side-by-side comparison of BERT and GPT as two contrasting 
        characters or machines. BERT on the left: a detective/analyst figure that 
        looks at everything from all angles simultaneously, with bidirectional 
        arrows and magnifying glass motifs. Applications float around it: search, 
        Q&A, classification. GPT on the right: a writer/creator figure that 
        generates forward, with a quill or typewriter motif, unidirectional 
        flowing text. Applications: writing, coding, dialogue. They share the 
        same transformer DNA (shown as matching core architecture) but diverge 
        in purpose and training. Diptych composition with complementary colors."""
    },
    {
        "filename": "slide_34_hackathon_cta.png",
        "prompt": """An energetic, inspiring composition showing hands on keyboards 
        with code on screens, transformer architecture diagrams, and a competitive 
        leaderboard. Multiple workstations arranged in a collaborative space. 
        A large screen shows a live leaderboard with progress bars. Code snippets 
        float in the air as holographic displays. The atmosphere is focused but 
        exciting — a hackathon in full swing. Warm collaborative lighting. 
        Google Colab notebooks visible on screens. A countdown timer adds urgency. 
        Team energy, focused concentration, competitive drive."""
    }
]

def create_placeholder(filename, directory="images"):
    """Creates a placeholder image using Pillow."""
    width, height = 1920, 1080
    background_color = (15, 23, 42)  # #0f172a
    text_color = (255, 255, 255)
    subtext_color = (148, 163, 184) # #94a3b8
    grid_color = (30, 41, 59) # #1e293b

    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    # Draw grid
    grid_size = 60
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    
    # Try to load a font, otherwise use default
    try:
        # Try a few common fonts
        font_paths = [
            "arial.ttf", "segoeui.ttf", "OpenSans-Regular.ttf", 
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        ]
        font = None
        for path in font_paths:
            try: 
                font = ImageFont.truetype(path, 80)
                sub_font = ImageFont.truetype(path, 40)
                break
            except: continue
            
        if font is None: raise Exception("No font found")
    except:
        font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    # Draw text
    title = filename.replace("slide_", "").replace(".png", "").replace("_", " ").title()
    
    # Center text
    # basic centering for default font or pil usage
    try:
        _, _, w, h = draw.textbbox((0, 0), title, font=font)
        draw.text(((width - w) / 2, (height - h) / 2 - 40), title, font=font, fill=text_color)
        
        subtext = "Image pending generation"
        _, _, w2, h2 = draw.textbbox((0, 0), subtext, font=sub_font)
        draw.text(((width - w2) / 2, (height - h) / 2 + 60), subtext, font=sub_font, fill=subtext_color)
    except Exception as e:
        # Fallback for old PIL versions
        draw.text((width/2 - 200, height/2), title, fill=text_color)
        draw.text((width/2 - 100, height/2 + 50), "Pending", fill=subtext_color)

    filepath = Path(directory) / filename
    image.save(filepath)
    logger.info(f"Created placeholder: {filepath}")

def generate_image(spec, api_key, directory="images", force=False, dry_run=False):
    """Generates a single image using Gemini API."""
    filename = spec["filename"]
    filepath = Path(directory) / filename
    
    if filepath.exists() and not force:
        logger.info(f"Skipping {filename} (exists)")
        return

    if dry_run:
        logger.info(f"[Dry Run] Would generate {filename}")
        create_placeholder(filename, directory)
        return

    prompt = spec["prompt"] + "\n\n" + STYLE_SUFFIX
    
    logger.info(f"Generating {filename}...")
    
    # Retry logic
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            genai.configure(api_key=api_key)
            # Use the correct model for image generation
            model = genai.GenerativeModel('gemini-2.5-flash-image') 
            
            response = model.generate_content(
                prompt
            )
            
            # Extract image
            # Look for image part
            if not response.parts:
                raise Exception("No parts in response")
                
            img_part = None
            for part in response.parts:
                # Actual implementation might vary based on latest library version structure
                # This logic attempts to find the inline data
                if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                    img_part = part
                    break
                # Fallback for checking if it's strictly an image object if library handles it differently
            
            if not img_part:
               # Try accessing candidates if direct parts failed or check API specific response structure
               if response.candidates and response.candidates[0].content.parts:
                   for part in response.candidates[0].content.parts:
                       if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                           img_part = part
                           break
            
            if img_part:
                img_data = img_part.inline_data.data
                with open(filepath, "wb") as f:
                    f.write(img_data)
                logger.info(f"Successfully saved {filepath}")
                return
            else:
                raise Exception("No valid image found in response")

        except Exception as e:
            logger.error(f"Attempt {attempt+1}/{max_retries} failed for {filename}: {str(e)}")
            if attempt < max_retries - 1:
                sleep_time = base_delay * (2 ** attempt)
                time.sleep(sleep_time)
            
    # If all retries fail, create placeholder
    logger.warning(f"Failed to generate {filename}, creating placeholder.")
    create_placeholder(filename, directory)

def main():
    parser = argparse.ArgumentParser(description="Generate presentation images using Gemini.")
    parser.add_argument("--api-key", help="Google API Key")
    parser.add_argument("--slides", help="Comma-separated list of slide numbers to generate (e.g., 1,3,6)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing images")
    parser.add_argument("--dry-run", action="store_true", help="Create placeholders only")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        logger.error("API Key must be provided via --api-key or GOOGLE_API_KEY env var (unless --dry-run)")
        return

    # Create images directory
    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)
    
    # Filter slides if requested
    specs_to_process = IMAGE_SPECS
    if args.slides:
        indices = [int(s.strip()) for s in args.slides.split(",")]
        # Match filenames like slide_01...
        filtered_specs = []
        for spec in IMAGE_SPECS:
            try:
                # Extract number from filename "slide_01_title.png"
                num = int(spec["filename"].split("_")[1])
                if num in indices:
                    filtered_specs.append(spec)
            except:
                pass
        specs_to_process = filtered_specs

    logger.info(f"Processing {len(specs_to_process)} images...")

    for spec in specs_to_process:
        generate_image(spec, api_key, directory="images", force=args.force, dry_run=args.dry_run)
        if not args.dry_run:
            # Polite delay between requests
            time.sleep(2)

if __name__ == "__main__":
    main()
