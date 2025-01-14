import os
import random
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import gradio as gr

# Initialize pipeline
def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe = pipe.to(device)
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    return pipe

pipeline = load_pipeline()

# Prompt Building Logic
def get_color_name(r, g, b):
    maximum = max(r, g, b)
    minimum = min(r, g, b)
    brightness = maximum / 255.0

    if brightness < 0.2: return "dark"
    if brightness > 0.8: return "light"

    if r > g + 50 and r > b + 50: return "red"
    if g > r + 50 and g > b + 50: return "green"
    if b > r + 50 and b > g + 50: return "blue"
    if r > b and g > b and abs(r - g) < 20: return "yellow"
    if abs(r - g) < 30 and abs(g - b) < 30: return "gray"

    return "colorful"

def build_prompt(room_type, change_options, aesthetic, color, wood_type):
    base_prompt = f"A beautifully designed {room_type}"
    changes = f" with {', '.join(change_options)}" if change_options else ""
    style = f" in {aesthetic} style" if aesthetic else ""

    if color and color.startswith('#'):
        r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        color_name = get_color_name(r, g, b)
        color_scheme = f" featuring {color_name} tones (RGB: {r},{g},{b})"
    else:
        color_scheme = ""

    wood_detail = f" with {wood_type} wood finishes" if wood_type else ""

    final_prompt = (f"{base_prompt}{changes}{style}{color_scheme}{wood_detail}. "
                   "Keep the windows where they are. Apply the color scheme to furniture, walls, and decor elements.")
    return final_prompt

def safe_generate(input_image, room_type, change_options, aesthetic, color, wood_type):
    if input_image is None:
        raise ValueError("No input image provided")

    prompt = build_prompt(room_type, change_options, aesthetic, color, wood_type)
    input_image = input_image.convert("RGB").resize((512, 512))

    result = pipeline(
        prompt=prompt,
        image=input_image,
        strength=0.7,
        guidance_scale=10,
        num_inference_steps=60,
        negative_prompt="wrong colors, mismatched colors, conflicting color scheme"
    )

    return result.images[0]

# Style quiz data
questions = [
    {
        "question": "What is your preferred color palette?",
        "options": ["Neutral tones", "Bold colors", "Soft pastels"],
        "styles": {"Neutral tones": "Minimalist", "Bold colors": "Modern", "Soft pastels": "Cottage Core"}
    },
    {
        "question": "Which type of furniture appeals to you the most?",
        "options": ["Clean lines and simplicity", "Vintage and ornate", "Functional and cozy"],
        "styles": {"Clean lines and simplicity": "Scandinavian", "Vintage and ornate": "Classic", "Functional and cozy": "Industrial"}
    },
    {
        "question": "What type of flooring do you like?",
        "options": ["Hardwood", "Carpet", "Polished concrete"],
        "styles": {"Hardwood": "Mid-Century Modern", "Carpet": "Luxurious", "Polished concrete": "Industrial"}
    },
    {
        "question": "What type of lighting appeals to you?",
        "options": ["Bright and airy", "Soft and warm", "Focused and functional"],
        "styles": {"Bright and airy": "Modern", "Soft and warm": "Cottage Core", "Focused and functional": "Industrial"}
    },
    {
        "question": "Which wall decor do you prefer?",
        "options": ["Abstract artwork", "Rustic signs", "Minimalist frames"],
        "styles": {"Abstract artwork": "Modern", "Rustic signs": "Cottage Core", "Minimalist frames": "Minimalist"}
    },
    {"question": "What's your ideal window treatment?",
     "options": ["Bare windows", "Heavy drapes", "Light sheer curtains"],
     "styles": {"Bare windows": "Minimalist", "Heavy drapes": "Luxurious", "Light sheer curtains": "Cottage Core"}},

    {"question": "How do you prefer to organize your space?",
     "options": ["Hidden storage", "Open shelving", "Vintage cabinets"],
     "styles": {"Hidden storage": "Minimalist", "Open shelving": "Industrial", "Vintage cabinets": "Classic"}},

    {"question": "What's your preferred accent material?",
     "options": ["Metal", "Natural wood", "Glass"],
     "styles": {"Metal": "Industrial", "Natural wood": "Scandinavian", "Glass": "Modern"}},

    {"question": "What kind of patterns do you prefer?",
     "options": ["Geometric", "Floral", "None"],
     "styles": {"Geometric": "Modern", "Floral": "Cottage Core", "None": "Minimalist"}},

    {"question": "What's your ideal room ambiance?",
     "options": ["Sleek and modern", "Warm and cozy", "Clean and minimal"],
     "styles": {"Sleek and modern": "Modern", "Warm and cozy": "Cottage Core", "Clean and minimal": "Minimalist"}},
    {"question": "What type of textiles appeal to you most?",
     "options": ["Floral prints and lace", "Rich velvets and silk", "Antique tapestries"],
     "styles": {"Floral prints and lace": "Cottage Core", "Rich velvets and silk": "Luxurious", "Antique tapestries": "Vintage"}},

    {"question": "Choose your preferred decorative elements:",
     "options": ["Dried flowers and botanicals", "Crystal and gold accents", "Vintage maps and postcards"],
     "styles": {"Dried flowers and botanicals": "Cottage Core", "Crystal and gold accents": "Luxurious", "Vintage maps and postcards": "Vintage"}},

    {"question": "What's your ideal ceiling fixture?",
     "options": ["Exposed beam with simple lights", "Crystal chandelier", "Antique brass fixture"],
     "styles": {"Exposed beam with simple lights": "Cottage Core", "Crystal chandelier": "Luxurious", "Antique brass fixture": "Vintage"}},

    {"question": "Select your preferred color combination:",
     "options": ["Muted pastels with floral accents", "Gold and deep jewel tones", "Faded colors with patina"],
     "styles": {"Muted pastels with floral accents": "Cottage Core", "Gold and deep jewel tones": "Luxurious", "Faded colors with patina": "Vintage"}},

    {"question": "What's your ideal seating style?",
     "options": ["Plush cushions with floral patterns", "Tufted velvet upholstery", "Weathered leather armchairs"],
     "styles": {"Plush cushions with floral patterns": "Cottage Core", "Tufted velvet upholstery": "Luxurious", "Weathered leather armchairs": "Vintage"}},

    {
            "question": "What's your preferred hardware finish?",
            "options": ["Brass and copper", "Matte black", "Brushed nickel"],
            "styles": {"Brass and copper": "Classic", "Matte black": "Industrial", "Brushed nickel": "Scandinavian"}
        },
        {
            "question": "Choose your ideal room divider:",
            "options": ["Floating shelves", "Glass partition", "Wooden screen"],
            "styles": {"Floating shelves": "Scandinavian", "Glass partition": "Modern", "Wooden screen": "Mid-Century Modern"}
        },
        {
            "question": "What's your preferred storage solution?",
            "options": ["Built-in cabinets", "Industrial shelving", "Modular units"],
            "styles": {"Built-in cabinets": "Classic", "Industrial shelving": "Industrial", "Modular units": "Mid-Century Modern"}
        },
        {
            "question": "Select your ideal surface finish:",
            "options": ["Polished wood", "Terrazzo", "Concrete"],
            "styles": {"Polished wood": "Mid-Century Modern", "Terrazzo": "Modern", "Concrete": "Industrial"}
        },
        {
            "question": "What's your preferred architectural detail?",
            "options": ["Exposed beams", "Crown molding", "Clean edges"],
            "styles": {"Exposed beams": "Industrial", "Crown molding": "Classic", "Clean edges": "Modern"}
        },
        {
            "question": "Choose your ideal furniture leg style:",
            "options": ["Hairpin legs", "Tapered wood", "Sculptural metal"],
            "styles": {"Hairpin legs": "Mid-Century Modern", "Tapered wood": "Scandinavian", "Sculptural metal": "Modern"}
        },
        {
            "question": "What's your preferred cabinet style?",
            "options": ["Shaker style", "Flat panel", "Glass front"],
            "styles": {"Shaker style": "Classic", "Flat panel": "Scandinavian", "Glass front": "Mid-Century Modern"}
        }

]

# Fixed quiz calculation function
def calculate_style_friendly(*args):
    # Convert args tuple to list and handle None values
    answers = ['' if a is None else a for a in args]

    # Validate all questions are answered
    if '' in answers or len(answers) != len(questions):
        return "Please answer all questions", "You must complete all questions to get your style recommendation."

    # Calculate style votes
    style_votes = {}
    for i, answer in enumerate(answers):
        if answer in questions[i]["styles"]:
            style = questions[i]["styles"][answer]
            style_votes[style] = style_votes.get(style, 0) + 1

    if not style_votes:
        return "Error", "Unable to determine style preferences. Please try again."

    # Determine winning style
    max_votes = max(style_votes.values())
    winning_styles = [style for style, votes in style_votes.items() if votes == max_votes]

    # Format results
    style_summary = "\n".join([
        f"• {style}: {votes} match{'es' if votes != 1 else ''}"
        for style, votes in sorted(style_votes.items(), key=lambda x: (-x[1], x[0]))
    ])

    return winning_styles[0], style_summary

# UI Styling
custom_css = """
.container { max-width: 1200px; margin: auto; padding: 20px; color:#1c1a1a;}
.header { background: linear-gradient(135deg, #1a1a1a, #2a2a2a); padding: 30px; border-radius: 15px; margin-bottom: 25px; border: 3px solid #8B0000; text-align: center; }
.header h1 { color: #ffffff; margin: 0; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); font-weight: bold; letter-spacing: 2px; }
.header p { color: #ffffff; font-size: 1.2em; margin-top: 10px; }
.input-group {background: #ffffff; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: 3px solid #8B0000; }
.generate-btn { background: linear-gradient(135deg, #8B0000, #660000); color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: bold; width: 100%; }
.generate-btn:hover { background: linear-gradient(135deg, #660000, #400000); }
.quiz-question { color: #1c1a1a; margin: 10px 0; font-weight: bold; }
.quiz-option { background: #1c1a1a; padding: 10px; border-radius: 5px; margin: 5px 0; color: #333333; }
.result-box { background: #472929; color: #000000; padding: 15px; border-radius: 8px; margin-top: 20px; border: 1px solid #cccccc; }
.markdown-text { color: #1c1a1a; }
"""

# Gradio Interface
with gr.Blocks(css=custom_css) as demo:
     with gr.Tab("Main Page"):
        # Main Page Header
        with gr.Row(elem_classes=["header"]):
            gr.Markdown("# 🎨 Room Transformation Studio")
            gr.Markdown("Transform your space with AI-powered interior design")

        # Main Page Content
        with gr.Row():
            with gr.Column(elem_classes=["input-group"]):
                input_image = gr.Image(label="Upload Room Image", type="pil")
                with gr.Row():
                    room_type = gr.Dropdown(
                        choices=["Bedroom", "Living Room", "Playroom", "Office", "Bathroom", "Kitchen", "Dining Room"],
                        label="Room Type", value="Living Room")
                    aesthetic = gr.Dropdown(
                        choices=["Modern", "Classic", "Cottage Core", "Luxurious", "Mid-Century Modern",
                                 "Retro-Futuristic", "Vintage", "Minimalist", "Industrial", "Scandinavian"],
                        label="Design Style", value="Modern")
                change_options = gr.CheckboxGroup(
                    choices=["Colors", "Furniture", "Lighting", "Decor"],
                    label="Elements to Change", value=["Colors", "Furniture"], elem_classes=["checkbox-group"])
                with gr.Row():
                    color_picker = gr.ColorPicker(label="Preferred Color")
                    wood_type = gr.Dropdown(
                        choices=["Dark Oak", "Light Oak", "Walnut", "Mahogany", "Pine", "Maple", "Cherry"],
                        label="Preferred Wood Finish", value="Dark Oak")
                output_image = gr.Image(label="Transformed Room")
                generate_btn = gr.Button("Transform Room", elem_classes=["generate-btn"])
                generate_btn.click(
                    fn=safe_generate,
                    inputs=[input_image, room_type, change_options, aesthetic, color_picker, wood_type],
                    outputs=output_image)

     with gr.Tab("Style Discovery Quiz"):
        gr.Markdown(""" # Welcome to the style quiz
       Complete each question to find your desired """)
        with gr.Column(elem_classes=["input-group"]):
            style_inputs = []
            for i, q in enumerate(questions):
                gr.Markdown(f"*", elem_classes=["quiz-question"])
                style_inputs.append(
                    gr.Radio(
                        choices=q["options"],
                        label=q['question'],
                        value=None,
                        elem_classes=["quiz-option"]
                    )
                )

            style_result = gr.Textbox(
                label="Your Preferred Style",
                interactive=False,
                elem_classes=["result-box"]
            )
            style_summary = gr.Markdown(
                value="",
                label="Style Details",
                elem_classes=["result-box"]
            )

            quiz_btn = gr.Button(
                "Discover My Style",
                elem_classes=["generate-btn"]
            )

            quiz_btn.click(
                fn=calculate_style_friendly,
                inputs=style_inputs,
                outputs=[style_result, style_summary]
            )

     with gr.Tab("About"):
        # About Page Header
        with gr.Row(elem_classes=["header"]):
            gr.Markdown("# About Room Transformation Studio")

        # About Page Content
        with gr.Row():
            gr.Markdown("""
# What is a Room Transformation Studio?

Room Transformation Studio is your dedicated personal assistant for reimagining and redesigning your living spaces, harnessing the innovative power of artificial intelligence. Our platform is designed to cater to a variety of design needs and styles, making it perfect for anyone looking to refresh their home.

With **Room Transformation Studio**, you can effortlessly explore a wide array of design concepts tailored specifically to your preferences. Whether you want to create a serene and inviting atmosphere in a cozy bedroom, infuse energy and warmth into your living area, or achieve a sleek and contemporary look in your kitchen, our application provides you with bespoke ideas that reflect your lifestyle.

Using advanced algorithms, the app analyzes your inputs—such as preferred colors, styles, and spatial dimensions—to generate unique design proposals that not only enhance the aesthetic appeal of your space but also improve its functionality. You can visualize your options, making it easier than ever to make informed decisions about your interiors.

Transform your home into a stylish sanctuary with **Room Transformation Studio**, where creativity meets practicality, and your vision is just a click away!


# Who created it?

Hi! I am Helga, a CS student at West Uviverity of Timisoara. I created this project to experiment with new technologies and learn more about programming.
This project was created to assist people who want to redecorate, making it more affordable for everyone to have a nicely decorated room.

# What's next?

Our vision for the future is to develop this project into a fully independent assistant that can redecorate any room. We want to add 3D rendering and a search engine for furniture.
""")

demo.launch()