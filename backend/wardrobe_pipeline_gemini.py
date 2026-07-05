import os
import sys
import json
import base64
from io import BytesIO
from PIL import Image
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# --- STEP 1: DEFINE THE JSON STRUCTURE SCHEMA ---
class WardrobeItem(BaseModel):
    category: str = Field(description="The generic category of the item (e.g., dress, top, skirt, pants, shoes, handbag).")
    description: str = Field(description="A highly accurate, granular description of the item's pattern, style, cut, and texture.")
    color: str = Field(description="The primary color or colors of the item.")

class WardrobeManifest(BaseModel):
    items: list[WardrobeItem] = Field(description="The collection of distinct clothing pieces and accessories worn by the model.")


def analyze_wardrobe_manifest(image_path, json_output_path):
    """
    Step 1: Uses Gemini multimodal vision to return a structured JSON inventory of garments.
    """
    print(f"\n[STEP 1] Analyzing image: {image_path}...")
    client = genai.Client()
    img = Image.open(image_path)

    prompt = """
    Analyze the person in this photograph. Dissect their outfit completely and return a list
    of every distinct garment, footwear item, and major accessory (like a handbag) they are wearing.
    Be precise about patterns (e.g., plaid, gingham) and structural properties.
    """

    try:
        # Requesting a structured JSON output mapping strictly to our Pydantic model
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=WardrobeManifest,
            ),
        )
        
        # Save the structured data
        manifest_data = json.loads(response.text)
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=4, ensure_ascii=False)
            
        print(f"✅ Success! Wardrobe manifest saved as JSON to: {json_output_path}")
        return manifest_data

    except Exception as e:
        print(f"❌ Error in Step 1: {str(e)}")
        return None


def generate_item_overlays(image_path, json_input_path, output_dir):
    """
    Step 2: Reads the JSON inventory and invokes Nano Banana to isolate each item sequentially.
    """
    print(f"\n[STEP 2] Processing manifest from: {json_input_path}...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    client = genai.Client()
    img = Image.open(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    with open(json_input_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for idx, item in enumerate(manifest.get("items", []), start=1):
        category = item["category"].lower().replace(" ", "_")
        description = item["description"]
        color = item["color"]
        
        print(f"\n -> Isolating Item #{idx}: {color} {category}...")

        # Crafting a localized generative prompt targeting only this specific item description
        prompt = f"""
        You are a high-end digital fashion catalog archivist. Examine the model in the provided photo.
        
        TASK:
        Isolate and extract ONLY the following item:
        - Category: {category}
        - Color: {color}
        - Description: {description}
        
        GENERATION REQUIREMENTS:
        - Generate a completely new image featuring ONLY this specific target item.
        - Style the output as a luxury commercial fashion flat-lay or a ghost-mannequin retail shot.
        -Symmetrically center the item on a completely solid, seamless, crisp off-white backdrop.
        - Completely erase and strip away the human model, their face, limbs, skin, background walls, or any other clothing pieces.
        - Retain flawless design fidelity to the original item's textile structures, drape, wrinkles, and patterns.
        - Use clean, balanced studio lighting casting a subtle, natural depth shadow directly underneath the item.
        - No text, hangers, watermarks, or ambient artifacts.
        """

        output_filename = f"{base_name}_{idx}_{category}.png"
        output_path = os.path.join(output_dir, output_filename)

        try:
            # Invoking Gemini Nano Banana Engine
            response = client.models.generate_content(
                model="gemini-3.1-flash-image",
                contents=[prompt, img],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio="1:1",
                        image_size="1K"
                    )
                )
            )

            for part in response.parts:
                if part.inline_data:
                    generated_bytes = part.inline_data.data
                    output_image = Image.open(BytesIO(generated_bytes))
                    output_image.save(output_path, "PNG")
                    print(f"   🎉 Showroom-ready asset saved: {output_path}")
                    break
            else:
                print(f"   ⚠️ Model executed but failed to deliver image parts for {category}.")

        except Exception as e:
            print(f"   ❌ Generation failed for {category}: {str(e)}")


def run_pipeline(input_image_path):
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    json_path = f"{base_name}_manifest.json"
    output_directory = "output_showroom"

    # Step 1: Core Parsing & Cataloging
    manifest = analyze_wardrobe_manifest(input_image_path, json_path)
    
    # Step 2: Generative Isolation
    if manifest:
        generate_item_overlays(input_image_path, json_path, output_directory)
        print("\n🚀 Entire automated workflow completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python wardrobe_pipeline.py <input_image.jpg>")
        sys.exit(1)
        
    run_pipeline(sys.argv[1])