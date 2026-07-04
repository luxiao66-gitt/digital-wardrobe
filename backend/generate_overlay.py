import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageFilter
from rembg import remove, new_session
from ultralytics import YOLO

def polish_garment_edges(pil_image, erode_pixels=1, blur_radius=2):
    """
    Takes a rough transparent PIL Image and refines its edges to look studio-ready.
    """
    # Convert to RGBA numpy array to work on individual color channels
    img_arr = np.array(pil_image)
    r, g, b, alpha = cv2.split(img_arr)
    
    # 1. Clean up edge contamination (Erosion)
    # Shaves off a tiny 1-pixel boundary to remove background wall/skin bleeding
    if erode_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha = cv2.erode(alpha, kernel, iterations=erode_pixels)
        
    # Recombine channels back into a PIL Image
    cleaned_arr = cv2.merge([r, g, b, alpha])
    polished_img = Image.fromarray(cleaned_arr)
    
    # 2. Smooth the transition (Alpha Feathering)
    # Blurs only the transparency mask layer so the edge melts into the background
    alpha_channel = polished_img.split()[3]
    smoothed_alpha = alpha_channel.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # Put the polished, soft-edge alpha mask back on the original sharp clothing pixels
    polished_img.putalpha(smoothed_alpha)
    
    return polished_img

def extract_wardrobe_and_shoes(input_path):
    if not os.path.exists(input_path):
        print(f"Error: Could not find the file '{input_path}'.")
        return

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # --- STEP 1: INITIALIZE MODELS ---
    print("Step 1: Initializing pre-trained segmentation engines...")
    cloth_session = new_session("u2net_cloth_seg")

    print(f"Step 2: Processing '{input_path}'...")
    img = Image.open(input_path).convert("RGBA")
    orig_w, orig_h = img.size
    
    clothing_strip = remove(img, session=cloth_session)

    print("Step 3: Analyzing apparel structure...")
    strip_w, strip_h = clothing_strip.size
    section_h = strip_h // 3

    upper_crop = clothing_strip.crop((0, 0, strip_w, section_h))
    lower_crop = clothing_strip.crop((0, section_h, strip_w, section_h * 2))
    full_crop = clothing_strip.crop((0, section_h * 2, strip_w, strip_h))

    upper_bbox = upper_crop.getbbox()
    lower_bbox = lower_crop.getbbox()
    full_bbox = full_crop.getbbox()

    has_upper = upper_bbox is not None and (upper_bbox[2]-upper_bbox[0] > 30)
    has_lower = lower_bbox is not None and (lower_bbox[2]-lower_bbox[0] > 30)
    has_full = full_bbox is not None and (full_bbox[2]-full_bbox[0] > 30)

    is_dress = False

    if has_upper and has_lower:
        top_of_skirt = lower_bbox[1]
        bottom_of_shirt = upper_bbox[3]
        vertical_gap = top_of_skirt - bottom_of_shirt
        if vertical_gap <= 15: 
            is_dress = True

    if has_full and not (has_upper and has_lower and not is_dress):
        is_dress = True

    # --- SAVE GARMENTS WITH POLISHED EDGES ---
    if is_dress:
        print(" -> One-Piece Dress detected. Healing layers...")
        merged_dress = Image.new("RGBA", (strip_w, section_h))
        if has_upper: merged_dress.paste(upper_crop, (0, 0), upper_crop)
        if has_lower: merged_dress.paste(lower_crop, (0, 0), lower_crop)
        if has_full: merged_dress.paste(full_crop, (0, 0), full_crop)
        
        tight_bbox = merged_dress.getbbox()
        if tight_bbox:
            final_dress = merged_dress.crop(tight_bbox)
            # Apply the edge polisher before saving
            polished_dress = polish_garment_edges(final_dress, erode_pixels=1, blur_radius=2)
            polished_dress.save(f"{base_name}_dress.png", "PNG")
            print(f"   -> Saved (Polished): {base_name}_dress.png")
    else:
        print(" -> Two-Piece Outfit detected.")
        if has_upper:
            tight_top = upper_crop.crop(upper_bbox)
            polished_top = polish_garment_edges(tight_top, erode_pixels=1, blur_radius=2)
            polished_top.save(f"{base_name}_top.png", "PNG")
            print(f"   -> Saved (Polished): {base_name}_top.png")
        if has_lower:
            tight_bottom = lower_crop.crop(lower_bbox)
            polished_bottom = polish_garment_edges(tight_bottom, erode_pixels=1, blur_radius=2)
            polished_bottom.save(f"{base_name}_skirt_pants.png", "PNG")
            print(f"   -> Saved (Polished): {base_name}_skirt_pants.png")

    # --- STEP 4: ISOLATE & POLISH THE SHOES ---
    print("Step 4: Pinpointing footwear...")
    yolo_model = YOLO("yolov8n-seg.pt") 
    results = yolo_model.predict(source=input_path, classes=[18], conf=0.05, verbose=False)
    result = results[0]
    
    shoes_extracted = False
    
    if result.masks is not None:
        master_shoe_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for mask in result.masks.data:
            mask_np = mask.cpu().numpy()
            mask_resized = Image.fromarray((mask_np * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.NEAREST)
            master_shoe_mask = np.maximum(master_shoe_mask, np.array(mask_resized))
            
        original_rgba = np.array(img)
        original_rgba[:, :, 3] = master_shoe_mask
        
        shoes_img = Image.fromarray(original_rgba)
        shoes_bbox = shoes_img.getbbox()
        
        if shoes_bbox:
            tight_shoes = shoes_img.crop(shoes_bbox)
            # Polish shoe edges to look smooth
            polished_shoes = polish_garment_edges(tight_shoes, erode_pixels=1, blur_radius=2)
            polished_shoes.save(f"{base_name}_shoes.png", "PNG")
            print(f"   -> Saved (Polished): {base_name}_shoes.png (via Object Detection)")
            shoes_extracted = True

    # Fallback shoe tracker if YOLO misses the angle completely
    if not shoes_extracted:
        print(" -> YOLO missed the shoes. Activating fallback human-segmentation matrix...")
        human_session = new_session("u2net_human_seg")
        full_human = remove(img, session=human_session).resize((orig_w, orig_h))
        
        human_arr = np.array(full_human)
        human_alpha = human_arr[:, :, 3]
        
        shoe_zone_start = int(orig_h * 0.82)
        fallback_mask = np.zeros_like(human_alpha)
        fallback_mask[shoe_zone_start:, :] = human_alpha[shoe_zone_start:, :]
        
        original_rgba = np.array(img)
        original_rgba[:, :, 3] = fallback_mask
        
        shoes_img = Image.fromarray(original_rgba)
        shoes_bbox = shoes_img.getbbox()
        
        if shoes_bbox:
            tight_shoes = shoes_img.crop(shoes_bbox)
            # Apply polish to fallback shoes too
            polished_shoes = polish_garment_edges(tight_shoes, erode_pixels=1, blur_radius=2)
            polished_shoes.save(f"{base_name}_shoes.png", "PNG")
            print(f"   -> Saved (Polished): {base_name}_shoes.png (via Fallback Anchor)")

    print("\nSmart execution complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_overlay.py <image_name>")
        sys.exit(1)
    extract_wardrobe_and_shoes(sys.argv[1])