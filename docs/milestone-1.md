# Milestone 1: Outfit Photo Processing MVP

## Goal

Upload an OOTD photo and automatically extract wardrobe items with editable metadata.

## Done When

- User can upload a photo from the frontend.
- FastAPI receives and stores the photo.
- YOLOv8 detects visible clothing items.
- Each detected item is cropped.
- rembg creates a clean item image.
- GPT-4o mini generates tags for each item.
- Item data is saved in the database.
- Frontend displays detected items in a wardrobe grid.

## Not Included Yet

- User accounts
- Outfit recommendations
- Calendar or wear history
- Duplicate item detection
- Advanced search
- Shopping links

