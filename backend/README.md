# Backend

FastAPI backend for image upload, wardrobe item processing, and database access.

## App Modules

- `app/api`: HTTP routes.
- `app/core`: configuration and shared app setup.
- `app/db`: database session, migrations, and persistence helpers.
- `app/models`: database models.
- `app/schemas`: API request and response schemas.
- `app/services`: business logic.
- `app/services/pipeline`: YOLOv8, rembg, and GPT tagging steps.
- `app/storage`: local file storage helpers.

## Wardrobe Extraction Function

The wardrobe extraction function lives in:

```text
app/services/pipeline/extract_outfit.py
```

It accepts a common image format such as JPG, JPEG, PNG, WEBP, BMP, or TIFF. It runs rembg with the `u2net_cloth_seg` model, isolates each clothing mask layer, saves transparent auto-cropped PNGs, and writes a `result.json` file with item metadata.

Run it from the `backend` folder:

```powershell
python -m scripts.extract_outfit C:\path\to\ootd.jpg
```

You can lower or raise the minimum mask size:

```powershell
python -m scripts.extract_outfit C:\path\to\ootd.jpg --min-pixels 1000
```

The output is written under:

```text
../data/processed/<photo_id>/result.json
../data/processed/<photo_id>/items/item_001.png
```

The rembg clothing mask values are:

```text
0 = background
1 = top
2 = bottom
3 = full-body / outerwear
```
