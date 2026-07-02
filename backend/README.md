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

