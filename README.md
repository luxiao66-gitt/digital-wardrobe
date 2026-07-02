# Digital Wardrobe

A full-stack wardrobe app that turns OOTD photos into structured wardrobe items.

## Milestone 1

Upload an outfit photo, detect visible clothing items, remove item backgrounds, generate AI tags, save the results, and display the wardrobe items in a web UI.

## Planned Stack

- Frontend: Next.js
- Backend: FastAPI
- Detection: YOLOv8
- Background removal: rembg
- Tagging: GPT-4o mini
- Database: SQLite for MVP, PostgreSQL later
- File storage: local filesystem for MVP, object storage later

## Project Layout

```text
digital-wardrobe/
  backend/              FastAPI backend and Python processing pipeline
  frontend/             Next.js web app
  data/                 Local development uploads, processed images, and DB files
  docs/                 Planning notes, API contracts, and milestone docs
```

