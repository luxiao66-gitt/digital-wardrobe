from fastapi import APIRouter, File, UploadFile

router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.post("/ootd")
async def upload_ootd_photo(file: UploadFile = File(...)) -> dict[str, str]:
    return {
        "filename": file.filename or "upload",
        "status": "received",
    }

