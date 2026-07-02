from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Digital Wardrobe API"
    environment: str = "development"
    database_url: str = "sqlite:///../../data/database/wardrobe.db"
    upload_dir: str = "../../data/uploads"
    processed_dir: str = "../../data/processed"
    openai_api_key: str = ""


settings = Settings()

