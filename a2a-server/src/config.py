import logging
from pydantic_settings import BaseSettings
from pydantic import computed_field

class Settings(BaseSettings):
    # Google / Gemini
    google_api_key: str = ""

    # Tavily (passed through to MCP server)
    tavily_api_key: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 10001
    agent_url: str = "http://localhost:8000/agent/"

    # Postgres — individual vars matching root .env & docker-compose
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "deep_research"
    postgres_host: str = "localhost"
    postgres_port: str = "5432"

    # Log level
    log_level: str = "INFO"

    @computed_field
    @property
    def db_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
