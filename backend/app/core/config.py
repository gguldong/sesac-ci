#Mysql 데이터베이스
import os
import secrets
from pydantic_settings import BaseSettings

# Endpoint = "testdb.cfmq6wqw0199.ap-northeast-2.rds.amazonaws.com"
# Port = 3306
# Username = "admin"
# Password = "Saltlux12345!"
# Database = "multimodal_final_project"
Endpoint = os.environ.get("Endpoint", "")
Port = int(os.environ.get("Port", "3306"))
Username = os.environ.get("Username", "admin")
Password = os.environ.get("Password", "Saltlux12345!")
Database = os.environ.get("Database", "multimodal_final_project")


class Settings(BaseSettings):
    SECRET_KEY: str = secrets.token_hex(32)
    SQLALCHEMY_DATABASE_URL: str = f"mysql+pymysql://{Username}:{Password}@{Endpoint}:{Port}/{Database}"
    ALLOWED_ORIGINS: list = [f"{Endpoint}:{Port}"]

settings = Settings()
