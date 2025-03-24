from pydantic import BaseModel, EmailStr, Field
from typing import List
from datetime import date

class UserDataRequest(BaseModel):
    user_id: str
    password: str
    username: str
    email: str
    phone: str
    area: str
    district: str
    birthDate: str
    gender: str
    incomeRange: str
    personalCharacteristics: List[str] = Field(default_factory=list)
    householdCharacteristics: List[str] = Field(default_factory=list)

class LoginRequest(BaseModel):
    user_id: str
    password: str
