from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr


# Common fields shared across requests/responses
class UserBase(BaseModel):
    email: EmailStr


# Schema for CREATING a user (Input payload)
class UserCreate(UserBase):
    password: str


# Schema for RETURNING a user (Output payload)
class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    # Tells Pydantic to read ORM objects directly from SQLAlchemy
    model_config = ConfigDict(from_attributes=True)