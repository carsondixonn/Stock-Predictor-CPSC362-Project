from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select
from pwdlib import PasswordHash
from database import User, engine


router = APIRouter()
password_hasher = PasswordHash.recommended()


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
def register_user(data: RegisterRequest):

    with Session(engine) as session:

        # Check if username already exists
        user = session.exec(
            select(User).where(User.username == data.username)
        ).first()

        if user:
            raise HTTPException(
                status_code=400,
                detail="Username already exists"
            )

        # Check if email already exists
        user = session.exec(
            select(User).where(User.email == data.email)
        ).first()

        if user:
            raise HTTPException(
                status_code=400,
                detail="Email already exists"
            )

        # Hash password (might be optional)
        hashed_password = password_hasher.hash(data.password)

        # Create and save user
        new_user = User(
            username=data.username,
            email=data.email,
            password_hash=hashed_password
        )

        session.add(new_user)
        session.commit()
        session.refresh(new_user)

        return {
            "message": "User created successfully",
            "id": new_user.id,
            "username": new_user.username,
            "email": new_user.email
        }


@router.post("/login")
def login_user(data: LoginRequest):

    with Session(engine) as session:

        user = session.exec(
            select(User).where(User.username == data.username)
        ).first()

        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )

        if not password_hasher.verify(data.password, user.password_hash):
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )

        return {
            "message": "Login successful",
            "id": user.id,
            "username": user.username,
            "email": user.email
        }
