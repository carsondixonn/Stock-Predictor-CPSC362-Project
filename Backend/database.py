from sqlalchemy import text
from sqlmodel import SQLModel, Field, Session, create_engine


# User table
class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str
    email: str
    password_hash: str


# Portfolio table
class Portfolio(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    # Connects this stock to a specific user
    user_id: int = Field(foreign_key="user.id")

    ticker: str
    shares: float = 0

    # Average price paid per share (None until the user records one)
    cost: float | None = None


# Database connection
database_file = "stocks.db"

database_url = f"sqlite:///{database_file}"

engine = create_engine(
    database_url,
    echo=True,
    connect_args={"check_same_thread": False}
)


# Create tables
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

    # create_all never alters an existing table, so a stocks.db made before the
    # "cost" column existed gets it added here (existing rows keep cost = NULL).
    with engine.begin() as conn:
        columns = [row[1] for row in conn.execute(text("PRAGMA table_info(portfolio)"))]
        if "cost" not in columns:
            conn.execute(text("ALTER TABLE portfolio ADD COLUMN cost FLOAT"))


# Database session
def get_session():
    with Session(engine) as session:
        yield session
