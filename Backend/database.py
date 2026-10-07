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


# Database session
def get_session():
    with Session(engine) as session:
        yield session
