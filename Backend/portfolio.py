from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from database import User, Portfolio, engine


router = APIRouter()


# info needed when adding a stock
class PortfolioRequest(BaseModel):
    user_id: int
    ticker: str
    shares: float = 0


# add a stock to a user's portfolio
@router.post("/portfolio")
def add_stock(data: PortfolioRequest):

    with Session(engine) as session:

        # make sure the user exists
        user = session.get(User, data.user_id)

        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )

        # check if stock is already in portfolio
        existing_stock = session.exec(
            select(Portfolio).where(
                Portfolio.user_id == data.user_id,
                Portfolio.ticker == data.ticker.upper()
            )
        ).first()

        if existing_stock:
            raise HTTPException(
                status_code=400,
                detail="Stock already in portfolio"
            )

        # create portfolio entry
        stock = Portfolio(
            user_id=data.user_id,
            ticker=data.ticker.upper(),
            shares=data.shares
        )

        session.add(stock)
        session.commit()
        session.refresh(stock)

        return {
            "message": "Stock added to portfolio",
            "id": stock.id,
            "user_id": stock.user_id,
            "ticker": stock.ticker,
            "shares": stock.shares
        }


# show all stocks for one user
@router.get("/portfolio/{user_id}")
def get_portfolio(user_id: int):

    with Session(engine) as session:

        stocks = session.exec(
            select(Portfolio).where(
                Portfolio.user_id == user_id
            )
        ).all()

        return stocks


# remove a stock from a user's portfolio
@router.delete("/portfolio/{user_id}/{ticker}")
def remove_stock(user_id: int, ticker: str):

    with Session(engine) as session:

        stock = session.exec(
            select(Portfolio).where(
                Portfolio.user_id == user_id,
                Portfolio.ticker == ticker.upper()
            )
        ).first()

        if not stock:
            raise HTTPException(
                status_code=404,
                detail="Stock not found in portfolio"
            )

        session.delete(stock)
        session.commit()

        return {
            "message": "Stock removed from portfolio"
        }
