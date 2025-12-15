from service_rag.app.modules.dataset_module import Base
from conect_databse.database import engine, SessionLocal
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    Base.metadata.create_all(bind=engine)
    app.state.engine = engine
    app.state.Session = SessionLocal
    yield
    engine.dispose()