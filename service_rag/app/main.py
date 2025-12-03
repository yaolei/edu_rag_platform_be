from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from service_rag.app.api.router import router
from service_rag.app.modules.dataset_module import Base
from conect_databse.database import engine

app = FastAPI(title='service-rag', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=["*"])
try:
    Base.metadata.create_all(bind=engine)
    print("Database created")
except Exception as e:
    raise e


app.include_router(router, prefix='/api')





