from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from service_rag.app.api.router import router
from service_rag.app.config.lifespan import lifespan
app = FastAPI(title='service-rag', version='1.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=["*"])
app.include_router(router, prefix='/api')





