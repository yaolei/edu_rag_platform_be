from fastapi import FastAPI, APIRouter
app = FastAPI(title='service-rag', version='1.0')

router = APIRouter()

@router.get('/')
def index():
    return {'message': 'Hello World'}