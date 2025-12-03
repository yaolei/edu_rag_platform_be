from os import write

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.requests import Request
import httpx
from gateway.app.core.config import settings
app = FastAPI(title='gateway', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
timeout = httpx.Timeout(
    connect=20,
    read=150,
    timeout=60,
    pool=10.0,
    write=10.0
)

@app.get('/')
def hello_world():
    return {'message': 'Hello World'}

@app.api_route("/{svc}{path:path}", methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])
async def proxy(svc:str, path:str, request: Request):
    if svc not in settings.SERVICE:
        return Response("Evan Service not found !", status_code=404)
    url = f"http://{settings.SERVICE[svc]}/{path}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method=request.method,
                url=url,
                headers=request.headers.raw,
                content= await request.body,
                params=request.query_params,
                timeout=timeout,
            )
            return Response(content=response.content, status_code=response.status_code, headers=response.headers)
    except httpx.ConnectTimeout:
        return Response("Connection timed out !", status_code=404)
    except httpx.ConnectError:
        return Response("Connection error !", status_code=404)
    except httpx.ReadTimeout:
        return Response("Read Time Connection timed out !", status_code=404)
    except Exception as e:
        print(f"gate way error : {str(e)}")
        return {
            "message": "Gateway error",
            "state": 500
        }