import uuid
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        with logger.contextualize(request_id=request_id):
            logger.info(f"Request id: {request_id}")
            response: Response = await call_next(request)
            logger.info(f"<- response: {response.status_code}")
            response.headers["X-Request-Id"] = request_id
            return response