# Standard Libraries

# 3rd Party Libraries
import structlog
import uvicorn
from llm.services.llm import LLMApi
from llm.settings import LLMSettings
from fastapi import FastAPI
from prometheus_async.aio.web import start_http_server

logger = structlog.get_logger(module=__name__)


class App:
    cfg: LLMSettings

    def __init__(self) -> None:
        self.cfg = LLMSettings()  # type: ignore

    async def start(self):
        await logger.ainfo("Application started")

        await logger.ainfo("Starting prometheus")
        await start_http_server(port=self.cfg.prometheus_port)

        await logger.ainfo("Starting service")

        await self.start_llm()

        await logger.ainfo("Application shutting down")

    def custom_exception_handler(self, loop, context):
        # first, handle with default handler
        loop.default_exception_handler(context)
        e = context.get("exception")
        logger.exception("asyncio exception", e)
        loop.stop()

    async def start_llm(self) -> None:
        app = FastAPI()
        routes = LLMApi(self.cfg)
        app.include_router(routes.router)
        config = uvicorn.Config(app, host="0.0.0.0", port=9900, log_level=self.cfg.log_level.lower(), loop="asyncio")
        server = uvicorn.Server(config)
        await server.serve()
