from fastapi import FastAPI
from .dependencies import container
from .routers import register_routers

def create_app() -> FastAPI:
    app = FastAPI()
    # 注入依赖
    container.wire(modules=[".routers.triage"])
    # 注册路由
    register_routers(app)
    return app