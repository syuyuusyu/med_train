from dependency_injector import containers, providers
import yaml
from .services import RetrieverService, doc_info
import os


class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    # 在容器初始化时加载 YAML 文件
    with open(config_path, "r") as f:
        config.from_dict(yaml.safe_load(f))

    doc_info_passages = providers.Singleton(doc_info)

    retriever_service = providers.Singleton(RetrieverService,passages=doc_info_passages)

container = Container()