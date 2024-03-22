# Standard Libraries

# 3rd Party Libraries
from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    llm_connection_string: str = ""
    llm_model_max_tokens: int = 8192
    prometheus_port: int = 9801
    log_level: str = "INFO"