from typing import Any, Optional

from langchain_aws import ChatBedrockConverse

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "max_tokens", "callbacks",
    "region_name", "credentials_profile_name",
)


class NormalizedChatBedrockConverse(ChatBedrockConverse):
    """ChatBedrockConverse with normalized content output."""

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class BedrockClient(BaseLLMClient):
    """Client for AWS Bedrock (uses default AWS profile credentials)."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatBedrockConverse instance."""
        self.warn_if_unknown_model()
        llm_kwargs: dict[str, Any] = {"model": self.model}

        region = self.kwargs.get("region_name")
        if region:
            llm_kwargs["region_name"] = region
        elif self.base_url:
            llm_kwargs["region_name"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatBedrockConverse(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Bedrock."""
        return validate_model("bedrock", self.model)
