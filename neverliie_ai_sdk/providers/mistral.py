from .openai_compatible import OpenAICompatible


class Mistral(OpenAICompatible):
    """Mistral AI API provider.
    
    Uses the OpenAI-compatible Mistral API. Supports text generation,
    function calling, and streaming.
    """
    default_base_url = "https://api.mistral.ai/v1"
    default_model = "mistral-small-latest"

    def __init__(self, api_key: str, base_url: str = None):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=self.default_model,
        )
