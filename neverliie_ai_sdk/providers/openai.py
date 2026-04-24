from .openai_compatible import OpenAICompatible


class OpenAI(OpenAICompatible):
    """OpenAI API provider.
    
    Uses the OpenAI chat completions API. Supports text generation,
    function calling, and streaming.
    """
    default_base_url = "https://api.openai.com/v1"
    default_model = "gpt-4o-mini"

    def __init__(self, api_key: str, base_url: str = None):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=self.default_model,
        )
