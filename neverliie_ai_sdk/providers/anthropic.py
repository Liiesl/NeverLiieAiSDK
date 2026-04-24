from .anthropic_compatible import AnthropicCompatible


class Anthropic(AnthropicCompatible):
    """Anthropic API provider.
    
    Uses the Anthropic Messages API. Supports text generation,
    function calling, and streaming.
    """
    default_base_url = "https://api.anthropic.com/v1"
    default_model = "claude-3-haiku-20240307"
    default_owned_by = "anthropic"

    def __init__(self, api_key: str, base_url: str = None):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=self.default_model,
            extra_headers={
                "anthropic-version": "2023-06-01",
            },
        )
