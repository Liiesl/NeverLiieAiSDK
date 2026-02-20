class APIError(Exception):
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class AuthenticationError(APIError):
    pass


class RateLimitError(APIError):
    pass


class NotFoundError(APIError):
    pass


class InvalidRequestError(APIError):
    pass
