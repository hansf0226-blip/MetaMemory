"""MetaMemory 错误类型定义"""
class BaseError(Exception):
    def __init__(self, message: str, code: int = 500, details: dict = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)

class ValidationError(BaseError): pass
class DatabaseError(BaseError): pass
class NotFoundError(BaseError): pass
class ConflictError(BaseError): pass
class AuthenticationError(BaseError): pass
class AuthorizationError(BaseError): pass
class RateLimitError(BaseError): pass
class ServiceUnavailableError(BaseError): pass
class TimeoutError(BaseError): pass
