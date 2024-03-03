from contextvars import ContextVar

parent_ctx = ContextVar("parent_ctx", default=None)

class ParentContextManager:
    def __init__(self, message_id: str):
        parent_ctx.set(message_id)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_tb):
        parent_ctx.set(None)


def parent(id: str) -> ParentContextManager:
    return ParentContextManager(id)
