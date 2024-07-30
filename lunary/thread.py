import uuid
from typing import List, TypedDict


class Message(TypedDict, total=False):
    id: str
    role: str
    content: str | None
    is_retry: bool | None
    tags: List[str] | None


class Thread:
    def __init__(
        self,
        track_event,
        user_id: str | None = None,
        user_props: dict | None = None,
        id: str | None = None,
        tags: List[str] | None = None,
        app_id: str | None = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.user_id = user_id
        self.user_props = user_props
        self.tags = tags
        self.track_event = track_event
        self.app_id = app_id

    def track_message(
        self, message: Message, user_id=None, user_props=None, feedback=None
    ) -> str:
        run_id = message.get("id", str(uuid.uuid4()))

        self.track_event(
            "thread",
            "chat",
            run_id=run_id,
            user_id=user_id or self.user_id,
            user_props=user_props or self.user_props,
            parent_run_id=self.id,
            thread_tags=self.tags,
            feedback=feedback,
            message=message,
            app_id=self.app_id,
        )
        return run_id
