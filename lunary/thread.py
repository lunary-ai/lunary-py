import uuid
from typing import Optional, List, TypedDict


class Message(TypedDict, total=False):
    id: str
    role: str
    content: Optional[str]
    is_retry: Optional[bool]
    tags: Optional[List[str]]


class Thread:
    def __init__(
        self, 
        track_event, 
        user_id: Optional[str] = None, 
        user_props: Optional[dict] = None,
        id: Optional[str] = None, 
        tags: Optional[List[str]] = None
    ):
        self.id = id or str(uuid.uuid4())
        self.user_id = user_id
        self.user_props = user_props
        self.tags = tags
        self.track_event = track_event

    def track_message(self, message: Message, user_id=None, user_props=None, feedback=None) -> str:
        run_id = message.get("id", str(uuid.uuid4()))

        self.track_event("thread", "chat",
                         run_id=run_id,
                         user_id=self.user_id,
                         user_props=self.user_props,
                         parent_run_id=self.id,
                         thread_tags=self.tags,
                         feedback=feedback,
                         message=message)
        return run_id
