from collections import defaultdict, deque
from typing import Deque, Dict, List
from .schemas import ChatMessage

class ChatSessions:
    def __init__(self, max_turns: int = 30):
        self.max_turns = max_turns
        self._by_id: Dict[str, Deque[ChatMessage]] = defaultdict(lambda: deque(maxlen=max_turns*2))

    def history(self, session_id: str) -> List[ChatMessage]:
        return list(self._by_id[session_id])

    def append(self, session_id: str, msg: ChatMessage) -> None:
        self._by_id[session_id].append(msg)

    def reset(self, session_id: str) -> None:
        self._by_id.pop(session_id, None)
