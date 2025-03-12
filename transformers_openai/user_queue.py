import asyncio
from typing import Dict, List, Optional

class AsyncUserQueue:
    def __init__(self, max_size: int):
        """
        Initialize the queue with a maximum size
        
        :param max_size: Maximum number of slots in the queue
        """
        self.max_size = max_size
        self.users: List[Optional[str]] = [None] * max_size
        self.lock = asyncio.Lock()

    async def enter(self, user_id: str) -> int:
        async with self.lock:
            for i, slot in enumerate(self.users):
                if slot is None:
                    self.users[i] = user_id
                    return i
            raise ValueError("Queue is full")

    async def leave(self, user_id: str) -> None:
        async with self.lock:
            try:
                index = self.users.index(user_id)
            except ValueError:
                return
            self.users[index] = None

    async def get_current_users(self) -> List[Optional[str]]:
        async with self.lock:
            return self.users.copy()

    async def is_full(self) -> bool:
        async with self.lock:
            return all(user is not None for user in self.users)

    async def available_slots(self) -> int:
        async with self.lock:
            return self.users.count(None)

class UserQueue:
    def __init__(self, max_size: int):
        """
        Initialize the queue with a maximum size
        
        :param max_size: Maximum number of slots in the queue
        """
        self.max_size = max_size
        self.users: List[Optional[str]] = [None] * max_size

    def enter(self, user_id: str) -> int:
        for i, slot in enumerate(self.users):
            if slot is None:
                self.users[i] = user_id
                return i
        raise ValueError("Queue is full")

    def leave(self, user_id: str) -> None:
        try:
            index = self.users.index(user_id)
        except ValueError:
            return
        self.users[index] = None

    def get_current_users(self) -> List[Optional[str]]:
        return self.users.copy()

    def is_full(self) -> bool:
        return all(user is not None for user in self.users)

    def available_slots(self) -> int:
        return self.users.count(None)