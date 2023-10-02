import threading


class EventQueue:
    def __init__(self):
        self.lock = threading.Lock()
        self.events = []

    def append(self, event):
        with self.lock:
            self.events.append(event)

    def get_batch(self):
        with self.lock:
            events = self.events
            self.events = []
            return events
