import threading
import requests


class EventQueue:
    def __init__(self, api_url, interval=0.5):
        self.queue = []
        self.interval = interval
        self.api_url = api_url
        self.lock = threading.Lock()
        self.start_timer()

    def add_event(self, event):
        with self.lock:
            self.queue.append(event)

    def send_events(self):
        with self.lock:
            batch = self.queue
            self.queue = []

        if batch:
            try:
                response = requests.post(
                    self.api_url,
                    json={"events": batch},
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code != 200:
                    print("Error sending events")
            except Exception as e:
                self.add_event(batch)

        self.start_timer()

    def start_timer(self):
        threading.Timer(self.interval, self.send_events).start()
