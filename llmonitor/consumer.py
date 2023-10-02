import time, atexit, requests, os
from threading import Thread

DEFAULT_API_URL = "https://app.llmonitor.com"


class Consumer(Thread):
    def __init__(self, event_queue, api_url=None):
        self.running = True
        self.event_queue = event_queue
        self.api_url = api_url or DEFAULT_API_URL

        Thread.__init__(self, daemon=True)
        atexit.register(self.stop)

    def run(self):
        while self.running:
            self.send_batch()
            time.sleep(0.5)

        self.send_batch()

    def send_batch(self):
        batch = self.event_queue.get_batch()
        if len(batch) > 0:
            try:
                response = requests.post(
                    self.api_url + "/api/report",
                    json={"events": batch},
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code != 200:
                    print("Error sending events")
            except Exception as e:
                self.event_queue.append(batch)

    def stop(self):
        self.running = False
        self.join()
