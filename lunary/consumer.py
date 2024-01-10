import time
import atexit
import requests
import os
import logging
from threading import Thread

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://app.lunary.ai"


class Consumer(Thread):
    def __init__(self, event_queue):
        self.running = True
        self.event_queue = event_queue
        self.api_url = (
            os.environ.get("LUNARY_API_URL")
            or os.environ.get("LLMONITOR_API_URL")
            or DEFAULT_API_URL
        )
        self.verbose = (
            os.environ.get("LUNARY_VERBOSE")
            or os.environ.get("LLMONITOR_VERBOSE")
            or False
        )

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
            if self.verbose:
                logging.info(f"[Lunary] Sending {len(batch)} events.")

            try:
                if self.verbose:
                    logging.info("[Lunary] Sending events to ", self.api_url)

                response = requests.post(
                    self.api_url + "/api/report",
                    json={"events": batch},
                    headers={"Content-Type": "application/json"},
                    timeout=5,
                )

                if self.verbose:
                    logging.info("[Lunary] Events sent.", response.status_code)

                if response.status_code != 200:
                    logging.error("[Lunary] Error sending events")
            except Exception as e:
                if self.verbose:
                    logging.error("[Lunary] Error sending events", e)

                self.event_queue.append(batch)

    def stop(self):
        self.running = False
        self.join()
