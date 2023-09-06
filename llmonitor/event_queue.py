import threading
import requests

from threading import Timer
from inspect import signature
import time


def debounce(wait):
    def decorator(fn):
        sig = signature(fn)
        caller = {}

        def debounced(*args, **kwargs):
            nonlocal caller

            try:
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                called_args = fn.__name__ + str(dict(bound_args.arguments))
            except:
                called_args = ''

            t_ = time.time()

            def call_it(key):
                try:
                    # always remove on call
                    caller.pop(key)
                except:
                    pass

                fn(*args, **kwargs)

            try:
                # Always try to cancel timer
                caller[called_args].cancel()
            except:
                pass

            caller[called_args] = Timer(wait, call_it, [called_args])
            caller[called_args].start()

        return debounced

    return decorator

class EventQueue:
    def __init__(self, api_url, interval=0.5):
        self.queue = []
        self.interval = interval
        self.api_url = api_url
        self.lock = threading.Lock()


    def add_event(self, event):
        with self.lock:
            self.queue.append(event)
            self.send_events()

    @debounce(0.5)
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
