"""
TODO:  
- [ ] queue
"""


import pprint
import openai_utils
import openai
import threading
import utils
import traceback
import time
import uuid
import functools

openai.api_key = "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"
pp = pprint.PrettyPrinter(depth=10)
tls = threading.local()


class LLMonitor:
    api_url = "https://app.llmonitor.com"

    def __init__(self, app_id):
        self.app_id = None

    def track_event(
        self, event_type: str, event_name: str, run_id: str, input={}, output={}
    ):
        timestamp = time.time() * 1000

        event = {
            "event": event_name,
            "type": event_type,
            "app": self.app_id,
            "runId": run_id,
            "timestamp": timestamp,
            **input,
            **output,
        }
        pp.pprint(event)
        print("\n")

    def track_error(self, event_type):
        timestamp = time.time() * 1000

        event = {
            "event": "error",
            "type": event_type,
            "app": self.app_id,
            "timestamp": timestamp,
        }
        pp.pprint(event)
        print("\n")


llmonitor = LLMonitor("123")


def wrap(func, type: str, input_parser=lambda a: a, output_parser=lambda a: a):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not hasattr(tls, "run_ids"):
            tls.run_ids = []
        run_id = uuid.uuid4().hex
        tls.run_ids.append(run_id)

        try:
            if len(tls.run_ids) > 1:
                parent_run_id = tls.run_ids[-2]
                print(parent_run_id)
            input = input_parser(kwargs)

            llmonitor.track_event(type, "start", run_id, input=input)
            output = output_parser(func(*args, **kwargs))

            llmonitor.track_event(type, "end", run_id, output=output)
            return output
        except Exception as inst:
            print(inst)
            traceback.print_exc()
            llmonitor.track_event("llm", "error")  # TODO
        finally:
            tls.run_ids.pop()

    return wrapper


def monitor(entity):
    entity.type = getattr(entity, "type", None)

    if entity.__name__ == "ChatCompletion":
        entity.create = wrap(
            entity.create,
            "llm",
            input_parser=openai_utils.input_parser,
            output_parser=openai_utils.output_parser,
        )
    elif entity.__name__ == "ABC":
        print("ABC")
    elif entity.type == "agent":
        globals()[entity.__name__] = wrap(entity, "agent")
    elif entity.type == "tool":
        globals()[entity.__name__] = wrap(entity, "tool")


def my_tool():
    print("tool")


def my_agent():
    my_tool()


my_tool.type = "tool"
my_agent.type = "agent"

monitor(openai.ChatCompletion)
monitor(my_agent)
monitor(my_tool)

my_agent()
my_tool()

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}]
)
