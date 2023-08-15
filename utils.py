import inspect


def get_caller():
    stack = inspect.stack()
    # print(stack)
    caller_frame = stack[1]
    caller_name = caller_frame[3]
    print(caller_name)
    return caller_name
