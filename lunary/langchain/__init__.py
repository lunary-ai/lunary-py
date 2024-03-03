try:
    from lunary.langchain.callback_handler import LunaryCallbackHandler
except ImportError:
    class LunaryCallbackHandler:
        def __init__(self):
            print("[Lunary] To use `LunaryCallbackHandler` you need to have the Langchain package installed. Please install it with `pip install langchain`")


__all__ = ['LunaryCallbackHandler']
