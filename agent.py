import unittest
from llmonitor import default_input_parser, agent


class TestInputParser(unittest.TestCase):
    def test_empty_args_and_kwargs(self):
        result = default_input_parser()
        self.assertEqual(result, {"input": ""})

    def test_single_arg(self):
        result = default_input_parser(1)
        self.assertEqual(result, {"input": "1"})

    def test_multiple_args(self):
        result = default_input_parser(1, 2)
        self.assertEqual(result, {"input": [1, 2]})

    def test_single_kwarg(self):
        result = default_input_parser(a=1)
        self.assertEqual(result, {"input": [{"a": 1}]})

    def test_single_arg_and_kwarg(self):
        result = default_input_parser(1, a=1)
        self.assertEqual(result, {"input": [1, {"a": 1}]})

    def test_multiple_args_and_kwargs(self):
        result = default_input_parser(1, 2, a=1, b=2)
        self.assertEqual(result, {"input": [1, 2, {"a": 1, "b": 2}]})


@agent(name="my_agent", user_id="123", tags=["test"])
def my_agent():
    pass


if __name__ == "__main__":
    unittest.main()
