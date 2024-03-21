from datetime import datetime
from typing import Callable


class TimeDiffPrinter:
    def __init__(self):
        self._t = None

    def start(self):
        self._t = datetime.now()

    def print(self, *args):
        """print time and restart stopwatch"""
        t2 = datetime.now()
        time_diff = t2 - self._t
        self._t = t2
        print(*args, f'time diff: {time_diff.seconds}s')


def time_func():
    def make_decorator(func: Callable):
        def decorator(*args, **kwargs):
            stop_watch = TimeDiffPrinter()
            stop_watch.start()
            ans = func(*args, **kwargs)
            stop_watch.print(func.__name__)
            return ans
        return decorator
    return make_decorator
