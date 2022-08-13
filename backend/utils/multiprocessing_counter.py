from torch import multiprocessing


class MultiprocessingCounter(object):
    def __init__(self):
        self.val = multiprocessing.Value("i", 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    def set(self, v: int):
        with self.val.get_lock():
            self.val.value = v

    @property
    def value(self):
        return self.val.value
