import sys
import threading

MAX_RECURSION = 2**(32-1)-1
MB_STACK_SIZE = 4096 * 256


class RecursionContext:
    def __init__(self, limit=MAX_RECURSION, stack = MB_STACK_SIZE):
        self.limit = limit
        self.stack = stack
        self.default_limit = sys.getrecursionlimit()
        self.default_stack = threading.stack_size()

    def __enter__(self):
        sys.setrecursionlimit(self.limit)
        threading.stack_size(self.stack)

    def __exit__(self, type, value, traceback):
        sys.setrecursionlimit(self.default_limit)
        threading.stack_size(self.default_stack)
