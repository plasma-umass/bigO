import sys
import textwrap
import time
import traceback

import termcolor
import threading


show_debug = False


def set_debug(value):
    global show_debug
    show_debug = value


class Output(threading.local):
    def __init__(self):
        self.pending = None
        self.contexts = []
        self.file = sys.stdout
        self.lock = threading.RLock()

    def timing_context(self, key):
        return self.OutputContext(self, key, color="cyan", start="[", end="]")

    class OutputContext:
        def __init__(
            self,
            outer,
            message,
            color = "cyan",
            start="",
            end="",
        ):
            self.outer = outer
            self.message = termcolor.colored(message + "...", color)
            self.color = color
            self.start = termcolor.colored(start, color)
            self.end = termcolor.colored(end, color)

        def __enter__(self):
            with self.outer.lock:
                self.outer.print_enter(self.message, start=self.start, end=self.end)
                self.outer.contexts += [self]
                self.start_time = time.time()

        def __exit__(self, exc_type, exc_value, traceback):
            with self.outer.lock:
                end_time = time.time()
                duration = int((end_time - self.start_time) * 1000)
                self.outer.contexts.pop()
                message = f"{termcolor.colored(f'{duration} ms', self.color)}"
                self.outer.print_exit(message, start=self.start, end=self.end)

    def write(self, message):
        self.file.write(message)
        self.file.flush()

    def flush(self):
        if self.pending is not None:
            self.file.write(self.pending)
            self.file.write("\n")
            self.pending = None

    def get_pad(self):
        return " " * self.pad_depth()

    def pad_depth(self):
        return len(self.contexts) * 2

    def print_enter(self, message, start="", end=""):
        self.flush()
        pad = self.get_pad()
        indented = textwrap.indent(f"{start}{message}", pad)
        self.write(indented)
        self.pending = end

    # assume message is one line
    def print_exit(self, message, start="", end=""):
        pad = self.get_pad()
        if self.pending is not None:
            if self.pending != end:
                self.write(f"Pending does not match end: {self.pending} != {end}")
            self.write(" ")
            self.write(message)
            self.write(self.pending)
            self.write("\n")
        else:
            self.write(f"{pad}{start}{message}{end}\n")
        self.pending = None

    def _format_exception(self, e):
        tb = e.__traceback__
        formatted_exception = "\nException:\n"
        formatted_exception += "".join(traceback.format_exception(type(e), e, tb))

        return formatted_exception

    def _print(self, color, args, start="", end=""):

        self.flush()
        start_len = len(start)
        pad = self.get_pad()

        message = " ".join(
            self._format_exception(a) if isinstance(a, Exception) else str(a)
            for a in args
        ).rstrip()

        lines = f"{start}{message}{termcolor.colored(end, color)}".split("\n")
        self.write(f"{pad}{termcolor.colored(lines[0], color)}")
        for line in lines[1:]:
            self.write("\n")
            self.write(f"{pad + (' ' * start_len)}{termcolor.colored(line, color)}")
        self.write("\n")

    def log(self, *args):
        with self.lock:
            self._print("cyan", args, start="[", end="]")


output = Output()


def log(*message):
    if show_debug:
        output.log(*message)


class EmptyContextManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def timer(key):
    if show_debug:
        return output.timing_context(key)
    else:
        return EmptyContextManager()


if __name__ == "__main__":
    # Test the logger
    output = Output()
    with output.timing_context("Main"):
        output.log("This is a log message")

    with output.timing_context("No time"):
        output.log("This is a log message")
        with output.timing_context("Beep"):
            output.log("This is a log message")
            with output.timing_context("Boop"):
                output.message("Real messages")
                output.log("This is a log message")
                output.error("MOo")
            output.log("This is a log message")
        with output.timing_context("Bop"):
            pass
