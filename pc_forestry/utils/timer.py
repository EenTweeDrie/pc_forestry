import time
import logging
from contextlib import ContextDecorator

logger = logging.getLogger(__name__)


class Timer(ContextDecorator):
    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"{self.message} начато.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, rem = divmod(rem, 60)
        seconds, milliseconds = divmod(rem, 1)
        milliseconds = int(milliseconds * 1000)
        formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}:{milliseconds:03}"
        logger.info(f"{self.message} завершено за {formatted_time}.")
