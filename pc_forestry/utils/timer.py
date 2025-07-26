import time
from loguru import logger
from contextlib import ContextDecorator
from functools import wraps


class Timer(ContextDecorator):
    """
    Таймер для измерения времени выполнения операций.
    Может использоваться как контекстный менеджер или декоратор.
    """

    def __init__(self, message=None):
        # Эта логика теперь не нужна, __call__ всё обработает.
        self.message = message

    def __enter__(self):
        self.start_time = time.time()
        # logger.info(f"'{self.message}' начато.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, rem = divmod(rem, 60)
        seconds, milliseconds = divmod(rem, 1)
        milliseconds = int(milliseconds * 1000)
        formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}:{milliseconds:03}"
        logger.info(f"'{self.message}' завершено за {formatted_time}.")

    def __call__(self, func):
        # Эта функция вызывается, только когда Timer используется как декоратор.
        # self.message будет либо строкой (для @Timer("..."))
        # либо None (для @Timer).

        # Если @Timer используется без аргументов, self.message будет None.
        # В этом случае мы берем имя функции.
        if self.message is None:
            self.message = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 'self' здесь — это экземпляр Timer, созданный декоратором.
            # Он содержит правильное сообщение.
            with self:
                return func(*args, **kwargs)
        return wrapper
