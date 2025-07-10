import contextlib
import logging
import time

LOGGER = logging.getLogger(__name__)


@contextlib.contextmanager
def log_time(process_name: str):
    start = time.time()
    yield
    duration = time.time() - start
    logging.info(f"{process_name} took {duration:.2f} seconds")
