import logging
import time
from functools import wraps
import os

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Handlers
stream_handler = logging.StreamHandler()
log_filename = os.path.join(os.getcwd(), "output.log")  # Ensure the path is absolute
file_handler = logging.FileHandler(filename=log_filename)

# Set formatter
formatter = logging.Formatter("%(name)s %(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
if not logger.hasHandlers():  # Avoid adding multiple handlers if rerun
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

class TimeFilter(logging.Filter):
    def filter(self, record):
        return "Running" in record.getMessage()

logger.addFilter(TimeFilter())
logger.propagate = False  # Prevent log propagation to the root logger

def time_logger(func):
    """
    Decorator function to log the time taken by any function.

    This decorator logs the execution time of the decorated function. It logs the start time before the function
    execution, the end time after the function execution, and calculates the execution time. The function name and
    execution time are then logged at the INFO level.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time before function execution
        result = func(*args, **kwargs)  # Function execution
        end_time = time.time()  # End time after function execution
        execution_time = end_time - start_time  # Calculate execution time
        logger.info(f"Running {func.__name__}: --- {execution_time} seconds ---")
        return result

    return wrapper
