# logger_config.py
import logging
import os

def setup_logging(log_file="stimulation_system.log"):
    """
    Sets up a centralized logging configuration.
    Logs to console and a file.
    """
    # Ensure the log directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_file)

    # Create a logger
    logger = logging.getLogger("StimulationSystem")
    logger.setLevel(logging.DEBUG) # Set minimum logging level to DEBUG for file, INFO for console

    # Prevent adding multiple handlers if setup is called multiple times
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File Handler
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(logging.DEBUG) # Log more details to file
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":
    # Example usage
    logger = setup_logging()
    logger.info("Logging system initialized.")
    logger.debug("This is a debug message (only in file).")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")