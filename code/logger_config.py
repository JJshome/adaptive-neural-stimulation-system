# adaptive-neural-stimulation-system/code/logger_config.py
import logging
import os

def setup_logging(log_file="stimulation_system.log"):
    """
    Sets up a centralized logging configuration.
    Logs to console and a file.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_file)

    logger = logging.getLogger("StimulationSystem")
    logger.setLevel(logging.DEBUG) # Overall lowest level

    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO) # Console shows INFO and above
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File Handler
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(logging.DEBUG) # File shows DEBUG and above
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
