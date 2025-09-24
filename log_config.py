import logging
import os
from typing import Optional


def configure_logging(log_level: Optional[str] = None) -> None:
    """
    Configure logging with LOG_LEVEL environment variable support.
    
    Args:
        log_level: Override log level, if None uses LOG_LEVEL env var
    """
    # Get log level from parameter or environment variable
    level_string = log_level or os.getenv('LOG_LEVEL', 'INFO')
    
    # Validate and convert log level string to logging constant
    level = get_log_level(level_string)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Log configuration info at DEBUG level
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured with level: {logging.getLevelName(level)} ({level})")
    
    # Optionally warn about invalid values
    if level_string.upper() != logging.getLevelName(level):
        logger.warning(f"Invalid LOG_LEVEL value '{level_string}' - falling back to INFO")


def get_log_level(level_string: str) -> int:
    """
    Convert log level string to logging constant with validation.
    
    Args:
        level_string: Log level as string (e.g., 'DEBUG', 'INFO', 'WARNING')
        
    Returns:
        Logging level constant (int)
    """
    # Normalize to uppercase
    level_string = level_string.upper()
    
    # Valid log levels
    valid_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'WARN': logging.WARNING,  # Common alias
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
        'FATAL': logging.CRITICAL  # Common alias
    }
    
    return valid_levels.get(level_string, logging.INFO)


# Configure logging on module import
configure_logging()