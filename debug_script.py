import os
import sys
import logging
from typing import Any
import modal
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_modal_debugging() -> None:
    """Set up Modal debugging environment."""
    # Enable Modal debug logging
    os.environ['MODAL_DEBUG'] = '1'
    logger.info("Modal debugging enabled")

def print_modal_info() -> None:
    """Print Modal configuration and environment information."""
    logger.info("Modal Configuration:")
    logger.info(f"Modal Version: {modal.__version__}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Current Working Directory: {os.getcwd()}")
    
    # Check for Modal token
    token = os.environ.get('MODAL_TOKEN_ID')
    if token:
        logger.info("Modal token is set")
    else:
        logger.warning("Modal token is not set")

def debug_function(func: Any) -> Any:
    """Decorator to add debugging information to a function."""
    def wrapper(*args, **kwargs):
        logger.info(f"Calling function: {func.__name__}")
        logger.info(f"Arguments: {args}")
        logger.info(f"Keyword arguments: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def check_modal_connection() -> bool:
    """Check if Modal connection is working."""
    try:
        # Try to create a simple stub to test connection
        stub = modal.Stub("debug-stub")
        logger.info("Modal connection successful")
        return True
    except Exception as e:
        logger.error(f"Modal connection failed: {str(e)}")
        return False

def main():
    """Main debugging function."""
    setup_modal_debugging()
    print_modal_info()
    
    if check_modal_connection():
        logger.info("Modal environment is properly configured")
    else:
        logger.error("Modal environment needs attention")

if __name__ == "__main__":
    main() 