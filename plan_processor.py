"""
Plan Processor Module with Retry Logic

This module provides comprehensive plan processing capabilities with sophisticated
error handling, retry logic, and logging functionality for industrial applications.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from log_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of error types for retry logic."""
    TRANSIENT = "transient"
    PERMANENT = "permanent"


class TransientErrorType(Enum):
    """Specific types of transient errors that can be retried."""
    FILE_PERMISSION = "file_permission"
    NETWORK_TIMEOUT = "network_timeout"
    IO_ERROR = "io_error"
    RESOURCE_BUSY = "resource_busy"


class PermanentErrorType(Enum):
    """Specific types of permanent errors that should not be retried."""
    FILE_NOT_FOUND = "file_not_found"
    OUT_OF_MEMORY = "out_of_memory"
    MALFORMED_PDF = "malformed_pdf"
    INVALID_FORMAT = "invalid_format"
    CORRUPTED_DATA = "corrupted_data"


@dataclass
class PlanProcessingError:
    """Structured error information for plan processing failures."""
    error_type: ErrorType
    specific_error: Union[TransientErrorType, PermanentErrorType]
    message: str
    traceback_info: str
    timestamp: datetime
    attempt_number: int
    plan_id: str
    plan_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class ErrorClassifier:
    """Classifies errors into transient or permanent categories."""

    def classify_error(self, error: Exception) -> Tuple[ErrorType, Union[TransientErrorType, PermanentErrorType]]:
        """
        Classify an exception into error type and specific category.

        Args:
            error: The exception to classify

        Returns:
            Tuple of (ErrorType, specific_error_type)
        """
        error_str = str(error).lower()
        error_type = type(error)

        # Permanent errors
        if isinstance(error, (FileNotFoundError, IsADirectoryError)):
            return ErrorType.PERMANENT, PermanentErrorType.FILE_NOT_FOUND

        if isinstance(error, MemoryError):
            return ErrorType.PERMANENT, PermanentErrorType.OUT_OF_MEMORY

        if isinstance(error, ValueError) and ("pdf" in error_str and ("corrupt" in error_str or "malform" in error_str)):
            return ErrorType.PERMANENT, PermanentErrorType.MALFORMED_PDF

        # Transient errors
        if isinstance(error, PermissionError):
            return ErrorType.TRANSIENT, TransientErrorType.FILE_PERMISSION

        if isinstance(error, TimeoutError):
            return ErrorType.TRANSIENT, TransientErrorType.NETWORK_TIMEOUT

        if isinstance(error, (OSError, IOError)):
            return ErrorType.TRANSIENT, TransientErrorType.IO_ERROR

        # Default to transient for unknown errors
        return ErrorType.TRANSIENT, TransientErrorType.IO_ERROR


class ErrorLogger:
    """Logs errors to structured files for analysis."""

    def __init__(self, log_directory: str):
        """
        Initialize error logger.

        Args:
            log_directory: Directory to store error logs
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

    def log_error(self, error: PlanProcessingError) -> str:
        """
        Log an error to a structured file.

        Args:
            error: The error to log

        Returns:
            Path to the log file
        """
        # Create filename based on error type and timestamp
        timestamp_str = error.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{error.error_type.value}_{error.specific_error.value}_{timestamp_str}_error.log"
        filepath = self.log_directory / filename

        # Write error details
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Error Type: {error.error_type.value}\n")
            f.write(f"Specific Error: {error.specific_error.value}\n")
            f.write(f"Message: {error.message}\n")
            f.write(f"Timestamp: {error.timestamp.isoformat()}\n")
            f.write(f"Attempt: {error.attempt_number}\n")
            f.write(f"Plan ID: {error.plan_id}\n")
            f.write(f"Parameters: {error.plan_parameters}\n")
            f.write("Traceback:\n")
            f.write(error.traceback_info)

@dataclass
class PlanProcessingResult:
    """Result of plan processing operation."""
    success: bool
    plan_id: str
    result_data: Optional[Dict[str, Any]] = None
    error: Optional[PlanProcessingError] = None
    attempts: int = 1
    processing_time: float = 0.0


class FeasibilityPlanProcessor:
    """Main processor for feasibility plans with retry logic."""

    def __init__(self, retry_config: Optional[RetryConfig] = None, log_directory: Optional[str] = None):
        """
        Initialize the plan processor.

        Args:
            retry_config: Configuration for retry logic
            log_directory: Directory for error logging
        """
        self.retry_config = retry_config or RetryConfig()
        self.error_logger = ErrorLogger(log_directory) if log_directory else None
        self.classifier = ErrorClassifier()

    def process_plan(self, plan_data: Dict[str, Any], plan_id: Optional[str] = None) -> PlanProcessingResult:
        """
        Process a feasibility plan with retry logic.

        Args:
            plan_data: Plan data to process
            plan_id: Unique identifier for the plan (generated if None)

        Returns:
            Processing result
        """
        if plan_id is None:
            # Generate plan ID from plan data
            import hashlib
            plan_str = str(sorted(plan_data.items()))
            plan_id = hashlib.md5(plan_str.encode()).hexdigest()[:8]

        attempt = 0
        last_error = None
        start_time = time.time()

        while attempt < self.retry_config.max_retries:
            attempt += 1
            try:
                result_data = self._execute_plan_processing(plan_data)
                processing_time = time.time() - start_time

                return PlanProcessingResult(
                    success=True,
                    plan_id=plan_id,
                    result_data=result_data,
                    attempts=attempt,
                    processing_time=processing_time
                )

            except Exception as exc:
                error_type, specific_error = self.classifier.classify_error(exc)

                error = PlanProcessingError(
                    error_type=error_type,
                    specific_error=specific_error,
                    message=str(exc),
                    traceback_info=str(exc),  # Simplified
                    timestamp=datetime.now(timezone.utc),
                    attempt_number=attempt,
                    plan_id=plan_id,
                    plan_parameters=plan_data
                )

                if self.error_logger:
                    self.error_logger.log_error(error)

                if error_type == ErrorType.PERMANENT:
                    processing_time = time.time() - start_time
                    return PlanProcessingResult(
                        success=False,
                        plan_id=plan_id,
                        error=error,
                        attempts=attempt,
                        processing_time=processing_time
                    )

                last_error = error

                # Calculate delay for retry
                if attempt < self.retry_config.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    LOGGER.info(f"Retrying plan {plan_id} in {delay:.2f} seconds (attempt {attempt})")
                    time.sleep(delay)

        # If we get here, all retries failed
        processing_time = time.time() - start_time
        return PlanProcessingResult(
            success=False,
            plan_id=plan_id,
            error=last_error,
            attempts=attempt,
            processing_time=processing_time
        )

    def _execute_plan_processing(self, plan_id: str, plan_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the actual plan processing logic.

        Args:
            plan_id: Plan identifier
            plan_parameters: Plan parameters

        Returns:
            Processing results
        """
        # Placeholder implementation - in real system this would process the plan
        LOGGER.info(f"Processing plan {plan_id} with parameters {plan_parameters}")

        # Simulate some processing time
        time.sleep(0.1)

        return {
            "plan_id": plan_id,
            "status": "completed",
            "results": plan_parameters,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempts with exponential backoff.

        Args:
            attempt: Current attempt number (1-based)

        Returns:
            Delay in seconds
        """
        delay = self.retry_config.base_delay * (self.retry_config.backoff_factor ** (attempt - 1))

        if self.retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 50% jitter

        return min(delay, self.retry_config.max_delay)


def create_sample_plans() -> List[Dict[str, Any]]:
    """
    Create sample plans for testing.

    Returns:
        List of sample plan dictionaries
    """
    return [
        {
            "id": "plan_001",
            "name": "Infrastructure Development",
            "parameters": {
                "budget": 1000000,
                "duration_months": 24,
                "risk_level": "medium"
            }
        },
        {
            "id": "plan_002",
            "name": "Education Program",
            "parameters": {
                "budget": 500000,
                "duration_months": 12,
                "risk_level": "low"
            }
        },
        {
            "id": "plan_003",
            "name": "Healthcare Initiative",
            "parameters": {
                "budget": 2000000,
                "duration_months": 36,
                "risk_level": "high"
            }
        }
    ]