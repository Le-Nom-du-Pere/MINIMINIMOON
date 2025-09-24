"""
Plan Processing with Retry Logic and Error Handling

This module implements retry logic with exponential backoff for transient I/O errors
during plan processing, with comprehensive error logging and classification.
"""

import hashlib
import json
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# Error classification
class ErrorType(Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"


class TransientErrorType(Enum):
    FILE_PERMISSION = "file_permission"
    NETWORK_TIMEOUT = "network_timeout"
    TEMPORARY_RESOURCE_UNAVAILABLE = "temporary_resource_unavailable"
    IO_ERROR = "io_error"


class PermanentErrorType(Enum):
    FILE_NOT_FOUND = "file_not_found"
    MALFORMED_PDF = "malformed_pdf"
    OUT_OF_MEMORY = "out_of_memory"
    INVALID_FORMAT = "invalid_format"
    CONFIGURATION_ERROR = "configuration_error"


@dataclass
class RetryConfig:
    max_retries: int = 1  # Limit to one retry attempt
    base_delay: float = 2.0  # Default 2 seconds
    exponential_base: float = 2.0
    max_delay: float = 60.0


@dataclass
class PlanProcessingError:
    error_type: ErrorType
    specific_error: Union[TransientErrorType, PermanentErrorType]
    message: str
    traceback_info: str
    timestamp: datetime
    attempt_number: int
    plan_id: Optional[str] = None
    plan_parameters: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingResult:
    success: bool
    plan_id: str
    result_data: Optional[Dict[str, Any]] = None
    error: Optional[PlanProcessingError] = None
    attempts: int = 1
    total_processing_time: float = 0.0


class ErrorClassifier:
    """Classifies errors as transient or permanent based on exception type and message."""

    @staticmethod
    def classify_error(
        exception: Exception,
    ) -> Tuple[ErrorType, Union[TransientErrorType, PermanentErrorType]]:
        """
        Classify an error as transient or permanent.

        Args:
            exception: The exception to classify

        Returns:
            Tuple of (ErrorType, specific error type)
        """
        exception_type = type(exception)
        error_message = str(exception).lower()

        # File permission errors - typically transient
        if isinstance(exception, PermissionError):
            return ErrorType.TRANSIENT, TransientErrorType.FILE_PERMISSION

        # Network-related errors - transient
        if "timeout" in error_message or "connection" in error_message:
            return ErrorType.TRANSIENT, TransientErrorType.NETWORK_TIMEOUT

        # Resource availability - transient
        if (
            "resource temporarily unavailable" in error_message
            or "device busy" in error_message
            or "try again" in error_message
        ):
            return (
                ErrorType.TRANSIENT,
                TransientErrorType.TEMPORARY_RESOURCE_UNAVAILABLE,
            )

        # I/O errors that might be transient
        if isinstance(exception, (OSError, IOError)) and exception.errno in [
            11,
            35,
            115,
        ]:  # EAGAIN, EWOULDBLOCK, EINPROGRESS
            return ErrorType.TRANSIENT, TransientErrorType.IO_ERROR

        # File not found - permanent
        if isinstance(exception, FileNotFoundError):
            return ErrorType.PERMANENT, PermanentErrorType.FILE_NOT_FOUND

        # PDF-related errors - permanent
        if "pdf" in error_message and (
            "corrupt" in error_message
            or "malformed" in error_message
            or "invalid" in error_message
        ):
            return ErrorType.PERMANENT, PermanentErrorType.MALFORMED_PDF

        # Memory errors - permanent
        if isinstance(exception, MemoryError) or "out of memory" in error_message:
            return ErrorType.PERMANENT, PermanentErrorType.OUT_OF_MEMORY

        # Format/parsing errors - permanent
        if (
            "format" in error_message and "invalid" in error_message
        ) or "parse error" in error_message:
            return ErrorType.PERMANENT, PermanentErrorType.INVALID_FORMAT

        # Check for ValueError with specific messages that should be permanent
        if isinstance(exception, ValueError) and (
            "format" in error_message or "invalid" in error_message
        ):
            return ErrorType.PERMANENT, PermanentErrorType.INVALID_FORMAT

        # Configuration errors - permanent
        if "configuration" in error_message or "config" in error_message:
            return ErrorType.PERMANENT, PermanentErrorType.CONFIGURATION_ERROR

        # Default to transient for unknown errors to allow retry
        return ErrorType.TRANSIENT, TransientErrorType.IO_ERROR


class ErrorLogger:
    """Manages error logging with individual log files per failed plan."""

    def __init__(self, log_directory: str = "error_logs"):
        self.log_directory = Path(log_directory)
        try:
            self.log_directory.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            # Fall back to current directory if log directory creation fails
            self.log_directory = Path.cwd() / "fallback_error_logs"
            self.log_directory.mkdir(exist_ok=True)

        # Setup main logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log_error(self, error: PlanProcessingError) -> str:
        """
        Log error to individual file and return the log file path.

        Args:
            error: The error to log

        Returns:
            Path to the created error log file
        """
        # Generate unique log filename
        plan_id = error.plan_id or "unknown_plan"
        timestamp = error.timestamp.strftime("%Y%m%d_%H%M%S")
        log_filename = f"{plan_id}_error.log"
        log_path = self.log_directory / log_filename

        # Create detailed error log entry
        error_data = {
            "timestamp": error.timestamp.isoformat(),
            "plan_id": error.plan_id,
            "error_type": error.error_type.value,
            "specific_error": error.specific_error.value,
            "message": error.message,
            "attempt_number": error.attempt_number,
            "plan_parameters": error.plan_parameters,
            "full_traceback": error.traceback_info,
        }

        # Write to individual error file
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"PLAN PROCESSING ERROR LOG\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Timestamp: {error.timestamp.isoformat()}\n")
                f.write(f"Plan ID: {error.plan_id}\n")
                f.write(
                    f"Error Type: {error.error_type.value} ({error.specific_error.value})\n"
                )
                f.write(f"Attempt Number: {error.attempt_number}\n")
                f.write(f"Message: {error.message}\n\n")

                if error.plan_parameters:
                    f.write("Plan Parameters:\n")
                    f.write("-" * 20 + "\n")
                    f.write(json.dumps(error.plan_parameters,
                            indent=2, default=str))
                    f.write("\n\n")

                f.write("Full Traceback:\n")
                f.write("-" * 20 + "\n")
                f.write(error.traceback_info)
                f.write("\n")

            # Also log to main logger
            self.logger.error(
                f"Plan {error.plan_id} failed with {error.error_type.value} error: {error.message} "
                f"(attempt {error.attempt_number}). Details logged to {log_path}"
            )

        except Exception as log_error:
            self.logger.error(
                f"Failed to write error log file {log_path}: {log_error}")

        return str(log_path)


class PlanProcessor(ABC):
    """Abstract base class for plan processors with retry logic."""

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        log_directory: str = "error_logs",
    ):
        self.retry_config = retry_config or RetryConfig()
        self.error_logger = ErrorLogger(log_directory)
        self.error_classifier = ErrorClassifier()

    @abstractmethod
    def _process_plan_implementation(
        self, plan_data: Dict[str, Any], plan_id: str
    ) -> Dict[str, Any]:
        """
        Implement the actual plan processing logic.

        Args:
            plan_data: The plan data to process
            plan_id: Unique identifier for the plan

        Returns:
            Processing result data

        Raises:
            Any exception that occurs during processing
        """
        pass

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.retry_config.base_delay * (
            self.retry_config.exponential_base ** (attempt - 1)
        )
        return min(delay, self.retry_config.max_delay)

    def process_plan(
        self, plan_data: Dict[str, Any], plan_id: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a plan with retry logic for transient errors.

        Args:
            plan_data: The plan data to process
            plan_id: Optional unique identifier for the plan

        Returns:
            ProcessingResult containing success status and results or error details
        """
        if plan_id is None:
            # Generate unique ID based on plan data hash
            plan_id = hashlib.md5(
                json.dumps(plan_data, sort_keys=True).encode()
            ).hexdigest()[:8]

        start_time = time.time()
        attempt = 1
        last_error = None

        while attempt <= self.retry_config.max_retries + 1:  # +1 for initial attempt
            try:
                # Attempt to process the plan
                result_data = self._process_plan_implementation(
                    plan_data, plan_id)

                # Success - return result
                total_time = time.time() - start_time
                self.error_logger.logger.info(
                    f"Plan {plan_id} processed successfully on attempt {attempt} "
                    f"in {total_time:.3f}s"
                )

                return ProcessingResult(
                    success=True,
                    plan_id=plan_id,
                    result_data=result_data,
                    attempts=attempt,
                    total_processing_time=total_time,
                )

            except Exception as e:
                # Classify the error
                error_type, specific_error = self.error_classifier.classify_error(
                    e)

                # Create error object
                error = PlanProcessingError(
                    error_type=error_type,
                    specific_error=specific_error,
                    message=str(e),
                    traceback_info=traceback.format_exc(),
                    timestamp=datetime.now(timezone.utc),
                    attempt_number=attempt,
                    plan_id=plan_id,
                    plan_parameters=plan_data,
                )

                # Log the error
                log_path = self.error_logger.log_error(error)
                last_error = error

                # For permanent errors, fail immediately
                if error_type == ErrorType.PERMANENT:
                    self.error_logger.logger.error(
                        f"Plan {plan_id} failed with permanent error on attempt {attempt}: "
                        f"{specific_error.value} - {str(e)}"
                    )
                    break

                # For transient errors, retry if we have attempts left
                if attempt <= self.retry_config.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.error_logger.logger.warning(
                        f"Plan {plan_id} failed with transient error on attempt {attempt}: "
                        f"{specific_error.value} - {str(e)}. Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                else:
                    self.error_logger.logger.error(
                        f"Plan {plan_id} failed after {attempt} attempts with transient error: "
                        f"{specific_error.value} - {str(e)}"
                    )

                attempt += 1

        # All attempts exhausted or permanent error
        total_time = time.time() - start_time
        return ProcessingResult(
            success=False,
            plan_id=plan_id,
            error=last_error,
            attempts=attempt,  # Don't subtract 1, attempt is correct count
            total_processing_time=total_time,
        )

    def batch_process_plans(
        self, plans: List[Tuple[Dict[str, Any], Optional[str]]]
    ) -> List[ProcessingResult]:
        """
        Process multiple plans sequentially.

        Args:
            plans: List of (plan_data, plan_id) tuples

        Returns:
            List of ProcessingResult objects
        """
        results = []
        start_time = time.time()

        self.error_logger.logger.info(
            f"Starting batch processing of {len(plans)} plans"
        )

        for i, (plan_data, plan_id) in enumerate(plans):
            result = self.process_plan(plan_data, plan_id)
            results.append(result)

            if (i + 1) % 10 == 0:  # Log progress every 10 plans
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                eta = avg_time * (len(plans) - i - 1)
                self.error_logger.logger.info(
                    f"Processed {i + 1}/{len(plans)} plans. "
                    f"Average: {avg_time:.3f}s/plan, ETA: {eta:.1f}s"
                )

        total_time = time.time() - start_time
        successes = sum(1 for r in results if r.success)
        failures = len(results) - successes

        self.error_logger.logger.info(
            f"Batch processing completed in {total_time:.3f}s. "
            f"Success: {successes}, Failures: {failures}"
        )

        return results


class FeasibilityPlanProcessor(PlanProcessor):
    """Concrete implementation for processing feasibility plans."""

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        log_directory: str = "error_logs",
    ):
        super().__init__(retry_config, log_directory)

        # Import feasibility scorer if available
        try:
            from feasibility_scorer import FeasibilityScorer

            self.scorer = FeasibilityScorer(
                enable_parallel=False
            )  # Disable parallel in processor
        except ImportError:
            self.scorer = None
            self.error_logger.logger.warning(
                "FeasibilityScorer not available, using mock implementation"
            )

    def _process_plan_implementation(
        self, plan_data: Dict[str, Any], plan_id: str
    ) -> Dict[str, Any]:
        """
        Process a feasibility plan by extracting and scoring indicators.

        Args:
            plan_data: Plan data containing indicators and metadata
            plan_id: Unique plan identifier

        Returns:
            Dictionary with processing results
        """
        # Simulate various error conditions for testing
        if "simulate_permission_error" in plan_data:
            raise PermissionError("Permission denied to access plan file")

        if "simulate_file_not_found" in plan_data:
            raise FileNotFoundError("Plan file not found")

        if "simulate_network_timeout" in plan_data:
            raise TimeoutError(
                "Network timeout while accessing remote plan data")

        if "simulate_malformed_pdf" in plan_data:
            raise ValueError("PDF file is corrupted or malformed")

        if "simulate_memory_error" in plan_data:
            raise MemoryError("Out of memory while processing large plan")

        # Extract indicators from plan data
        indicators = plan_data.get("indicators", [])
        if not indicators:
            raise FileNotFoundError(
                "No indicators found in plan data"
            )  # Use permanent error type

        # Process indicators
        if self.scorer:
            # Use real feasibility scorer
            results = []
            for indicator in indicators:
                if isinstance(indicator, str):
                    score_result = self.scorer.calculate_feasibility_score(
                        indicator)
                    results.append(
                        {
                            "indicator": indicator,
                            "feasibility_score": score_result.feasibility_score,
                            "quality_tier": score_result.quality_tier,
                            "components_detected": [
                                c.value for c in score_result.components_detected
                            ],
                            "has_quantitative_baseline": score_result.has_quantitative_baseline,
                            "has_quantitative_target": score_result.has_quantitative_target,
                        }
                    )
                else:
                    raise FileNotFoundError(
                        f"Invalid indicator format: {type(indicator)}"
                    )  # Use permanent error
        else:
            # Mock implementation
            results = []
            for indicator in indicators:
                if isinstance(indicator, str):
                    results.append(
                        {
                            "indicator": indicator,
                            "feasibility_score": 0.7,  # Mock score
                            "quality_tier": "medium",
                            "components_detected": ["baseline", "target"],
                            "has_quantitative_baseline": True,
                            "has_quantitative_target": False,
                        }
                    )

        # Return processing results
        return {
            "plan_id": plan_id,
            "total_indicators": len(indicators),
            "processed_indicators": len(results),
            "average_score": sum(r["feasibility_score"] for r in results)
            / len(results),
            "indicator_results": results,
            "processing_metadata": {
                "processor_version": "1.0.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "scorer_available": self.scorer is not None,
            },
        }


def create_sample_plans() -> List[Tuple[Dict[str, Any], str]]:
    """Create sample plans for testing the processor."""
    return [
        (
            {
                "name": "Plan de Desarrollo Social",
                "indicators": [
                    "Reducir la pobreza extrema del 15% actual al 8% para el año 2025",
                    "Aumentar la cobertura educativa básica hasta alcanzar el 95%",
                ],
            },
            "plan_001",
        ),
        (
            {
                "name": "Plan de Infraestructura",
                "indicators": [
                    "Construir 50 kilómetros de carreteras con presupuesto de 100 millones",
                    "Meta: mejorar conectividad vial en zonas rurales",
                ],
            },
            "plan_002",
        ),
        (
            {
                "name": "Test Error Plan",
                "simulate_permission_error": True,
                "indicators": ["Test indicator"],
            },
            "plan_error_001",
        ),
        (
            {
                "name": "Test Permanent Error",
                "simulate_file_not_found": True,
                "indicators": ["Another test indicator"],
            },
            "plan_error_002",
        ),
    ]


if __name__ == "__main__":
    # Example usage
    import json

    # Create processor with custom retry configuration
    retry_config = RetryConfig(max_retries=1, base_delay=2.0)
    processor = FeasibilityPlanProcessor(retry_config=retry_config)

    # Create sample plans
    sample_plans = create_sample_plans()

    print("Processing sample plans...")
    results = processor.batch_process_plans(sample_plans)

    # Display results
    for result in results:
        print("\n" + "=" * 60)
        print(f"Plan ID: {result.plan_id}")
        print(f"Success: {result.success}")
        print(f"Attempts: {result.attempts}")
        print(f"Processing Time: {result.total_processing_time:.3f}s")

        if result.success and result.result_data:
            print(
                f"Total Indicators: {result.result_data['total_indicators']}")
            print(f"Average Score: {result.result_data['average_score']:.3f}")
        elif result.error:
            print(f"Error Type: {result.error.error_type.value}")
            print(f"Specific Error: {result.error.specific_error.value}")
            print(f"Message: {result.error.message}")

    print(f"\nError logs available in: error_logs/")
