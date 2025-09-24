import logging
import tempfile
from pathlib import Path
from typing import Optional
from text_truncation_logger import (
    log_error_with_text,
    log_info_with_text,
    log_warning_with_text,
)

logger = logging.getLogger(__name__)

# Hardcoded template data for fallback
DECALOGO_INDUSTRIAL_TEMPLATE = """
1. Prioritize safety in all operations and decisions
2. Maintain highest quality standards in production
3. Ensure environmental responsibility and sustainability
4. Foster continuous improvement and innovation
5. Promote teamwork and effective communication
6. Uphold integrity and ethical business practices
7. Invest in employee development and well-being
8. Deliver value to customers and stakeholders
9. Embrace technology and digital transformation
10. Build resilient and adaptable operations
"""


def load_decalogo_industrial(target_path: Optional[str] = None) -> str:
    """
    Load DECALOGO_INDUSTRIAL template, attempting to write to file first,
    then falling back to in-memory template on failure.

    Args:
        target_path: Optional path where the template should be written

    Returns:
        The DECALOGO_INDUSTRIAL template content
    """
    template_content = DECALOGO_INDUSTRIAL_TEMPLATE.strip()

    if target_path:
        target_file = Path(target_path)

        try:
            # Create temporary file in the same directory as target
            temp_dir = target_file.parent
            temp_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=temp_dir,
                prefix=f".{target_file.name}_tmp_",
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as temp_file:
                temp_file.write(template_content)
                temp_file.flush()
                temp_path = Path(temp_file.name)

            # Atomically rename temporary file to target location
            temp_path.rename(target_file)
            log_info_with_text(
                logger,
                f"DECALOGO_INDUSTRIAL loaded and written to file: {target_path}",
                template_content,
            )

        except (PermissionError, OSError, IOError) as e:
            log_warning_with_text(
                logger,
                f"Failed to write DECALOGO_INDUSTRIAL to {target_path}: {e}. Using in-memory fallback template.",
                template_content,
            )
            # Clean up temp file if it exists
            try:
                if "temp_path" in locals() and temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass  # Best effort cleanup

            log_info_with_text(
                logger,
                "DECALOGO_INDUSTRIAL loaded from in-memory fallback template",
                template_content,
            )

        except Exception as e:
            log_error_with_text(
                logger,
                f"Unexpected error writing DECALOGO_INDUSTRIAL to {target_path}: {e}. Using in-memory fallback template.",
                template_content,
            )
            # Clean up temp file if it exists
            try:
                if "temp_path" in locals() and temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass  # Best effort cleanup

            log_info_with_text(
                logger,
                "DECALOGO_INDUSTRIAL loaded from in-memory fallback template",
                template_content,
            )
    else:
        log_info_with_text(
            logger,
            "DECALOGO_INDUSTRIAL loaded from in-memory template (no target path specified)",
            template_content,
        )

    return template_content


def get_decalogo_industrial(
    cache_path: Optional[str] = "decalogo_industrial.txt",
) -> str:
    """
    Convenience function to get DECALOGO_INDUSTRIAL with optional caching.

    Args:
        cache_path: Path where template should be cached, or None to skip caching

    Returns:
        The DECALOGO_INDUSTRIAL template content
    """
    return load_decalogo_industrial(cache_path)
