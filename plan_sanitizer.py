"""
Plan name sanitization and JSON key standardization utilities.

This module provides functions to:
1. Sanitize plan names for safe filesystem usage
2. Standardize JSON output keys to use underscore versions without tildes
3. Maintain Spanish tilded versions only in Markdown display text
"""

import os
import re
import unicodedata
from typing import Any, Dict


class PlanSanitizer:
    """Handles plan name sanitization and JSON key standardization."""

    # Characters that are invalid for filenames on various operating systems
    INVALID_CHARS = {
        # Windows forbidden characters
        "<",
        ">",
        ":",
        '"',
        "/",
        "\\",
        "|",
        "?",
        "*",
        # Additional problematic characters
        "\x00",
        "\x01",
        "\x02",
        "\x03",
        "\x04",
        "\x05",
        "\x06",
        "\x07",
        "\x08",
        "\x09",
        "\x0a",
        "\x0b",
        "\x0c",
        "\x0d",
        "\x0e",
        "\x0f",
        "\x10",
        "\x11",
        "\x12",
        "\x13",
        "\x14",
        "\x15",
        "\x16",
        "\x17",
        "\x18",
        "\x19",
        "\x1a",
        "\x1b",
        "\x1c",
        "\x1d",
        "\x1e",
        "\x1f",
    }

    # Windows reserved names
    RESERVED_NAMES = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }

    # Mapping of Spanish characters with tildes to underscore versions for JSON keys
    JSON_KEY_REPLACEMENTS = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "Á": "A",
        "É": "E",
        "Í": "I",
        "Ó": "O",
        "Ú": "U",
        "ñ": "n",
        "Ñ": "N",
        "ü": "u",
        "Ü": "U",
        # Common Spanish combinations that need standardization
        "línea": "linea",
        "número": "numero",
        "página": "pagina",
        "evaluación": "evaluacion",
        "implementación": "implementacion",
        "identificación": "identificacion",
        "descripción": "descripcion",
        "situación": "situacion",
        "población": "poblacion",
        "duración": "duracion",
        "versión": "version",
        "creación": "creacion",
        "modificación": "modificacion",
    }

    @staticmethod
    def sanitize_plan_name(plan_name: str, max_length: int = 255) -> str:
        """
        Sanitize a plan name for safe filesystem usage.

        Args:
            plan_name: The original plan name to sanitize
            max_length: Maximum length for the sanitized name (default 255)

        Returns:
            Sanitized plan name safe for filesystem usage

        Examples:
            >>> PlanSanitizer.sanitize_plan_name("Plan: Meta/Objetivo 2024*")
            'Plan - Meta-Objetivo 2024'

            >>> PlanSanitizer.sanitize_plan_name("Plan <importante>")
            'Plan (importante)'
        """
        if not plan_name or not plan_name.strip():
            return "plan_sin_nombre"

        # Start with original name
        sanitized = plan_name.strip()

        # Replace invalid characters with safe alternatives
        char_replacements = {
            "/": "-",
            "\\": "-",
            ":": " - ",
            "*": "",
            "?": "",
            '"': "'",
            "<": "(",
            ">": ")",
            "|": "-",
            "\t": " ",
            "\n": " ",
            "\r": " ",
        }

        for invalid_char, replacement in char_replacements.items():
            sanitized = sanitized.replace(invalid_char, replacement)

        # Remove any remaining invalid control characters
        sanitized = "".join(
            char for char in sanitized if char not in PlanSanitizer.INVALID_CHARS
        )

        # Handle consecutive brackets - collapse multiple to single
        sanitized = re.sub(r"\(+", "(", sanitized)
        sanitized = re.sub(r"\)+", ")", sanitized)

        # Normalize multiple spaces/dashes
        sanitized = re.sub(r"\s+", " ", sanitized)
        sanitized = re.sub(r"-+", "-", sanitized)
        # Keep spaces around dashes for readability unless at word boundaries
        sanitized = re.sub(r"\s*-\s*", " - ", sanitized)
        sanitized = re.sub(r"^\s*-\s*", "", sanitized)  # Remove leading dash
        sanitized = re.sub(r"\s*-\s*$", "", sanitized)  # Remove trailing dash

        # Remove leading/trailing spaces and dashes
        sanitized = sanitized.strip(" -.")

        # Handle reserved names
        base_name = sanitized.upper()
        if (
            base_name in PlanSanitizer.RESERVED_NAMES
            or base_name.split(".")[0] in PlanSanitizer.RESERVED_NAMES
        ):
            sanitized = f"plan_{sanitized}"

        # Truncate if too long while preserving readability
        if len(sanitized) > max_length:
            # Try to truncate at word boundary
            truncated = sanitized[:max_length]
            last_space = truncated.rfind(" ")
            last_dash = truncated.rfind("-")
            cut_point = max(last_space, last_dash)

            if (
                cut_point > max_length * 0.7
            ):  # Only cut at boundary if it's not too short
                sanitized = truncated[:cut_point]
            else:
                sanitized = truncated

            sanitized = sanitized.rstrip(" -.")

        # Ensure we have something after all processing
        if not sanitized:
            sanitized = "plan_sin_nombre"

        return sanitized

    @staticmethod
    def create_safe_directory(
        plan_name: str, base_path: str = ".", create: bool = True
    ) -> str:
        """
        Create a safe directory name and optionally create the directory.

        Args:
            plan_name: Original plan name
            base_path: Base path where directory should be created
            create: Whether to actually create the directory

        Returns:
            Full path to the safe directory
        """
        sanitized_name = PlanSanitizer.sanitize_plan_name(plan_name)
        full_path = os.path.join(base_path, sanitized_name)

        # Handle potential duplicates
        if os.path.exists(full_path):
            counter = 1
            while os.path.exists(f"{full_path}_{counter}"):
                counter += 1
            full_path = f"{full_path}_{counter}"

        if create:
            os.makedirs(full_path, exist_ok=True)

        return full_path

    @staticmethod
    def standardize_json_key(key: str) -> str:
        """
        Convert a JSON key to standardized underscore format without tildes.

        Args:
            key: Original key that may contain tildes or other characters

        Returns:
            Standardized key with underscores and no tildes

        Examples:
            >>> PlanSanitizer.standardize_json_key("línea_base")
            'linea_base'

            >>> PlanSanitizer.standardize_json_key("númeroPágina")
            'numero_pagina'
        """
        if not key:
            return key

        # First handle complete word replacements
        key_lower = key.lower()
        for spanish_word, english_word in PlanSanitizer.JSON_KEY_REPLACEMENTS.items():
            if len(spanish_word) > 1 and spanish_word in key_lower:
                # Use case-insensitive replacement
                pattern = re.compile(re.escape(spanish_word), re.IGNORECASE)
                key = pattern.sub(english_word, key)
                key_lower = key.lower()

        # Then handle individual character replacements
        for spanish_char, replacement in PlanSanitizer.JSON_KEY_REPLACEMENTS.items():
            if len(spanish_char) == 1:  # Only single character replacements
                key = key.replace(spanish_char, replacement)

        # Convert camelCase to snake_case
        key = re.sub("([a-z0-9])([A-Z])", r"\1_\2", key)

        # Replace spaces and dashes with underscores
        key = re.sub(r"[-\s]+", "_", key)

        # Remove non-alphanumeric characters except underscores
        # But preserve letters that were converted from tildes
        key = re.sub(r"[^a-zA-Z0-9_áéíóúñüÁÉÍÓÚÑÜ]", "", key)

        # Now handle any remaining tilded characters
        final_replacements = {
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "Á": "A",
            "É": "E",
            "Í": "I",
            "Ó": "O",
            "Ú": "U",
            "ñ": "n",
            "Ñ": "N",
            "ü": "u",
            "Ü": "U",
        }
        for tilded_char, replacement in final_replacements.items():
            key = key.replace(tilded_char, replacement)

        # Normalize multiple underscores
        key = re.sub(r"_+", "_", key)

        # Remove leading/trailing underscores
        key = key.strip("_")

        # Ensure lowercase
        key = key.lower()

        return key

    @staticmethod
    def standardize_json_object(obj: Any, preserve_display_keys: bool = True) -> Any:
        """
        Recursively standardize all keys in a JSON object/dictionary.

        Args:
            obj: The object to standardize (dict, list, or primitive)
            preserve_display_keys: Whether to preserve original keys with "_display" suffix

        Returns:
            Object with standardized keys
        """
        if isinstance(obj, dict):
            standardized = {}
            for key, value in obj.items():
                new_key = PlanSanitizer.standardize_json_key(key)

                # Preserve original key with tildes for display purposes if requested
                if (
                    preserve_display_keys
                    and new_key != key
                    and any(char in key for char in "áéíóúñüÁÉÍÓÚÑÜ")
                ):
                    standardized[f"{new_key}_display"] = key

                # Recursively standardize the value
                standardized[new_key] = PlanSanitizer.standardize_json_object(
                    value, preserve_display_keys
                )

            return standardized
        elif isinstance(obj, list):
            return [
                PlanSanitizer.standardize_json_object(
                    item, preserve_display_keys)
                for item in obj
            ]
        else:
            return obj

    @staticmethod
    def get_markdown_display_key(
        standardized_key: str, json_obj: Dict[str, Any]
    ) -> str:
        """
        Get the tilded Spanish version for Markdown display from a standardized key.

        Args:
            standardized_key: The underscore key without tildes
            json_obj: The JSON object that may contain display versions

        Returns:
            Display version with tildes if available, otherwise the standardized key
        """
        display_key = f"{standardized_key}_display"
        if display_key in json_obj:
            return json_obj[display_key]

        # Fallback: reverse lookup common patterns
        display_mappings = {
            "linea_base": "línea base",
            "numero_pagina": "número página",
            "evaluacion": "evaluación",
            "implementacion": "implementación",
            "identificacion": "identificación",
            "descripcion": "descripción",
            "situacion": "situación",
            "poblacion": "población",
            "duracion": "duración",
            "version": "versión",
            "creacion": "creación",
            "modificacion": "modificación",
        }

        return display_mappings.get(
            standardized_key, standardized_key.replace("_", " ")
        )


# Convenience functions for common use cases
def sanitize_plan_name(plan_name: str, max_length: int = 255) -> str:
    """Convenience function for plan name sanitization."""
    return PlanSanitizer.sanitize_plan_name(plan_name, max_length)


def standardize_json_keys(
    json_obj: Dict[str, Any], preserve_display: bool = True
) -> Dict[str, Any]:
    """Convenience function for JSON key standardization."""
    return PlanSanitizer.standardize_json_object(json_obj, preserve_display)


def create_plan_directory(plan_name: str, base_path: str = ".") -> str:
    """Convenience function to create a safe directory for a plan."""
    return PlanSanitizer.create_safe_directory(plan_name, base_path, create=True)
