"""
DECALOGO_INDUSTRIAL template loader with atomic file operations and fallback.
"""
import os
import tempfile
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback template in case file operations fail
DECALOGO_INDUSTRIAL_TEMPLATE = """# DECÁLOGO INDUSTRIAL

1. **Innovación Constante**: Invertir en investigación y desarrollo para mantener la competitividad.

2. **Sostenibilidad**: Implementar procesos productivos que minimicen el impacto ambiental.

3. **Digitalización**: Adoptar tecnologías de Industria 4.0 para optimizar la producción.

4. **Capital Humano**: Formar y retener talento especializado en nuevas tecnologías.

5. **Calidad y Estandarización**: Mantener los más altos estándares de calidad en productos y procesos.

6. **Internacionalización**: Buscar oportunidades en mercados globales para expandir operaciones.

7. **Colaboración**: Establecer alianzas estratégicas con universidades, centros de investigación y otras empresas.

8. **Infraestructura**: Modernizar las instalaciones para aumentar la eficiencia energética y productiva.

9. **Financiación**: Diversificar fuentes de financiación para proyectos industriales.

10. **Cumplimiento Normativo**: Adaptarse proactivamente a regulaciones nacionales e internacionales."""

# Cache for template content
_cached_template = None


def load_decalogo_industrial(file_path: str = "decalogo_industrial.txt") -> str:
    """
    Load the DECALOGO_INDUSTRIAL template with atomic file operations and fallback.

    Args:
        file_path: Path to the template file

    Returns:
        str: Content of the DECALOGO_INDUSTRIAL template
    """
    global _cached_template

    # Return cached template if available
    if _cached_template is not None:
        return _cached_template

    try:
        # Check if file exists
        if os.path.exists(file_path):
            logger.info(f"Loading DECALOGO_INDUSTRIAL from {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                _cached_template = content
                return content
        else:
            # Create file with template content using atomic operations
            logger.info(f"Creating DECALOGO_INDUSTRIAL at {file_path}")
            write_template_atomically(file_path, DECALOGO_INDUSTRIAL_TEMPLATE)
            _cached_template = DECALOGO_INDUSTRIAL_TEMPLATE
            return DECALOGO_INDUSTRIAL_TEMPLATE
    except (IOError, OSError, PermissionError) as e:
        # Fallback to hardcoded template on any file error
        logger.warning(
            f"Error accessing {file_path}: {e}. Using fallback template."
        )
        _cached_template = DECALOGO_INDUSTRIAL_TEMPLATE
        return DECALOGO_INDUSTRIAL_TEMPLATE


def write_template_atomically(file_path: str, content: str) -> bool:
    """
    Write template content to file using atomic operations.

    Args:
        file_path: Path where to write the template
        content: Template content to write

    Returns:
        bool: True if successful, False otherwise
    """
    # Get directory from file path
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except (IOError, OSError, PermissionError) as e:
            logger.error(f"Cannot create directory {directory}: {e}")
            return False

    try:
        # Create temporary file in the same directory
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=directory if directory else None,
            encoding="utf-8",
            prefix="decalogo_",
            suffix=".tmp",
        ) as temp_file:
            # Write content to temporary file
            temp_file.write(content)
            temp_path = temp_file.name

        # Rename temporary file to target path (atomic operation)
        os.replace(temp_path, file_path)
        logger.info(f"Successfully wrote template to {file_path}")
        return True
    except (IOError, OSError, PermissionError) as e:
        logger.error(f"Error writing template: {e}")
        try:
            # Clean up temporary file if it exists
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass
        return False


def get_decalogo_industrial() -> str:
    """
    Convenience wrapper to get the DECALOGO_INDUSTRIAL template.

    Returns:
        str: Content of the DECALOGO_INDUSTRIAL template
    """
    return load_decalogo_industrial()
