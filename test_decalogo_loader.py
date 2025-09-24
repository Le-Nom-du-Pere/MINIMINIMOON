import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from decalogo_loader import (
    DECALOGO_INDUSTRIAL_TEMPLATE,
    get_decalogo_industrial,
    load_decalogo_industrial,
)


class TestDecalogoLoader:
    
    def test_load_decalogo_no_target_path(self, caplog):
        """Test loading without target path returns template and logs appropriately."""
        caplog.set_level(logging.INFO)
        result = load_decalogo_industrial(None)

        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert "loaded from in-memory template (no target path specified)" in caplog.text

    def test_load_decalogo_successful_file_write(self, tmp_path, caplog):
        """Test successful file write and atomic rename."""
        target_path = tmp_path / "decalogo.txt"
        caplog.set_level(logging.INFO)

        result = load_decalogo_industrial(str(target_path))
        
        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert target_path.exists()
        assert target_path.read_text(encoding='utf-8') == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert f"loaded and written to file: {target_path}" in caplog.text
    
    def test_load_decalogo_permission_error_fallback(self, caplog):
        """Test fallback to in-memory template on permission error."""
        caplog.set_level(logging.INFO)
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.side_effect = PermissionError("Access denied")
            
            result = load_decalogo_industrial("/restricted/path/decalogo.txt")
            
            assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
            assert "Failed to write DECALOGO_INDUSTRIAL" in caplog.text
            assert "PermissionError" in caplog.text or "Access denied" in caplog.text
            assert "loaded from in-memory fallback template" in caplog.text
    
    def test_load_decalogo_io_error_fallback(self, caplog):
        """Test fallback to in-memory template on I/O error."""
        caplog.set_level(logging.INFO)
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.side_effect = IOError("Disk full")
            
            result = load_decalogo_industrial("/tmp/decalogo.txt")
            
            assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
            assert "Failed to write DECALOGO_INDUSTRIAL" in caplog.text
            assert "loaded from in-memory fallback template" in caplog.text
    
    def test_load_decalogo_os_error_fallback(self, caplog):
        """Test fallback to in-memory template on OS error."""
        caplog.set_level(logging.INFO)
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.side_effect = OSError("No space left on device")
            
            result = load_decalogo_industrial("/tmp/decalogo.txt")
            
            assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
            assert "Failed to write DECALOGO_INDUSTRIAL" in caplog.text
            assert "loaded from in-memory fallback template" in caplog.text
    
    def test_load_decalogo_unexpected_error_fallback(self, caplog):
        """Test fallback to in-memory template on unexpected error."""
        caplog.set_level(logging.INFO)
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.side_effect = ValueError("Unexpected error")
            
            result = load_decalogo_industrial("/tmp/decalogo.txt")
            
            assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
            assert "Unexpected error writing DECALOGO_INDUSTRIAL" in caplog.text
            assert "loaded from in-memory fallback template" in caplog.text
    
    def test_load_decalogo_rename_failure(self, tmp_path, caplog):
        """Test fallback when atomic rename fails."""
        target_path = tmp_path / "decalogo.txt"
        caplog.set_level(logging.INFO)

        with patch('pathlib.Path.rename') as mock_rename:
            mock_rename.side_effect = PermissionError("Cannot rename file")
            
            result = load_decalogo_industrial(str(target_path))
            
            assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
            assert "Failed to write DECALOGO_INDUSTRIAL" in caplog.text
            assert "loaded from in-memory fallback template" in caplog.text
    
    def test_get_decalogo_convenience_function(self, tmp_path):
        """Test convenience function with caching."""
        cache_path = tmp_path / "cached_decalogo.txt"
        
        result = get_decalogo_industrial(str(cache_path))
        
        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert cache_path.exists()
        assert cache_path.read_text(encoding='utf-8') == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
    
    def test_get_decalogo_no_cache(self, caplog):
        """Test convenience function without caching."""
        caplog.set_level(logging.INFO)
        result = get_decalogo_industrial(None)

        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert "no target path specified" in caplog.text
    
    def test_temp_file_cleanup_on_error(self, tmp_path):
        """Test that temporary files are cleaned up when rename fails."""
        target_path = tmp_path / "decalogo.txt"
        
        with patch('pathlib.Path.rename') as mock_rename:
            mock_rename.side_effect = PermissionError("Cannot rename")
            
            load_decalogo_industrial(str(target_path))
            
            # Check that no temporary files are left behind
            temp_files = list(tmp_path.glob(".*_tmp_*"))
            assert len(temp_files) == 0
    
    def test_directory_creation(self, tmp_path):
        """Test that parent directories are created when they don't exist."""
        nested_path = tmp_path / "nested" / "dirs" / "decalogo.txt"

        result = load_decalogo_industrial(str(nested_path))

        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert nested_path.exists()
        assert nested_path.parent.exists()
        assert nested_path.read_text(encoding='utf-8') == DECALOGO_INDUSTRIAL_TEMPLATE.strip()

    def test_get_decalogo_idempotent(self, tmp_path):
        """Repeated calls should return the same template content."""
        cache_path = tmp_path / "cached.txt"
        first = get_decalogo_industrial(str(cache_path))
        second = get_decalogo_industrial(str(cache_path))

        assert first == second == DECALOGO_INDUSTRIAL_TEMPLATE.strip()


def test_decalogo_industrial_json_alignment():
    """El archivo JSON estructurado debe permanecer alineado con los eslabones industriales."""

    json_path = Path("decalogo_industrial.json")
    assert json_path.exists(), "Se requiere decalogo_industrial.json para validar la integración"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(data, list), "La raíz del decálogo debe ser una lista"
    assert len(data) == 10, "El decálogo industrial debe contener exactamente 10 dimensiones"

    allowed_tipos = {"INSUMOS", "PROCESOS", "PRODUCTOS", "RESULTADOS", "IMPACTOS"}

    for idx, dimension in enumerate(data, start=1):
        assert dimension.get("id") == idx, f"Dimensión esperada con id {idx}"
        for key in ("nombre", "cluster", "teoria_cambio", "eslabones"):
            assert key in dimension, f"Falta el campo obligatorio '{key}' en la dimensión {idx}"

        teoria_cambio = dimension["teoria_cambio"]
        assert set(teoria_cambio) == {
            "supuestos_causales",
            "mediadores",
            "resultados_intermedios",
            "precondiciones",
        }, f"Teoría de cambio incompleta en dimensión {idx}"
        assert isinstance(teoria_cambio["mediadores"], dict)

        eslabones = dimension["eslabones"]
        assert isinstance(eslabones, list) and eslabones, f"Dimensión {idx} debe incluir eslabones"

        for eslabon in eslabones:
            assert eslabon.get("id"), f"Eslabón sin identificador en dimensión {idx}"
            tipo = eslabon.get("tipo")
            assert tipo in allowed_tipos, f"Tipo de eslabón no reconocido: {tipo}"
            indicadores = eslabon.get("indicadores", [])
            assert isinstance(indicadores, list) and indicadores, "Cada eslabón requiere indicadores"
            capacidades = eslabon.get("capacidades_requeridas", [])
            assert isinstance(capacidades, list) and capacidades, "Cada eslabón requiere capacidades"
            puntos = eslabon.get("puntos_criticos", [])
            assert isinstance(puntos, list) and puntos, "Cada eslabón requiere puntos críticos"
            ventana = eslabon.get("ventana_temporal", [])
            assert (
                isinstance(ventana, list)
                and len(ventana) == 2
                and isinstance(ventana[0], (int, float))
                and isinstance(ventana[1], (int, float))
                and ventana[0] <= ventana[1]
            ), "La ventana temporal debe incluir un rango válido"
            ponderacion = eslabon.get("kpi_ponderacion")
            assert ponderacion is not None and float(ponderacion) > 0, "Cada eslabón requiere KPI positivo"
