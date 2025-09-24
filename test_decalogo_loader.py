#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Sistema Integral de Evaluación de Cadenas de Valor en Planes de Desarrollo Municipal
Versión: 8.1 — Marco Teórico-Institucional con Análisis Causal Multinivel
"""

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from decalogo_loader import (
    DECALOGO_INDUSTRIAL_TEMPLATE,
    get_decalogo_industrial,
    load_decalogo_industrial,
)


class TestDecalogoLoader:
    """Test suite for industrial decalog loader with v8.1 specifications."""

    @staticmethod
    def test_load_decalogo_no_target_path(caplog):
        """Test loading without target path returns template and logs appropriately."""
        caplog.set_level(logging.INFO)
        result = load_decalogo_industrial(None)

        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert "loaded from in-memory template (no target path specified)" in caplog.text

    def test_load_decalogo_successful_file_write(self, tmp_path, caplog):
        """Test successful file write and atomic rename."""
        target_path = tmp_path / "decalogo.txt"

        result = load_decalogo_industrial(str(target_path))

        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert target_path.exists()
        assert target_path.read_text(encoding="utf-8") == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert f"loaded and written to file: {target_path}" in caplog.text

    def test_load_decalogo_permission_error_fallback(self, caplog):
        """Test fallback to in-memory template on permission error."""
        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.side_effect = PermissionError("Access denied")

            result = load_decalogo_industrial("/restricted/path/decalogo.txt")

            assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
            assert "Failed to write DECALOGO_INDUSTRIAL" in caplog.text
            assert "loaded from in-memory fallback template" in caplog.text

    def test_load_decalogo_io_error_fallback(self, caplog):
        """Test fallback to in-memory template on I/O error."""
        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.side_effect = IOError("Disk full")

            result = load_decalogo_industrial("/tmp/decalogo.txt")

            assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
            assert "Failed to write DECALOGO_INDUSTRIAL" in caplog.text
            assert "loaded from in-memory fallback template" in caplog.text

    def test_load_decalogo_os_error_fallback(self, caplog):
        """Test fallback to in-memory template on OS error."""
        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.side_effect = OSError("No space left on device")

            result = load_decalogo_industrial("/tmp/decalogo.txt")

            assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
            assert "Failed to write DECALOGO_INDUSTRIAL" in caplog.text
            assert "loaded from in-memory fallback template" in caplog.text

    def test_load_decalogo_unexpected_error_fallback(self, caplog):
        """Test fallback to in-memory template on unexpected error."""
        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.side_effect = ValueError("Unexpected error")

            result = load_decalogo_industrial("/tmp/decalogo.txt")

            assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
            assert "Unexpected error writing DECALOGO_INDUSTRIAL" in caplog.text
            assert "loaded from in-memory fallback template" in caplog.text

    def test_load_decalogo_rename_failure(self, tmp_path, caplog):
        """Test fallback when atomic rename fails."""
        target_path = tmp_path / "decalogo.txt"

        with patch("pathlib.Path.rename") as mock_rename:
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
        assert cache_path.read_text(encoding="utf-8") == DECALOGO_INDUSTRIAL_TEMPLATE.strip()

    def test_get_decalogo_no_cache(self, caplog):
        """Test convenience function without caching."""
        caplog.set_level(logging.INFO)
        result = get_decalogo_industrial(None)

        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert "no target path specified" in caplog.text

    def test_temp_file_cleanup_on_error(self, tmp_path):
        """Test that temporary files are cleaned up when rename fails."""
        target_path = tmp_path / "decalogo.txt"

        with patch("pathlib.Path.rename") as mock_rename:
            mock_rename.side_effect = PermissionError("Cannot rename")

            load_decalogo_industrial(str(target_path))

            temp_files = list(tmp_path.glob(".*_tmp_*"))
            assert len(temp_files) == 0

    def test_directory_creation(self, tmp_path):
        """Test that parent directories are created when they don't exist."""
        nested_path = tmp_path / "nested" / "dirs" / "decalogo.txt"

        result = load_decalogo_industrial(str(nested_path))

        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert nested_path.exists()
        assert nested_path.parent.exists()
        assert nested_path.read_text(encoding="utf-8") == DECALOGO_INDUSTRIAL_TEMPLATE.strip()

    @staticmethod
    def test_get_decalogo_idempotent(tmp_path):
        """Repeated calls should return the same template content."""
        cache_path = tmp_path / "cached.txt"
        first = get_decalogo_industrial(str(cache_path))
        second = get_decalogo_industrial(str(cache_path))

        assert first == second == DECALOGO_INDUSTRIAL_TEMPLATE.strip()


class TestDecalogoIndustrialJSONValidation:
    """Test suite for industrial JSON structure validation per v8.1 specifications."""

    @staticmethod
    def test_decalogo_industrial_json_structure():
        """Validate JSON structure aligns with v8.1 industrial framework."""
        json_path = Path("decalogo_industrial.json")
        
        if not json_path.exists():
            pytest.skip("decalogo_industrial.json not found - template generation test")
            
        data = json.loads(json_path.read_text(encoding="utf-8"))
        
        # Core structure validation
        assert isinstance(data, list), "Root must be list of 10 dimensions"
        assert len(data) == 10, "Industrial decalog requires exactly 10 dimensions"
        
        # v8.1 allowed types for value chain links
        allowed_tipos = {"INSUMOS", "PROCESOS", "PRODUCTOS", "RESULTADOS", "IMPACTOS"}
        
        for idx, dimension in enumerate(data, start=1):
            # Dimension structure validation
            assert dimension.get("id") == idx, f"Dimension {idx} ID mismatch"
            required_fields = {"nombre", "cluster", "teoria_cambio", "eslabones"}
            assert all(key in dimension for key in required_fields), f"Dimension {idx} missing fields"
            
            # Theory of Change validation (v8.1 causal framework)
            teoria_cambio = dimension["teoria_cambio"]
            tc_required = {"supuestos_causales", "mediadores", "resultados_intermedios", "precondiciones"}
            assert set(teoria_cambio.keys()) == tc_required, f"Theory of change incomplete in dimension {idx}"
            assert isinstance(teoria_cambio["mediadores"], dict), "Mediators must be dictionary"
            
            # Value chain links validation
            eslabones = dimension["eslabones"]
            assert isinstance(eslabones, list) and eslabones, f"Dimension {idx} requires links"
            
            for eslabon in eslabones:
                # Link structure validation
                assert eslabon.get("id"), f"Link missing ID in dimension {idx}"
                assert eslabon.get("tipo") in allowed_tipos, f"Invalid link type: {eslabon.get('tipo')}"
                
                # Indicators validation (v8.1 measurement framework)
                indicadores = eslabon.get("indicadores", [])
                assert isinstance(indicadores, list) and indicadores, "Each link requires indicators"
                
                # Capacities validation
                capacidades = eslabon.get("capacidades_requeridas", [])
                assert isinstance(capacidades, list) and capacidades, "Each link requires capacities"
                
                # Critical points validation
                puntos = eslabon.get("puntos_criticos", [])
                assert isinstance(puntos, list) and puntos, "Each link requires critical points"
                
                # Temporal window validation (industrial lead time)
                ventana = eslabon.get("ventana_temporal", [])
                assert (isinstance(ventana, list) and len(ventana) == 2 
                       and all(isinstance(x, (int, float)) for x in ventana)
                       and ventana[0] <= ventana[1]), "Invalid temporal window"
                
                # KPI weighting validation (v8.1 scoring system)
                ponderacion = eslabon.get("kpi_ponderacion")
                assert (ponderacion is not None and isinstance(ponderacion, (int, float)) 
                       and float(ponderacion) > 0), "Positive KPI weighting required"

    @staticmethod
    def test_decalogo_causal_coherence_validation():
        """Test causal coherence requirements per v8.1 framework."""
        json_path = Path("decalogo_industrial.json")
        
        if not json_path.exists():
            pytest.skip("decalogo_industrial.json not found")
            
        data = json.loads(json_path.read_text(encoding="utf-8"))
        
        for idx, dimension in enumerate(data, start=1):
            teoria_cambio = dimension["teoria_cambio"]
            
            # Causal assumptions validation
            supuestos = teoria_cambio["supuestos_causales"]
            assert len(supuestos) >= 1, f"Dimension {idx} needs causal assumptions"
            
            # Mediators categorization validation
            mediadores = teoria_cambio["mediadores"]
            assert len(mediadores) >= 1, f"Dimension {idx} needs mediator categories"
            for categoria, lista in mediadores.items():
                assert isinstance(lista, list) and lista, f"Empty mediator category: {categoria}"
            
            # Intermediate results validation
            resultados = teoria_cambio["resultados_intermedios"]
            assert len(resultados) >= 1, f"Dimension {idx} needs intermediate results"
            
            # Preconditions validation
            precondiciones = teoria_cambio["precondiciones"]
            assert len(precondiciones) >= 1, f"Dimension {idx} needs preconditions"

    @staticmethod
    def test_decalogo_value_chain_completeness():
        """Test value chain completeness per v8.1 industrial standards."""
        json_path = Path("decalogo_industrial.json")
        
        if not json_path.exists():
            pytest.skip("decalogo_industrial.json not found")
            
        data = json.loads(json_path.read_text(encoding="utf-8"))
        essential_types = {"INSUMOS", "PROCESOS", "PRODUCTOS"}
        
        for idx, dimension in enumerate(data, start=1):
            eslabones = dimension["eslabones"]
            tipos_presentes = {eslabon["tipo"] for eslabon in eslabones}
            
            # Essential links validation
            missing = essential_types - tipos_presentes
            assert not missing, f"Dimension {idx} missing essential links: {missing}"
            
            # KPI weighting range validation (v8.1 industrial calibration)
            for eslabon in eslabones:
                kpi = float(eslabon["kpi_ponderacion"])
                assert 0.1 <= kpi <= 2.0, f"KPI weighting out of range [0.1, 2.0]: {kpi}"