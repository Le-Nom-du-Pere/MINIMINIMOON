# -*- coding: utf-8 -*-
"""Proveedor centralizado de decálogos canónicos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import yaml

from .decalogo_loader_adapter import load_decalogos

CONFIG_PATH = Path("pdm_contra/config/decalogo.yaml")


class DecalogoProviderError(RuntimeError):
    """Error de configuración del proveedor de decálogos."""


def _read_config(path: Path = CONFIG_PATH) -> Dict[str, object]:
    if not path.exists():
        raise DecalogoProviderError(
            f"No existe configuración de decálogo en {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def provide_decalogos() -> Dict[str, object]:
    config = _read_config()
    if not config.get("autoload", False):
        raise DecalogoProviderError(
            "El autoload está deshabilitado en la configuración"
        )
    bundle_dir = Path(config.get("paths", {}).get("full", ".")).parent
    full_path = Path(config["paths"]["full"])
    industrial_path = Path(config["paths"]["industrial"])
    dnp_path = Path(config["paths"]["dnp"])
    crosswalk_path = Path(config.get(
        "crosswalk", bundle_dir / "crosswalk.latest.json"))

    for path in [full_path, industrial_path, dnp_path, crosswalk_path]:
        if not Path(path).exists():
            raise DecalogoProviderError(
                f"No se encuentra el archivo requerido: {path}")

    bundle = load_decalogos(
        [str(full_path), str(industrial_path), str(dnp_path)],
        crosswalk_path=str(crosswalk_path),
    )
    return bundle


__all__ = ["provide_decalogos", "DecalogoProviderError"]
