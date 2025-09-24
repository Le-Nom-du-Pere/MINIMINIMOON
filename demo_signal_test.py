#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo simplificado para mostrar el funcionamiento del manejo de señales
sin dependencias pesadas como spaCy, sentence-transformers, etc.
"""

import atexit
import json
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


# ==================== SISTEMA DE MONITOREO SIMPLIFICADO ====================
class SistemaMonitoreoDemo:
    """Versión demo del sistema de monitoreo para testing"""

    def __init__(self):
        self.ejecuciones = []
        self.tiempo_inicio = None
        self.trabajadores_activos = set()
        self.lock = threading.RLock()
        self.interrumpido = False

    def iniciar_monitoreo(self):
        self.tiempo_inicio = datetime.now()
        print(f"🚀 Sistema de monitoreo iniciado: {self.tiempo_inicio}")

    def registrar_trabajador(self, trabajador_id: str):
        with self.lock:
            self.trabajadores_activos.add(trabajador_id)
            print(f"➕ Trabajador registrado: {trabajador_id}")

    def desregistrar_trabajador(self, trabajador_id: str):
        with self.lock:
            self.trabajadores_activos.discard(trabajador_id)
            print(f"➖ Trabajador desregistrado: {trabajador_id}")

    def terminar_trabajadores(self):
        with self.lock:
            self.interrumpido = True
            trabajadores_copia = self.trabajadores_activos.copy()
        print(f"🛑 Terminando {len(trabajadores_copia)} trabajadores...")

    def registrar_ejecucion(self, nombre: str, resultado: dict):
        with self.lock:
            ejecucion = {
                "nombre": nombre,
                "resultado": resultado,
                "timestamp": datetime.now().isoformat(),
            }
            self.ejecuciones.append(ejecucion)
            print(
                f"📝 Ejecución registrada: {nombre} - {resultado.get('status', 'unknown')}"
            )

    def generar_dump_emergencia(self, output_dir: Path) -> Path:
        with self.lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dump_path = output_dir / f"dump_emergencia_demo_{timestamp}.json"

            estado = {
                "timestamp_dump": datetime.now().isoformat(),
                "sistema_interrumpido": self.interrumpido,
                "trabajadores_activos": list(self.trabajadores_activos),
                "tiempo_inicio": (
                    self.tiempo_inicio.isoformat() if self.tiempo_inicio else None
                ),
                "total_ejecuciones": len(self.ejecuciones),
                "ejecuciones": self.ejecuciones,
            }

            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(estado, f, indent=2, ensure_ascii=False)

            print(f"💾 Dump de emergencia guardado: {dump_path}")
            return dump_path


# ==================== MANEJO DE SEÑALES ====================
_sistema_monitoreo_global = None
_output_dir_global = None
_signal_handler_lock = threading.RLock()


def signal_handler(signum, frame):
    """Manejador de señales demo"""
    global _sistema_monitoreo_global, _output_dir_global

    with _signal_handler_lock:
        print(f"\n🚨 SEÑAL {signum} RECIBIDA - Iniciando terminación graciosa...")

        if _sistema_monitoreo_global:
            try:
                _sistema_monitoreo_global.terminar_trabajadores()

                if _output_dir_global:
                    dump_path = _sistema_monitoreo_global.generar_dump_emergencia(
                        _output_dir_global
                    )
                    print(f"📊 Estado guardado en: {dump_path}")

            except Exception as e:
                print(f"❌ Error durante terminación: {e}")

        print("🛑 Terminación graciosa completada")
        sys.exit(1)


def atexit_handler():
    """Handler para terminación inesperada"""
    global _sistema_monitoreo_global, _output_dir_global

    with _signal_handler_lock:
        print("\n🚨 Terminación inesperada detectada...")

        if _sistema_monitoreo_global and _output_dir_global:
            try:
                dump_path = _sistema_monitoreo_global.generar_dump_emergencia(
                    _output_dir_global
                )
                print(f"💾 Estado de emergencia guardado: {dump_path}")
            except Exception as e:
                print(f"❌ Error en atexit handler: {e}")


def simular_trabajo(trabajador_id: str, duracion: int):
    """Simula trabajo de procesamiento"""
    global _sistema_monitoreo_global

    if _sistema_monitoreo_global:
        _sistema_monitoreo_global.registrar_trabajador(trabajador_id)

    try:
        print(f"🔄 {trabajador_id} iniciando trabajo por {duracion}s...")

        for i in range(duracion):
            time.sleep(1)
            if _sistema_monitoreo_global and _sistema_monitoreo_global.interrumpido:
                print(f"⚠️  {trabajador_id} detectó interrupción, terminando...")
                break
            print(f"🔄 {trabajador_id} trabajando... ({i + 1}/{duracion})")

        # Simular resultado
        resultado = {
            "status": (
                "completed"
                if not (
                    _sistema_monitoreo_global and _sistema_monitoreo_global.interrumpido
                )
                else "interrupted"
            ),
            "puntaje": (
                85.5
                if not (
                    _sistema_monitoreo_global and _sistema_monitoreo_global.interrumpido
                )
                else 0
            ),
        }

        if _sistema_monitoreo_global:
            _sistema_monitoreo_global.registrar_ejecucion(
                f"plan_{trabajador_id}", resultado
            )

        print(f"✅ {trabajador_id} completado")

    except Exception as e:
        print(f"❌ Error en {trabajador_id}: {e}")
        if _sistema_monitoreo_global:
            _sistema_monitoreo_global.registrar_ejecucion(
                f"plan_{trabajador_id}", {"status": "failed", "error": str(e)}
            )
    finally:
        if _sistema_monitoreo_global:
            _sistema_monitoreo_global.desregistrar_trabajador(trabajador_id)


def main():
    """Función principal demo"""
    global _sistema_monitoreo_global, _output_dir_global

    print("🏭 DEMO: Sistema de Manejo de Señales")
    print("=" * 50)
    print("Presiona Ctrl+C para probar la terminación graciosa")
    print("=" * 50)

    # Configurar output
    output_dir = Path("demo_resultados")
    output_dir.mkdir(exist_ok=True)
    _output_dir_global = output_dir

    # Inicializar monitoreo
    sistema = SistemaMonitoreoDemo()
    _sistema_monitoreo_global = sistema
    sistema.iniciar_monitoreo()

    # Configurar handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(atexit_handler)

    # Simular procesamiento con threads
    threads = []
    for i in range(3):
        thread = threading.Thread(
            target=simular_trabajo, args=(f"worker_{i + 1}", 10), daemon=True
        )
        threads.append(thread)
        thread.start()

    try:
        # Esperar a que terminen los threads
        for thread in threads:
            thread.join()

        print("\n✅ Procesamiento completado normalmente")

    except KeyboardInterrupt:
        # El signal handler se encarga de esto
        pass

    print(f"\n📊 Resultados disponibles en: {output_dir}")


if __name__ == "__main__":
    main()
