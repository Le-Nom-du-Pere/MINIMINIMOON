        except:
            return 'neutral'
    
    def _calcular_puntaje_innovacion(self, evidencia: Dict[str, Any]) -> float:
        """Calcula un puntaje de innovación basado en características avanzadas."""
        try:
            puntaje_innovacion = 0.0
            
            # Factor de densidad causal alta
            densidad_causal = evidencia.get('densidad_causal_agregada', 0.0)
            if densidad_causal > 0.7:
                puntaje_innovacion += 0.3
            elif densidad_causal > 0.5:
                puntaje_innovacion += 0.15
            
            # Factor de cuantificación
            if evidencia.get('tiene_cuantificacion', False):
                puntaje_innovacion += 0.25
            
            # Factor de temporalidad explícita
            if evidencia.get('temporalidad_explicita', False):
                puntaje_innovacion += 0.2
            
            # Factor de nivel institucional específico
            nivel_inst = evidencia.get('nivel_institucional', 'indeterminado')
            if nivel_inst != 'indeterminado':
                puntaje_innovacion += 0.15
            
            # Factor de calidad de contenido alta
            calidad = evidencia.get('calidad_contenido', 0.5)
            if calidad > 0.8:
                puntaje_innovacion += 0.1
            
            return min(1.0, puntaje_innovacion)
        
        except Exception:
            return 0.3
    
    def buscar_evidencia_causal_avanzadas(
        self, query: str, conceptos_clave: List[str], top_k: int = 5, umbral_certeza: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Método auxiliar para búsqueda con configuración estándar."""
        return self.buscar_evidencia_causal_avanzada(
            query, conceptos_clave, top_k, umbral_certeza
        )

# -------------------- Evaluación causal avanzada --------------------
@dataclass
class EvaluacionCausalIndustrialAvanzada:
    """Evaluación causal avanzada con capacidades matemáticas de frontera."""
    consistencia_logica: float
    identificabilidad_causal: float
    factibilidad_operativa: float
    certeza_probabilistica: float
    robustez_causal: float
    innovacion_metodologica: float
    sostenibilidad_resultados: float
    escalabilidad_territorial: float
    riesgos_implementacion: List[str]
    supuestos_criticos: List[str]
    evidencia_soporte: int
    brechas_criticas: int
    factores_contextuales: Dict[str, float] = field(default_factory=dict)
    interdependencias_detectadas: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validación avanzada de campos."""
        campos_numericos = [
            'consistencia_logica', 'identificabilidad_causal', 'factibilidad_operativa',
            'certeza_probabilistica', 'robustez_causal', 'innovacion_metodologica',
            'sostenibilidad_resultados', 'escalabilidad_territorial'
        ]
        
        for campo in campos_numericos:
            valor = getattr(self, campo)
            if not isinstance(valor, (int, float)) or not (0.0 <= valor <= 1.0):
                raise ValueError(f"Campo {campo} debe estar entre 0.0 y 1.0, recibido: {valor}")
    
    @property
    def puntaje_global_avanzado(self) -> float:
        """Cálculo de puntaje global con pesos sofisticados."""
        # Pesos adaptivos basados en contexto
        peso_base = {
            'consistencia_logica': 0.18,
            'identificabilidad_causal': 0.16,
            'factibilidad_operativa': 0.15,
            'certeza_probabilistica': 0.14,
            'robustez_causal': 0.13,
            'innovacion_metodologica': 0.10,
            'sostenibilidad_resultados': 0.08,
            'escalabilidad_territorial': 0.06
        }
        
        # Ajustes por factores contextuales
        pesos_ajustados = peso_base.copy()
        
        # Si hay alta incertidumbre, dar más peso a certeza probabilística
        if self.certeza_probabilistica < 0.6:
            pesos_ajustados['certeza_probabilistica'] *= 1.2
            pesos_ajustados['robustez_causal'] *= 1.1
        
        # Si hay muchos riesgos, dar más peso a factibilidad
        if len(self.riesgos_implementacion) > 5:
            pesos_ajustados['factibilidad_operativa'] *= 1.15
        
        # Normalizar pesos
        suma_pesos = sum(pesos_ajustados.values())
        pesos_normalizados = {k: v/suma_pesos for k, v in pesos_ajustados.items()}
        
        # Cálculo ponderado
        puntaje = (
            self.consistencia_logica * pesos_normalizados['consistencia_logica'] +
            self.identificabilidad_causal * pesos_normalizados['identificabilidad_causal'] +
            self.factibilidad_operativa * pesos_normalizados['factibilidad_operativa'] +
            self.certeza_probabilistica * pesos_normalizados['certeza_probabilistica'] +
            self.robustez_causal * pesos_normalizados['robustez_causal'] +
            self.innovacion_metodologica * pesos_normalizados['innovacion_metodologica'] +
            self.sostenibilidad_resultados * pesos_normalizados['sostenibilidad_resultados'] +
            self.escalabilidad_territorial * pesos_normalizados['escalabilidad_territorial']
        )
        
        return puntaje
    
    @property
    def nivel_certidumbre_avanzado(self) -> str:
        """Clasificación avanzada de certidumbre con más granularidad."""
        p = self.puntaje_global_avanzado
        cert = self.certeza_probabilistica
        rob = self.robustez_causal
        
        # Análisis multidimensional
        if p >= 0.9 and cert >= 0.8 and rob >= 0.8:
            return "EXCELENTE - Certidumbre científica sólida"
        elif p >= 0.85 and cert >= 0.75:
            return "MUY ALTA - Certidumbre robusta con evidencia fuerte"
        elif p >= 0.75 and cert >= 0.65:
            return "ALTA - Certidumbre sólida con algunas limitaciones menores"
        elif p >= 0.65 and cert >= 0.55:
            return "MEDIA-ALTA - Certidumbre aceptable con evidencia moderada"
        elif p >= 0.55 and cert >= 0.45:
            return "MEDIA - Certidumbre limitada, requiere fortalecimiento"
        elif p >= 0.4 and cert >= 0.35:
            return "MEDIA-BAJA - Certidumbre insuficiente, evidencia débil"
        elif p >= 0.3:
            return "BAJA - Alta incertidumbre, evidencia muy limitada"
        else:
            return "CRÍTICA - Incertidumbre extrema, rediseño urgente necesario"
    
    @property
    def recomendacion_estrategica_avanzada(self) -> str:
        """Recomendación estratégica sofisticada basada en análisis multifactorial."""
        # Análisis de patrones complejos
        if (self.factibilidad_operativa < 0.5 and 
            len(self.riesgos_implementacion) > 5 and 
            self.robustez_causal < 0.6):
            return "REDISEÑO INTEGRAL URGENTE - Problemas estructurales graves"
        
        elif (self.certeza_probabilistica < 0.6 and 
              self.evidencia_soporte < 5 and 
              self.identificabilidad_causal < 0.7):
            return "FORTALECIMIENTO DE EVIDENCIA CRÍTICO - Base empírica insuficiente"
        
        elif (self.consistencia_logica < 0.6 and 
              len(self.supuestos_criticos) > 8):
            return "REVISIÓN TEÓRICA FUNDAMENTAL - Marco conceptual inadecuado"
        
        elif (self.factibilidad_operativa < 0.7 and 
              self.escalabilidad_territorial < 0.5):
            return "OPTIMIZACIÓN OPERATIVA REQUERIDA - Problemas de implementación"
        
        elif (self.sostenibilidad_resultados < 0.6 and 
              self.innovacion_metodologica < 0.5):
            return "FORTALECIMIENTO DE SOSTENIBILIDAD - Riesgo de deterioro"
        
        elif self.puntaje_global_avanzado >= 0.85:
            return "IMPLEMENTACIÓN RECOMENDADA - Condiciones óptimas"
        
        elif self.puntaje_global_avanzado >= 0.7:
            return "IMPLEMENTACIÓN GRADUAL - Con mejoras específicas"
        
        elif self.puntaje_global_avanzado >= 0.55:
            return "IMPLEMENTACIÓN PILOTO - Fase de prueba y ajuste"
        
        else:
            return "DESARROLLO ADICIONAL REQUERIDO - No listo para implementación"
    
    def calcular_indice_madurez_tecnologica(self) -> Dict[str, Any]:
        """Calcula índice de madurez tecnológica adaptado a políticas públicas."""
        try:
            # TRL adaptado a políticas públicas
            factores_trl = {
                'concepto_basico': self.consistencia_logica,
                'formulacion_tecnologica': self.identificabilidad_causal,
                'prueba_concepto_analitica': self.certeza_probabilistica,
                'validacion_laboratorio': min(1.0, self.evidencia_soporte / 10),
                'validacion_entorno_relevante': self.factibilidad_operativa,
                'demostracion_entorno_operativo': self.robustez_causal,
                'demostracion_prototipo_sistema': self.innovacion_metodologica,
                'sistema_completo_calificado': self.sostenibilidad_resultados,
                'sistema_probado_operacional': self.escalabilidad_territorial
            }
            
            # Cálculo de TRL
            trl_scores = list(factores_trl.values())
            trl_promedio = np.mean(trl_scores)
            
            # Mapeo a escala TRL 1-9
            trl_nivel = min(9, max(1, int(trl_promedio * 9) + 1))
            
            # Interpretación de TRL
            interpretaciones_trl = {
                1: "Principios básicos observados",
                2: "Concepto de tecnología formulado",
                3: "Prueba de concepto analítica/experimental",
                4: "Validación de componentes en laboratorio",
                5: "Validación de componentes en entorno relevante",
                6: "Demostración de tecnología en entorno operativo",
                7: "Demostración de prototipo de sistema en entorno operativo",
                8: "Sistema completo y calificado",
                9: "Sistema probado exitosamente en entorno operacional"
            }
            
            return {
                'trl_nivel': trl_nivel,
                'trl_interpretacion': interpretaciones_trl.get(trl_nivel, "Nivel indeterminado"),
                'factores_detalle': factores_trl,
                'recomendacion_desarrollo': self._generar_recomendacion_trl(trl_nivel),
                'factores_limitantes': self._identificar_factores_limitantes_trl(factores_trl)
            }
        
        except Exception as e:
            return {
                'trl_nivel': 3,
                'trl_interpretacion': "Error en cálculo - nivel por defecto",
                'error': str(e)
            }
    
    def _generar_recomendacion_trl(self, trl_nivel: int) -> str:
        """Genera recomendación específica basada en TRL."""
        if trl_nivel <= 3:
            return "Fortalecer marco conceptual y evidencia básica"
        elif trl_nivel <= 5:
            return "Desarrollar pruebas piloto y validación de componentes"
        elif trl_nivel <= 7:
            return "Implementar demostraciones en entorno operativo"
        else:
            return "Sistema listo para implementación a escala"
    
    def _identificar_factores_limitantes_trl(self, factores: Dict[str, float]) -> List[str]:
        """Identifica factores que limitan el avance en TRL."""
        factores_limitantes = []
        umbral_limitante = 0.5
        
        for factor, valor in factores.items():
            if valor < umbral_limitante:
                factores_limitantes.append(f"{factor}: {valor:.2f}")
        
        return factores_limitantes
    
    def generar_matriz_decision_multicriterio(self) -> Dict[str, Any]:
        """Genera matriz de decisión multicriterio avanzada."""
        try:
            # Criterios y pesos
            criterios = {
                'Viabilidad Técnica': {
                    'valor': (self.consistencia_logica + self.robustez_causal) / 2,
                    'peso': 0.25
                },
                'Viabilidad Económica': {
                    'valor': self.factibilidad_operativa,
                    'peso': 0.20
                },
                'Viabilidad Social': {
                    'valor': (self.escalabilidad_territorial + self.sostenibilidad_resultados) / 2,
                    'peso': 0.20
                },
                'Riesgo': {
                    'valor': max(0, 1.0 - len(self.riesgos_implementacion) / 10),
                    'peso': 0.15
                },
                'Innovación': {
                    'valor': self.innovacion_metodologica,
                    'peso': 0.10
                },
                'Evidencia': {
                    'valor': min(1.0, self.evidencia_soporte / 15),
                    'peso': 0.10
                }
            }
            
            # Cálculo de utilidad multiatributo
            utilidad_total = sum(
                criterio['valor'] * criterio['peso'] 
                for criterio in criterios.values()
            )
            
            # Análisis de sensibilidad simple
            sensibilidades = {}
            for nombre_criterio in criterios:
                # Aumentar peso del criterio en 10% y recalcular
                criterios_temp = {k: v.copy() for k, v in criterios.items()}
                criterios_temp[nombre_criterio]['peso'] *= 1.1
                
                # Renormalizar pesos
                suma_pesos = sum(c['peso'] for c in criterios_temp.values())
                for c in criterios_temp.values():
                    c['peso'] /= suma_pesos
                
                utilidad_sensibilidad = sum(
                    c['valor'] * c['peso'] for c in criterios_temp.values()
                )
                
                sensibilidades[nombre_criterio] = utilidad_sensibilidad - utilidad_total
            
            return {
                'criterios_evaluacion': criterios,
                'utilidad_multiatributo': utilidad_total,
                'ranking_criterios': sorted(
                    criterios.items(), 
                    key=lambda x: x[1]['valor'] * x[1]['peso'], 
                    reverse=True
                ),
                'analisis_sensibilidad': sensibilidades,
                'criterio_mas_critico': min(criterios.items(), key=lambda x: x[1]['valor'])[0],
                'recomendacion_priorizacion': self._generar_recomendacion_priorizacion(criterios)
            }
        
        except Exception as e:
            return {'error': f"Error en matriz de decisión: {e}"}
    
    def _generar_recomendacion_priorizacion(self, criterios: Dict[str, Dict]) -> str:
        """Genera recomendación de priorización basada en criterios."""
        criterio_mas_bajo = min(criterios.items(), key=lambda x: x[1]['valor'])
        nombre_criterio, datos_criterio = criterio_mas_bajo
        
        if datos_criterio['valor'] < 0.4:
            return f"PRIORIDAD CRÍTICA: Fortalecer {nombre_criterio} (valor actual: {datos_criterio['valor']:.2f})"
        elif datos_criterio['valor'] < 0.6:
            return f"PRIORIDAD ALTA: Mejorar {nombre_criterio} (valor actual: {datos_criterio['valor']:.2f})"
        else:
            return f"OPTIMIZACIÓN: Refinar {nombre_criterio} para maximizar impacto"

# Continuación de la estructura de evaluación...
@dataclass
class ResultadoDimensionIndustrialAvanzado:
    """Resultado avanzado de evaluación dimensional con capacidades de frontera."""
    dimension: DimensionDecalogoAvanzada
    evaluacion_causal: EvaluacionCausalIndustrialAvanzada
    evidencia: Dict[str, List[Dict[str, Any]]]
    brechas_identificadas: List[str]
    recomendaciones: List[str]
    matriz_trazabilidad: Optional[pd.DataFrame] = None
    analisis_interdependencias: Dict[str, Any] = field(default_factory=dict)
    proyeccion_impacto: Dict[str, float] = field(default_factory=dict)
    analisis_costo_beneficio: Dict[str, Any] = field(default_factory=dict)
    timestamp_evaluacion: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def puntaje_final_avanzado(self) -> float:
        """Puntaje final avanzado con ajustes contextuales."""
        puntaje_base = self.evaluacion_causal.puntaje_global_avanzado * 100
        
        # Ajustes por contexto
        ajuste_evidencia = min(5, self.evaluacion_causal.evidencia_soporte * 0.5)
        ajuste_riesgos = max(-10, -len(self.evaluacion_causal.riesgos_implementacion) * 0.8)
        ajuste_brechas = max(-8, -len(self.brechas_identificadas) * 0.6)
        
        puntaje_ajustado = puntaje_base + ajuste_evidencia + ajuste_riesgos + ajuste_brechas
        
        return max(0, min(100, puntaje_ajustado))
    
    @property
    def nivel_madurez_avanzado(self) -> str:
        """Nivel de madurez avanzado con más granularidad."""
        p = self.puntaje_final_avanzado
        trl_info = self.evaluacion_causal.calcular_indice_madurez_tecnologica()
        trl_nivel = trl_info.get('trl_nivel', 3)
        
        if p >= 90 and trl_nivel >= 8:
            return "NIVEL 5+ - Optimizado y Escalable"
        elif p >= 85 and trl_nivel >= 7:
            return "NIVEL 5 - Optimizado"
        elif p >= 75 and trl_nivel >= 6:
            return "NIVEL 4+ - Gestionado Cuantitativamente Avanzado"
        elif p >= 65 and trl_nivel >= 5:
            return "NIVEL 4 - Gestionado Cuantitativamente"
        elif p >= 55 and trl_nivel >= 4:
            return "NIVEL 3+ - Definido Avanzado"
        elif p >= 45 and trl_nivel >= 3:
            return "NIVEL 3 - Definido"
        elif p >= 35:
            return "NIVEL 2+ - Gestionado Básico"
        elif p >= 25:
            return "NIVEL 2 - Gestionado"
        else:
            return "NIVEL 1 - Inicial/Crítico"
    
    def generar_reporte_tecnico_avanzado(self) -> Dict[str, Any]:
        """Genera reporte técnico avanzado con análisis profundo."""
        try:
            ev = self.evaluacion_causal
            trl_info = ev.calcular_indice_madurez_tecnologica()
            matriz_decision = ev.generar_matriz_decision_multicriterio()
            
            # Análisis de riesgos probabilístico
            analisis_riesgos = self._analizar_riesgos_probabilistico()
            
            # Proyección de impacto temporal
            proyeccion_impacto = self._proyectar_impacto_temporal()
            
            # Análisis de costo-efectividad
            analisis_costo_efectividad = self._analizar_costo_efectividad()
            
            return {
                "metadata_avanzada": {
                    "dimension_id": self.dimension.id,
                    "dimension_nombre": self.dimension.nombre,
                    "cluster": self.dimension.cluster,
                    "prioridad_estrategica": self.dimension.prioridad_estrategica,
                    "complejidad_implementacion": self.dimension.complejidad_implementacion,
                    "timestamp_evaluacion": self.timestamp_evaluacion,
                    "version_sistema": "9.0-industrial-frontier",
                    "hash_dimension": self.dimension.teoria_cambio.construir_grafo_causal_avanzado().number_of_nodes(),
                },
                "evaluacion_causal_avanzada": {
                    "puntaje_global_avanzado": ev.puntaje_global_avanzado,
                    "nivel_certidumbre": ev.nivel_certidumbre_avanzado,
                    "recomendacion_estrategica": ev.recomendacion_estrategica_avanzada,
                    "metricas_detalle_avanzadas": {
                        "consistencia_logica": ev.consistencia_logica,
                        "identificabilidad_causal": ev.identificabilidad_causal,
                        "factibilidad_operativa": ev.factibilidad_operativa,
                        "certeza_probabilistica": ev.certeza_probabilistica,
                        "robustez_causal": ev.robustez_causal,
                        "innovacion_metodologica": ev.innovacion_metodologica,
                        "sostenibilidad_resultados": ev.sostenibilidad_resultados,
                        "escalabilidad_territorial": ev.escalabilidad_territorial,
                    },
                    "factores_contextuales": ev.factores_contextuales,
                    "interdependencias_detectadas": ev.interdependencias_detectadas
                },
                "analisis_madurez_tecnologica": trl_info,
                "matriz_decision_multicriterio": matriz_decision,
                "diagnostico_avanzado": {
                    "brechas_criticas": len(self.brechas_identificadas),
                    "brechas_detalle": self.brechas_identificadas[:10],
                    "riesgos_principales": ev.riesgos_implementacion[:8],
                    "analisis_riesgos_probabilistico": analisis_riesgos,
                    "evidencia_disponible_total": sum(len(v) for v in self.evidencia.values()),
                    "evidencia_por_tipo": {k: len(v) for k, v in self.evidencia.items()},
                    "nivel_madurez_avanzado": self.nivel_madurez_avanzado,
                    "supuestos_criticos": ev.supuestos_criticos[:6]
                },
                "proyecciones_impacto": {
                    "proyeccion_temporal": proyeccion_impacto,
                    "impacto_estimado_poblacion": self.proyeccion_impacto.get('poblacion_beneficiada', 0),
                    "impacto_estimado_territorial": self.proyeccion_impacto.get('cobertura_territorial', 0),
                    "sostenibilidad_proyectada": ev.sostenibilidad_resultados
                },
                "analisis_economico": {
                    "costo_efectividad": analisis_costo_efectividad,
                    "analisis_costo_beneficio": self.analisis_costo_beneficio,
                    "retorno_inversion_estimado": self._calcular_roi_estimado(),
                    "eficiencia_asignacion_recursos": self._evaluar_eficiencia_recursos()
                },
                "recomendaciones_priorizadas": self._priorizar_recomendaciones(),
                "plan_implementacion_sugerido": self._generar_plan_implementacion(),
                "indicadores_monitoreo_avanzados": self._generar_indicadores_monitoreo_avanzados(),
                "trazabilidad_evidencia": self._generar_resumen_trazabilidad()
            }
        
        except Exception as e:
            LOGGER.error(f"Error generando reporte técnico avanzado: {e}")
            return {
                "metadata_avanzada": {"error": str(e)},
                "evaluacion_causal_avanzada": {},
                "diagnostico_avanzado": {"error_diagnostico": str(e)}
            }
    
    def _analizar_riesgos_probabilistico(self) -> Dict[str, Any]:
        """Análisis probabilístico de riesgos."""
        try:
            riesgos = self.evaluacion_causal.riesgos_implementacion
            
            # Clasificación de riesgos por tipo
            tipos_riesgo = {
                'financiero': [],
                'temporal': [],
                'institucional': [],
                'técnico': [],
                'político': [],
                'social': []
            }
            
            for riesgo in riesgos:
                riesgo_lower = riesgo.lower()
                if any(term in riesgo_lower for term in ['presupuest', 'financ', 'recurs', 'cost']):
                    tipos_riesgo['financiero'].append(riesgo)
                elif any(term in riesgo_lower for term in ['tiempo', 'plazo', 'cronogram', 'temporal']):
                    tipos_riesgo['temporal'].append(riesgo)
                elif any(term in riesgo_lower for term in ['institucional', 'organizacion', 'capacidad']):
                    tipos_riesgo['institucional'].append(riesgo)
                elif any(term in riesgo_lower for term in ['técnico', 'tecnolog', 'metodolog']):
                    tipos_riesgo['técnico'].append(riesgo)
                elif any(term in riesgo_lower for term in ['polític', 'gobierno', 'normativ']):
                    tipos_riesgo['político'].append(riesgo)
                else:
                    tipos_riesgo['social'].append(riesgo)
            
            # Estimación de probabilidades e impactos
            matriz_riesgo = {}
            for tipo, lista_riesgos in tipos_riesgo.items():
                if lista_riesgos:
                    # Probabilidades basadas en tipo de riesgo
                    prob_base = {
                        'financiero': 0.4, 'temporal': 0.5, 'institucional': 0.3,
                        'técnico': 0.3, 'político': 0.6, 'social': 0.4
                    }
                    
                    # Impactos basados en tipo de riesgo
                    impacto_base = {
                        'financiero': 0.7, 'temporal': 0.5, 'institucional': 0.6,
                        'técnico': 0.4, 'político': 0.8, 'social': 0.6
                    }
                    
                    probabilidad = min(0.9, prob_base[tipo] + len(lista_riesgos) * 0.1)
                    impacto = min(0.9, impacto_base[tipo])
                    
                    matriz_riesgo[tipo] = {
                        'riesgos': lista_riesgos,
                        'probabilidad': probabilidad,
                        'impacto': impacto,
                        'riesgo_agregado': probabilidad * impacto,
                        'clasificacion': 'ALTO' if probabilidad * impacto > 0.6 else 
                                       'MEDIO' if probabilidad * impacto > 0.3 else 'BAJO'
                    }
            
            # Riesgo agregado total
            if matriz_riesgo:
                riesgo_total = 1.0 - np.prod([
                    1.0 - info['riesgo_agregado'] for info in matriz_riesgo.values()
                ])
            else:
                riesgo_total = 0.2
            
            return {
                'matriz_riesgo_por_tipo': matriz_riesgo,
                'riesgo_agregado_total': riesgo_total,
                'tipos_riesgo_prioritarios': sorted(
                    matriz_riesgo.items(),
                    key=lambda x: x[1]['riesgo_agregado'],
                    reverse=True
                )[:3],
                'recomendacion_mitigacion': self._generar_estrategia_mitigacion(matriz_riesgo)
            }
        
        except Exception as e:
            return {'error': f"Error en análisis de riesgos: {e}"}
    
    def _generar_estrategia_mitigacion(self, matriz_riesgo: Dict) -> List[str]:
        """Genera estrategias de mitigación específicas."""
        estrategias = []
        
        for tipo, info in sorted(matriz_riesgo.items(), key=lambda x: x[1]['riesgo_agregado'], reverse=True):
            if info['riesgo_agregado'] > 0.4:
                if tipo == 'financiero':
                    estrategias.append("Establecer fondo de contingencia y diversificar fuentes de financiación")
                elif tipo == 'temporal':
                    estrategias.append("Implementar metodología ágil con hitos de            conceptos_categoria = set(concepto.lower() for concepto in conceptos_clave)
                terminos_categoria = set(termino.lower() for termino in terminos)
                
                if conceptos_categoria.intersection(terminos_categoria):
                    for termino in terminos:
                        if termino.lower() in texto_lower:
                            coincidencias_semanticas += 0.5
            
            # Cálculo de relevancia normalizada
            total_coincidencias = coincidencias_exactas + coincidencias_parciales + coincidencias_semanticas
            relevancia_normalizada = min(1.0, total_coincidencias / len(conceptos_clave))
            
            # Boost por densidad de coincidencias
            densidad_coincidencias = total_coincidencias / max(1, len(texto.split()) / 50)
            boost_densidad = min(0.3, densidad_coincidencias * 0.1)
            
            return min(1.0, relevancia_normalizada + boost_densidad)
        
        except Exception:
            return sum(1 for c in conceptos_clave if c.lower() in texto.lower()) / max(1, len(conceptos_clave))
    
    def _evaluar_calidad_contenido(self, texto: str, metadata: Dict[str, Any]) -> float:
        """Evalúa la calidad del contenido basado en múltiples criterios."""
        try:
            factores_calidad = []
            
            # Factor de longitud (textos ni muy cortos ni muy largos)
            longitud_palabras = metadata['longitud_palabras']
            if 20 <= longitud_palabras <= 200:
                factor_longitud = 1.0
            elif 10 <= longitud_palabras < 20 or 200 < longitud_palabras <= 300:
                factor_longitud = 0.8
            else:
                factor_longitud = 0.6
            factores_calidad.append(factor_longitud)
            
            # Factor de densidad informativa
            densidad_info = (
                metadata['densidad_numerica'] * 0.4 +
                metadata['densidad_fechas'] * 0.3 +
                metadata['densidad_monetaria'] * 0.3
            ) / 10  # Normalización
            factor_densidad = min(1.0, densidad_info + 0.3)  # Base mínima
            factores_calidad.append(factor_densidad)
            
            # Factor de complejidad sintáctica (moderada es mejor)
            complejidad = metadata['complejidad_sintactica']
            if 8 <= complejidad <= 20:
                factor_complejidad = 1.0
            elif 5 <= complejidad < 8 or 20 < complejidad <= 30:
                factor_complejidad = 0.8
            else:
                factor_complejidad = 0.6
            factores_calidad.append(factor_complejidad)
            
            # Factor de estructura (presencia de patrones organizacionales)
            patrones_estructura = [
                r'\b(objetivo|meta|indicador|producto|resultado)\b',
                r'\b(responsable|encargado|coordinador)\b',
                r'\b(plazo|cronograma|tiempo|duracion)\b',
                r'\b(presupuesto|recursos|financiacion)\b'
            ]
            
            estructura_matches = sum(
                len(re.findall(patron, texto.lower())) for patron in patrones_estructura
            )
            factor_estructura = min(1.0, estructura_matches / 5.0 + 0.4)
            factores_calidad.append(factor_estructura)
            
            # Promedio ponderado
            return np.average(factores_calidad, weights=[0.2, 0.3, 0.2, 0.3])
        
        except Exception:
            return 0.6  # Valor por defecto moderado
    
    def _diversificar_resultados(
        self, resultados: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Diversifica resultados para evitar redundancia."""
        if not resultados or len(resultados) <= top_k:
            return resultados
        
        try:
            resultados_diversificados = []
            textos_seleccionados = []
            
            # Seleccionar el mejor resultado inicialmente
            resultados_diversificados.append(resultados[0])
            textos_seleccionados.append(resultados[0]['texto'])
            
            # Seleccionar resultados adicionales basados en diversidad
            for resultado in resultados[1:]:
                if len(resultados_diversificados) >= top_k:
                    break
                
                texto_candidato = resultado['texto']
                
                # Calcular similitud máxima con textos ya seleccionados
                similitud_maxima = 0.0
                for texto_previo in textos_seleccionados:
                    # Similitud básica basada en palabras comunes
                    palabras_candidato = set(texto_candidato.lower().split())
                    palabras_previo = set(texto_previo.lower().split())
                    
                    if palabras_candidato and palabras_previo:
                        similitud = len(palabras_candidato.intersection(palabras_previo)) / \
                                  len(palabras_candidato.union(palabras_previo))
                        similitud_maxima = max(similitud_maxima, similitud)
                
                # Agregar si la similitud no es muy alta (umbral de diversidad)
                if similitud_maxima < 0.7:  # Umbral de diversidad
                    resultados_diversificados.append(resultado)
                    textos_seleccionados.append(texto_candidato)
            
            return resultados_diversificados
        
        except Exception:
            return resultados[:top_k]
    
    def _buscar_evidencia_fallback(
        self, query: str, conceptos_clave: List[str], top_k: int = 5, umbral_certeza: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Búsqueda fallback cuando los embeddings no están disponibles."""
        if not self.textos_originales:
            log_warning_with_text(self.logger, "⚠️ No hay textos para búsqueda fallback")
            return []
        
        try:
            # Usar TF-IDF si está disponible
            if self.tfidf_matrix is not None:
                query_tfidf = self.vectorizer_tfidf.transform([query])
                similitudes_tfidf = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
            else:
                # Fallback a coincidencias de términos
                similitudes_tfidf = []
                query_terms = set(query.lower().split())
                
                for texto in self.textos_originales:
                    texto_terms = set(texto.lower().split())
                    if query_terms and texto_terms:
                        similitud = len(query_terms.intersection(texto_terms)) / \
                                   len(query_terms.union(texto_terms))
                    else:
                        similitud = 0.0
                    similitudes_tfidf.append(similitud)
            
            resultados = []
            for idx, sim_score in enumerate(similitudes_tfidf):
                if idx >= len(self.textos_originales):
                    continue
                
                texto = self.textos_originales[idx]
                
                # Encontrar página original
                pagina = None
                for p, t in self.documentos:
                    if t == texto:
                        pagina = p
                        break
                
                if pagina is None:
                    continue
                
                # Cálculos básicos
                relevancia = sum(1 for c in conceptos_clave if c.lower() in texto.lower()) / max(1, len(conceptos_clave))
                densidad_causal = self._calcular_densidad_causal_avanzada(texto)['densidad_causal_agregada']
                
                score_final = sim_score * 0.6 + relevancia * 0.3 + densidad_causal * 0.1
                
                if score_final >= umbral_certeza:
                    resultado = {
                        'texto': texto,
                        'pagina': pagina,
                        'similitud_semantica': float(sim_score),
                        'relevancia_conceptual': relevancia,
                        'densidad_causal_agregada': densidad_causal,
                        'score_final': score_final,
                        'tipo_contenido': 'desconocido',
                        'hash_segmento': hashlib.md5(texto.encode()).hexdigest()[:8],
                        'timestamp_extraccion': datetime.now().isoformat(),
                        'confianza_global': min(1.0, score_final * 1.1),
                        'metodo_busqueda': 'fallback_tfidf' if self.tfidf_matrix is not None else 'fallback_terminos'
                    }
                    resultados.append(resultado)
            
            return sorted(resultados, key=lambda x: x['score_final'], reverse=True)[:top_k]
        
        except Exception as e:
            log_error_with_text(self.logger, f"❌ Error en búsqueda fallback: {e}")
            return []
    
    def buscar_patrones_ontologicos_avanzados(
        self, texto: str, categorias: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Búsqueda avanzada de patrones usando la ontología."""
        if categorias is None:
            categorias = list(self.ontologia.patrones_linguisticos.keys())
        
        resultados_categorias = {}
        
        for categoria in categorias:
            if categoria in self.ontologia.patrones_linguisticos:
                patrones_encontrados = self.ontologia.buscar_patrones_avanzados(texto, categoria)
                
                # Enriquecer con análisis contextual
                for patron in patrones_encontrados:
                    # Análisis de contexto expandido
                    inicio_contexto = max(0, patron['posicion_inicio'] - 100)
                    fin_contexto = min(len(texto), patron['posicion_fin'] + 100)
                    contexto_expandido = texto[inicio_contexto:fin_contexto]
                    
                    # Análisis de densidad en el contexto
                    densidad_contextual = self._calcular_densidad_causal_avanzada(contexto_expandido)
                    
                    # Enriquecer resultado
                    patron.update({
                        'contexto_expandido': contexto_expandido,
                        'densidad_causal_contexto': densidad_contextual['densidad_causal_agregada'],
                        'calidad_contexto': self._evaluar_calidad_contenido(
                            contexto_expandido, 
                            self._extraer_caracteristicas_texto(contexto_expandido)
                        )
                    })
                
                resultados_categorias[categoria] = patrones_encontrados
        
        return resultados_categorias
    
    def extraer_variables_operativas_avanzadas(
        self, dimension: DimensionDecalogoAvanzada
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extracción avanzada de variables operativas con análisis profundo."""
        variables_avanzadas = {
            'indicadores': [], 'metas': [], 'recursos': [], 'responsables': [],
            'plazos': [], 'riesgos': [], 'stakeholders': [], 'procesos': []
        }
        
        try:
            # Extracción por eslabones con análisis específico
            for eslabon in dimension.eslabones:
                # Búsqueda de indicadores específicos del eslabón
                for indicador in eslabon.indicadores:
                    query_indicador = (
                        f"indicador {indicador} meta objetivo línea base "
                        f"{dimension.nombre} {eslabon.tipo.value}"
                    )
                    
                    resultados_indicador = self.buscar_evidencia_causal_avanzada(
                        query_indicador,
                        [indicador, "meta", "objetivo", "línea base", "indicador", "medición"],
                        top_k=5,
                        umbral_certeza=0.65,
                        filtros_tipo_contenido=['indicadores_metricas', 'presupuestal_financiero'],
                        pesos_criterios={
                            'similitud_semantica': 0.35,
                            'relevancia_conceptual': 0.35,
                            'densidad_causal': 0.15,
                            'calidad_contenido': 0.15
                        }
                    )
                    
                    for resultado in resultados_indicador:
                        resultado.update({
                            'eslabon_origen': eslabon.id,
                            'tipo_eslabon': eslabon.tipo.value,
                            'indicador_especifico': indicador,
                            'tipo_variable': 'indicador'
                        })
                        variables_avanzadas['indicadores'].append(resultado)
            
            # Búsqueda de recursos con análisis financiero específico
            query_recursos = (
                f"presupuesto financiación recursos inversión "
                f"para {dimension.nombre} asignación fondos"
            )
            
            recursos_encontrados = self.buscar_evidencia_causal_avanzada(
                query_recursos,
                ["presupuesto", "financiación", "recursos", "inversión", "asignación", 
                 "fondos", "subsidio", "transferencia", "cofinanciación", "contrapartida"],
                top_k=8,
                umbral_certeza=0.6,
                filtros_tipo_contenido=['presupuestal_financiero'],
                pesos_criterios={
                    'similitud_semantica': 0.3,
                    'relevancia_conceptual': 0.4,
                    'densidad_causal': 0.1,
                    'calidad_contenido': 0.2
                }
            )
            
            for resultado in recursos_encontrados:
                # Análisis específico de montos
                texto = resultado['texto']
                montos_detectados = re.findall(
                    r'\$\s*[\d,.]+(?: millones?| mil(?:es)?| billones?)?|COP\s*[\d,.]+',
                    texto
                )
                
                resultado.update({
                    'tipo_variable': 'recurso',
                    'montos_detectados': montos_detectados,
                    'tiene_cuantificacion': len(montos_detectados) > 0
                })
                variables_avanzadas['recursos'].append(resultado)
            
            # Búsqueda de responsables con análisis institucional
            query_responsables = (
                f"responsable encargado coordinador lidera gestiona "
                f"{dimension.nombre} institución entidad"
            )
            
            responsables_encontrados = self.buscar_evidencia_causal_avanzadas(
                query_responsables,
                ["responsable", "encargado", "lidera", "coordina", "gestiona", 
                 "ejecuta", "implementa", "secretaría", "ministerio", "entidad"],
                top_k=6,
                umbral_certeza=0.55
            )
            
            for resultado in responsables_encontrados:
                # Detección de entidades específicas
                texto = resultado['texto']
                entidades_detectadas = re.findall(
                    r'\b(secretaría|ministerio|departamento|alcaldía|gobernación|entidad|instituto|agencia)\b[^.]*',
                    texto.lower()
                )
                
                resultado.update({
                    'tipo_variable': 'responsable',
                    'entidades_detectadas': entidades_detectadas,
                    'nivel_institucional': self._clasificar_nivel_institucional(entidades_detectadas)
                })
                variables_avanzadas['responsables'].append(resultado)
            
            # Búsqueda de plazos con análisis temporal
            query_plazos = (
                f"plazo cronograma calendario tiempo duración "
                f"{dimension.nombre} implementación ejecución"
            )
            
            plazos_encontrados = self.buscar_evidencia_causal_avanzada(
                query_plazos,
                ["plazo", "cronograma", "calendario", "tiempo", "duración", 
                 "inicio", "finalización", "trimestre", "semestre", "año"],
                top_k=6,
                umbral_certeza=0.55,
                filtros_tipo_contenido=['cronogramas_plazos']
            )
            
            for resultado in plazos_encontrados:
                # Detección de fechas y períodos específicos
                texto = resultado['texto']
                fechas_detectadas = re.findall(
                    r'\b(20\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4})\b',
                    texto
                )
                periodos_detectados = re.findall(
                    r'\b(trimestre|semestre|año|meses?|días?)\b',
                    texto.lower()
                )
                
                resultado.update({
                    'tipo_variable': 'plazo',
                    'fechas_detectadas': fechas_detectadas,
                    'periodos_detectados': periodos_detectados,
                    'temporalidad_explicita': len(fechas_detectadas) > 0 or len(periodos_detectados) > 0
                })
                variables_avanzadas['plazos'].append(resultado)
            
            # Análisis de riesgos específicos por dimensión
            for eslabon in dimension.eslabones:
                if eslabon.riesgos_especificos:
                    for riesgo in eslabon.riesgos_especificos:
                        query_riesgo = f"riesgo {riesgo} {dimension.nombre} mitigación"
                        
                        riesgos_encontrados = self.buscar_evidencia_causal_avanzada(
                            query_riesgo,
                            ["riesgo", "amenaza", "vulnerabilidad", "contingencia", "mitigación"],
                            top_k=3,
                            umbral_certeza=0.5
                        )
                        
                        for resultado in riesgos_encontrados:
                            resultado.update({
                                'tipo_variable': 'riesgo',
                                'riesgo_especifico': riesgo,
                                'eslabon_asociado': eslabon.id
                            })
                            variables_avanzadas['riesgos'].append(resultado)
            
            # Búsqueda de stakeholders
            query_stakeholders = (
                f"actor actores participante participantes {dimension.nombre} "
                f"comunidad sociedad civil sector privado"
            )
            
            stakeholders_encontrados = self.buscar_evidencia_causal_avanzada(
                query_stakeholders,
                ["actor", "actores", "participante", "comunidad", "organización",
                 "sector privado", "sociedad civil", "beneficiario"],
                top_k=5,
                umbral_certeza=0.5
            )
            
            for resultado in stakeholders_encontrados:
                resultado.update({
                    'tipo_variable': 'stakeholder'
                })
                variables_avanzadas['stakeholders'].append(resultado)
            
            # Log de resultados
            total_variables = sum(len(vars_tipo) for vars_tipo in variables_avanzadas.values())
            log_info_with_text(
                self.logger,
                f"✅ Extracción avanzada dimensión {dimension.id}: {total_variables} variables"
            )
            
            return variables_avanzadas
        
        except Exception as e:
            log_error_with_text(
                self.logger,
                f"❌ Error en extracción avanzada (dim {dimension.id}): {e}"
            )
            return variables_avanzadas  # Retornar lo que se haya podido extraer
    
    def _clasificar_nivel_institucional(self, entidades: List[str]) -> str:
        """Clasifica el nivel institucional basado en las entidades detectadas."""
        if not entidades:
            return 'indeterminado'
        
        entidades_texto = ' '.join(entidades).lower()
        
        if any(term in entidades_texto for term in ['ministerio', 'nacional', 'república']):
            return 'nacional'
        elif any(term in entidades_texto for term in ['departamento', 'gobernación', 'departamental']):
            return 'departamental'
        elif any(term in entidades_texto for term in ['alcaldía', 'municipal', 'municipio', 'secretaría']):
            return 'municipal'
        else:
            return 'sectorial'
    
    def generar_matriz_trazabilidad_avanzada(
        self, dimension: DimensionDecalogoAvanzada
    ) -> pd.DataFrame:
        """Genera matriz de trazabilidad avanzada con análisis profundo."""
        try:
            variables_avanzadas = self.extraer_variables_operativas_avanzadas(dimension)
            datos_matriz = []
            
            for tipo_variable, resultados in variables_avanzadas.items():
                for resultado in resultados:
                    # Análisis de confiabilidad
                    confiabilidad = self._calcular_confiabilidad_evidencia(resultado)
                    
                    # Análisis de completitud
                    completitud = self._evaluar_completitud_evidencia(resultado, tipo_variable)
                    
                    # Análisis de coherencia temporal
                    coherencia_temporal = self._evaluar_coherencia_temporal_evidencia(resultado)
                    
                    fila_datos = {
                        'dimension_id': dimension.id,
                        'dimension_nombre': dimension.nombre,
                        'cluster': dimension.cluster,
                        'tipo_variable': tipo_variable,
                        'subtipo_variable': resultado.get('subtipo_variable', 'general'),
                        'texto_evidencia': (resultado.get('texto', '')[:300] + '...') if len(resultado.get('texto', '')) > 300 else resultado.get('texto', ''),
                        'pagina': resultado.get('pagina', 0),
                        'score_confianza': resultado.get('score_final', 0.0),
                        'confiabilidad_evidencia': confiabilidad,
                        'completitud_evidencia': completitud,
                        'coherencia_temporal': coherencia_temporal,
                        'tipo_contenido': resultado.get('tipo_contenido', 'desconocido'),
                        'densidad_causal': resultado.get('densidad_causal_agregada', 0.0),
                        'calidad_contenido': resultado.get('calidad_contenido', 0.0),
                        'sentimiento_predominante': self._determinar_sentimiento_predominante(
                            resultado.get('analisis_sentimientos', {})
                        ),
                        'eslabon_asociado': resultado.get('eslabon_origen', ''),
                        'hash_evidencia': resultado.get('hash_segmento', 'N/A'),
                        'timestamp_extraccion': resultado.get('timestamp_extraccion', ''),
                        'tiene_cuantificacion': resultado.get('tiene_cuantificacion', False),
                        'nivel_institucional': resultado.get('nivel_institucional', 'indeterminado'),
                        'temporalidad_explicita': resultado.get('temporalidad_explicita', False),
                        'metodo_extraccion': resultado.get('metodo_busqueda', 'embeddings_avanzado'),
                        'puntaje_innovacion': self._calcular_puntaje_innovacion(resultado)
                    }
                    
                    datos_matriz.append(fila_datos)
            
            # Crear DataFrame con análisis estadístico
            if datos_matriz:
                df_matriz = pd.DataFrame(datos_matriz)
                
                # Agregar estadísticas agregadas
                df_matriz['percentil_confianza'] = df_matriz['score_confianza'].rank(pct=True)
                df_matriz['zscore_calidad'] = (
                    (df_matriz['calidad_contenido'] - df_matriz['calidad_contenido'].mean()) /
                    df_matriz['calidad_contenido'].std()
                )
                
                return df_matriz
            else:
                # DataFrame vacío con columnas predefinidas
                columnas_vacias = [
                    'dimension_id', 'dimension_nombre', 'cluster', 'tipo_variable',
                    'texto_evidencia', 'pagina', 'score_confianza', 'confiabilidad_evidencia',
                    'hash_evidencia', 'timestamp_extraccion'
                ]
                return pd.DataFrame(columns=columnas_vacias)
        
        except Exception as e:
            log_error_with_text(
                self.logger,
                f"❌ Error generando matriz de trazabilidad avanzada: {e}"
            )
            return pd.DataFrame()
    
    def _calcular_confiabilidad_evidencia(self, evidencia: Dict[str, Any]) -> float:
        """Calcula confiabilidad de la evidencia basada en múltiples factores."""
        try:
            factores_confiabilidad = []
            
            # Factor de score base
            score_base = evidencia.get('score_final', 0.5)
            factores_confiabilidad.append(score_base)
            
            # Factor de calidad de contenido
            calidad = evidencia.get('calidad_contenido', 0.5)
            factores_confiabilidad.append(calidad)
            
            # Factor de densidad causal
            densidad = evidencia.get('densidad_causal_agregada', 0.0)
            factor_causal = min(1.0, densidad * 2)  # Boost para evidencia causal
            factores_confiabilidad.append(factor_causal)
            
            # Factor de cuantificación (si aplica)
            if evidencia.get('tiene_cuantificacion', False):
                factores_confiabilidad.append(0.8)
            else:
                factores_confiabilidad.append(0.4)
            
            # Factor de temporalidad explícita (si aplica)
            if evidencia.get('temporalidad_explicita', False):
                factores_confiabilidad.append(0.8)
            else:
                factores_confiabilidad.append(0.5)
            
            # Promedio ponderado
            pesos = [0.3, 0.25, 0.2, 0.15, 0.1]
            return np.average(factores_confiabilidad, weights=pesos)
        
        except Exception:
            return evidencia.get('score_final', 0.5)
    
    def _evaluar_completitud_evidencia(self, evidencia: Dict[str, Any], tipo_variable: str) -> float:
        """Evalúa la completitud de la evidencia según el tipo de variable."""
        try:
            elementos_esperados = {
                'indicadores': ['cuantificacion', 'temporalidad', 'responsable'],
                'recursos': ['cuantificacion', 'fuente', 'temporalidad'],
                'responsables': ['nivel_institucional', 'funciones'],
                'plazos': ['temporalidad', 'hitos'],
                'riesgos': ['mitigacion', 'probabilidad'],
                'stakeholders': ['rol', 'nivel_participacion']
            }
            
            elementos_tipo = elementos_esperados.get(tipo_variable, ['informacion_basica'])
            elementos_encontrados = 0
            
            for elemento in elementos_tipo:
                if elemento == 'cuantificacion' and evidencia.get('tiene_cuantificacion', False):
                    elementos_encontrados += 1
                elif elemento == 'temporalidad' and evidencia.get('temporalidad_explicita', False):
                    elementos_encontrados += 1
                elif elemento == 'nivel_institucional' and evidencia.get('nivel_institucional', '') != 'indeterminado':
                    elementos_encontrados += 1
                elif elemento in ['responsable', 'fuente', 'funciones', 'mitigacion', 'probabilidad', 'rol', 'nivel_participacion']:
                    # Análisis textual básico para detectar estos elementos
                    texto = evidencia.get('texto', '').lower()
                    if elemento == 'responsable' and any(term in texto for term in ['responsable', 'encargado', 'lidera']):
                        elementos_encontrados += 1
                    elif elemento == 'fuente' and any(term in texto for term in ['fuente', 'financiación', 'presupuesto']):
                        elementos_encontrados += 1
                    # Otros elementos se pueden agregar según necesidades
            
            return elementos_encontrados / len(elementos_tipo) if elementos_tipo else 1.0
        
        except Exception:
            return 0.5
    
    def _evaluar_coherencia_temporal_evidencia(self, evidencia: Dict[str, Any]) -> float:
        """Evalúa la coherencia temporal de la evidencia."""
        try:
            # Análisis básico de coherencia temporal
            fechas = evidencia.get('fechas_detectadas', [])
            periodos = evidencia.get('periodos_detectados', [])
            
            # Factor base
            coherencia_base = 0.5
            
            # Boost por presencia de fechas
            if fechas:
                coherencia_base += 0.3
            
            # Boost por presencia de períodos
            if periodos:
                coherencia_base += 0.2
            
            # Penalización por inconsistencias (fechas muy dispersas)
            if len(fechas) > 3:
                # Análisis de dispersión temporal básico
                try:
                    años_detectados = [int(f) for f in fechas if f.isdigit() and 2020 <= int(f) <= 2030]
                    if años_detectados:
                        dispersión = max(años_detectados) - min(años_detectados)
                        if dispersión > 5:  # Más de 5 años de dispersión
                            coherencia_base -= 0.1
                except:
                    pass
            
            return min(1.0, coherencia_base)
        
        except Exception:
            return 0.5
    
    def _determinar_sentimiento_predominante(self, sentimientos: Dict[str, float]) -> str:
        """Determina el sentimiento predominante del análisis."""
        if not sentimientos:
            return 'neutral'
        
        try:
            max_sentimiento = max(sentimientos.items(), key=lambda x: x[1])
            return max_sentimiento[0].replace('sentimiento_', '')
        except:
            return            ]
        },
        "interconexiones": {"1-5": 0.8, "1-8": 0.7, "5-8": 0.9},
        "complejidad_agregada": 2.1,
        "prioridad_politica": 2.8
    },
    "CLUSTER_2": {
        "titulo": "CLUSTER 2: DERECHOS DE GRUPOS POBLACIONALES Y EQUIDAD INTEGRAL",
        "descripcion_extendida": """
        Este cluster se centra en la garantía de derechos de grupos poblacionales que enfrentan
        discriminación estructural y vulnerabilidades específicas. Incluye enfoque diferencial,
        equidad de género y participación democrática como elementos transversales para la
        construcción de sociedades más justas e incluyentes.
        """,
        "puntos": [2, 6, 9],
        "logica": """
        La agrupación responde a la necesidad de abordar de manera integral las múltiples
        discriminaciones que enfrentan diversos grupos poblacionales, reconociendo las
        interseccionalidades y promoviendo la participación efectiva en la vida democrática.
        """,
        "teoria_cambio_cluster": {
            "hipotesis_principal": "La garantía efectiva de derechos diferenciales fortalece la participación democrática y la equidad social",
            "supuestos_criticos": [
                "Existe reconocimiento institucional de las diferencias y necesidades específicas",
                "Los grupos poblacionales tienen capacidad organizativa",
                "Los mecanismos de participación son efectivos y accesibles"
            ]
        },
        "interconexiones": {"2-6": 0.85, "2-9": 0.75, "6-9": 0.8},
        "complejidad_agregada": 2.3,
        "prioridad_politica": 2.5
    },
    "CLUSTER_3": {
        "titulo": "CLUSTER 3: TERRITORIO SOSTENIBLE Y DESARROLLO RURAL INTEGRAL",
        "descripcion_extendida": """
        Este cluster integra la gestión sostenible del territorio con el desarrollo rural
        y la soberanía alimentaria. Se enfoca en la armonización entre conservación ambiental,
        productividad agropecuaria sostenible y bienestar de las comunidades rurales,
        promoviendo modelos de desarrollo que respeten los límites planetarios.
        """,
        "puntos": [3, 7],
        "logica": """
        La lógica de agrupación reconoce la interdependencia entre sostenibilidad ambiental
        y desarrollo rural, entendiendo que la prosperidad rural debe basarse en modelos
        productivos que conserven y regeneren los ecosistemas territoriales.
        """,
        "teoria_cambio_cluster": {
            "hipotesis_principal": "El desarrollo rural sostenible es condición necesaria para la conservación ambiental y la seguridad alimentaria",
            "supuestos_criticos": [
                "Las comunidades rurales adoptan prácticas sostenibles",
                "Existe mercados para productos sostenibles",
                "Las políticas públicas incentivan la sostenibilidad"
            ]
        },
        "interconexiones": {"3-7": 0.9},
        "complejidad_agregada": 1.8,
        "prioridad_politica": 2.4
    },
    "CLUSTER_4": {
        "titulo": "CLUSTER 4: DERECHOS SOCIALES FUNDAMENTALES Y GESTIÓN HUMANITARIA",
        "descripcion_extendida": """
        Este cluster aborda la garantía de derechos sociales fundamentales (salud, educación,
        vivienda, trabajo decente) y la capacidad de respuesta ante crisis humanitarias.
        Integra la perspectiva de desarrollo humano con la gestión del riesgo y la preparación
        para emergencias, reconociendo que la vulnerabilidad social amplifica los impactos
        de las crisis.
        """,
        "puntos": [4, 10],
        "logica": """
        La agrupación se fundamenta en el reconocimiento de que los derechos sociales
        fundamentales constituyen la base para la resiliencia comunitaria ante crisis
        humanitarias, y que la gestión efectiva del riesgo protege las conquistas
        en desarrollo social.
        """,
        "teoria_cambio_cluster": {
            "hipotesis_principal": "La garantía de derechos sociales fundamentales fortalece la resiliencia ante crisis humanitarias",
            "supuestos_criticos": [
                "Los sistemas de protección social funcionan efectivamente",
                "Existe capacidad institucional para respuesta a emergencias",
                "Las comunidades participan en la gestión del riesgo"
            ]
        },
        "interconexiones": {"4-10": 0.7},
        "complejidad_agregada": 2.0,
        "prioridad_politica": 2.9
    }
}

# Cache del contexto avanzado
_DECALOGO_CONTEXTO_AVANZADO_CACHE: Optional[DecalogoContextoAvanzado] = None

def obtener_decalogo_contexto_avanzado() -> DecalogoContextoAvanzado:
    """Obtiene el contexto avanzado del decálogo con análisis completo."""
    global _DECALOGO_CONTEXTO_AVANZADO_CACHE
    
    if _DECALOGO_CONTEXTO_AVANZADO_CACHE is not None:
        return _DECALOGO_CONTEXTO_AVANZADO_CACHE
    
    try:
        # Construcción de dimensiones por ID
        dimensiones_por_id = {d.id: d for d in DECALOGO_INDUSTRIAL_AVANZADO}
        
        # Construcción de clusters avanzados
        clusters_por_id: Dict[str, ClusterMetadataAvanzada] = {}
        cluster_por_dimension: Dict[int, ClusterMetadataAvanzada] = {}
        
        for cluster_id, data in _CLUSTER_DEFINITIONS_AVANZADAS.items():
            metadata = ClusterMetadataAvanzada(
                cluster_id=cluster_id,
                titulo=data["titulo"],
                descripcion_extendida=data["descripcion_extendida"],
                puntos=data["puntos"],
                logica_agrupacion=data["logica"],
                teoria_cambio_cluster=data["teoria_cambio_cluster"],
                interconexiones=data["interconexiones"],
                complejidad_agregada=data["complejidad_agregada"],
                prioridad_politica=data["prioridad_politica"]
            )
            
            clusters_por_id[cluster_id] = metadata
            
            # Mapeo dimensión -> cluster
            for punto_id in data["puntos"]:
                cluster_por_dimension[punto_id] = metadata
        
        # Construcción de matriz de interdependencias
        n_dims = len(dimensiones_por_id)
        matriz_interdependencias = np.zeros((n_dims, n_dims))
        
        for i, dim in dimensiones_por_id.items():
            for j in dim.interdependencias:
                if 1 <= j <= n_dims:
                    matriz_interdependencias[i-1, j-1] = 1.0
                    matriz_interdependencias[j-1, i-1] = 0.7  # Relación asimétrica
        
        # Carga de ontología avanzada
        ontologia_avanzada = OntologiaPoliticasAvanzada.cargar_ontologia_avanzada()
        
        # Construcción del contexto
        contexto = DecalogoContextoAvanzado(
            dimensiones_por_id=dimensiones_por_id,
            clusters_por_id=clusters_por_id,
            cluster_por_dimension=cluster_por_dimension,
            matriz_interdependencias=matriz_interdependencias,
            ontologia=ontologia_avanzada
        )
        
        # Cálculo de interdependencias avanzadas
        _ = contexto.calcular_interdependencias_avanzadas()
        
        _DECALOGO_CONTEXTO_AVANZADO_CACHE = contexto
        LOGGER.info("✅ Contexto avanzado del decálogo construido exitosamente")
        
        return contexto
    
    except Exception as e:
        LOGGER.error(f"❌ Error construyendo contexto avanzado: {e}")
        raise SystemExit("Fallo en construcción de contexto avanzado del decálogo")

# -------------------- Extractor de evidencia con capacidades de frontera --------------------
class ExtractorEvidenciaIndustrialAvanzado:
    """Extractor de evidencia con capacidades matemáticas y de IA de frontera."""
    
    def __init__(self, documentos: List[Tuple[int, str]], nombre_plan: str = "desconocido"):
        self.documentos = documentos
        self.nombre_plan = nombre_plan
        self.ontologia = OntologiaPoliticasAvanzada.cargar_ontologia_avanzada()
        self.embeddings_doc: Optional[torch.Tensor] = None
        self.embeddings_metadata: List[Dict[str, Any]] = []
        self.textos_originales = [doc[1] for doc in documentos]
        self.vectorizer_tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.logger = get_truncation_logger(f"ExtractorAvanzado_{nombre_plan}")
        
        # Configuración avanzada
        self.device_config = get_device_config()
        self.batch_size = self.device_config.get_batch_size()
        self.precision = self.device_config.get_precision()
        
        # Análisis de sentimientos si está disponible
        self.sentiment_analyzer = ADVANCED_NLP_PIPELINE
        
        # Inicialización avanzada
        self._inicializar_capacidades_avanzadas()
    
    def _inicializar_capacidades_avanzadas(self):
        """Inicialización de capacidades avanzadas de procesamiento."""
        try:
            # Precomputar embeddings con metadatos
            self._precomputar_embeddings_avanzados()
            
            # Precomputar matriz TF-IDF
            self._precomputar_tfidf()
            
            # Análisis de estructura documental
            self._analizar_estructura_documental()
            
            log_info_with_text(
                self.logger, 
                f"✅ Capacidades avanzadas inicializadas - {self.nombre_plan}"
            )
        
        except Exception as e:
            log_error_with_text(
                self.logger, 
                f"❌ Error inicializando capacidades avanzadas: {e}"
            )
    
    def _precomputar_embeddings_avanzados(self):
        """Precomputación de embeddings con metadatos enriquecidos."""
        textos_validos = []
        metadata_valida = []
        
        for i, (pagina, texto) in enumerate(self.documentos):
            if len(texto.strip()) > 20:  # Umbral más exigente
                textos_validos.append(texto)
                
                # Análisis de características del texto
                caracteristicas = self._extraer_caracteristicas_texto(texto)
                
                metadata = {
                    'pagina': pagina,
                    'indice_original': i,
                    'longitud_caracteres': len(texto),
                    'longitud_palabras': len(texto.split()),
                    'densidad_numerica': caracteristicas['densidad_numerica'],
                    'densidad_fechas': caracteristicas['densidad_fechas'],
                    'densidad_monetaria': caracteristicas['densidad_monetaria'],
                    'complejidad_sintactica': caracteristicas['complejidad_sintactica'],
                    'tipo_contenido_estimado': caracteristicas['tipo_contenido'],
                    'hash_contenido': hashlib.md5(texto.encode()).hexdigest()[:12]
                }
                metadata_valida.append(metadata)
        
        if textos_validos:
            try:
                # Embeddings en lotes para optimización
                embeddings_lotes = []
                
                for i in range(0, len(textos_validos), self.batch_size):
                    lote = textos_validos[i:i + self.batch_size]
                    embeddings_lote = EMBEDDING_MODEL.encode(
                        lote,
                        convert_to_tensor=True,
                        batch_size=min(self.batch_size, len(lote)),
                        precision=self.precision
                    )
                    embeddings_lotes.append(embeddings_lote)
                
                # Concatenar embeddings
                self.embeddings_doc = torch.cat(embeddings_lotes, dim=0)
                self.embeddings_metadata = metadata_valida
                
                log_info_with_text(
                    self.logger,
                    f"✅ Embeddings avanzados precomputados: {len(textos_validos)} segmentos"
                )
            
            except Exception as e:
                log_error_with_text(self.logger, f"❌ Error precomputando embeddings: {e}")
                self.embeddings_doc = torch.tensor([])
        else:
            self.embeddings_doc = torch.tensor([])
            log_warning_with_text(
                self.logger,
                "⚠️ Textos insuficientes para embeddings avanzados"
            )
    
    def _extraer_caracteristicas_texto(self, texto: str) -> Dict[str, Any]:
        """Extrae características avanzadas del texto."""
        try:
            # Análisis numérico
            numeros = re.findall(r'\d+(?:[.,]\d+)*', texto)
            densidad_numerica = len(numeros) / max(1, len(texto.split())) * 100
            
            # Análisis de fechas
            fechas = re.findall(r'\b(20\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4})\b', texto)
            densidad_fechas = len(fechas) / max(1, len(texto.split())) * 100
            
            # Análisis monetario
            montos = re.findall(r'\$[\d,.]+(?: millones?| mil(?:es)?| billones?)?|COP\s*[\d,.]+', texto)
            densidad_monetaria = len(montos) / max(1, len(texto.split())) * 100
            
            # Complejidad sintáctica (aproximada)
            oraciones = re.split(r'[.!?]+', texto)
            palabras_por_oracion = [len(oracion.split()) for oracion in oraciones if oracion.strip()]
            complejidad_sintactica = np.mean(palabras_por_oracion) if palabras_por_oracion else 0
            
            # Clasificación de tipo de contenido
            tipo_contenido = self._clasificar_tipo_contenido(
                densidad_numerica, densidad_fechas, densidad_monetaria, complejidad_sintactica
            )
            
            return {
                'densidad_numerica': densidad_numerica,
                'densidad_fechas': densidad_fechas,
                'densidad_monetaria': densidad_monetaria,
                'complejidad_sintactica': complejidad_sintactica,
                'tipo_contenido': tipo_contenido
            }
        
        except Exception:
            return {
                'densidad_numerica': 0.0,
                'densidad_fechas': 0.0,
                'densidad_monetaria': 0.0,
                'complejidad_sintactica': 10.0,
                'tipo_contenido': 'general'
            }
    
    def _clasificar_tipo_contenido(
        self, dens_num: float, dens_fecha: float, dens_mon: float, complejidad: float
    ) -> str:
        """Clasifica el tipo de contenido basado en características."""
        if dens_mon > 2.0 or dens_num > 5.0:
            return 'presupuestal_financiero'
        elif dens_fecha > 2.0:
            return 'cronogramas_plazos'
        elif complejidad > 20:
            return 'normativo_legal'
        elif dens_num > 1.0:
            return 'indicadores_metricas'
        else:
            return 'narrativo_descriptivo'
    
    def _precomputar_tfidf(self):
        """Precomputa matriz TF-IDF para análisis lexical."""
        try:
            if self.textos_originales:
                self.tfidf_matrix = self.vectorizer_tfidf.fit_transform(self.textos_originales)
                log_info_with_text(
                    self.logger,
                    f"✅ Matriz TF-IDF precomputada: {self.tfidf_matrix.shape}"
                )
        except Exception as e:
            log_error_with_text(self.logger, f"❌ Error precomputando TF-IDF: {e}")
            self.tfidf_matrix = None
    
    def _analizar_estructura_documental(self):
        """Analiza la estructura general del documento."""
        try:
            # Análisis de distribución de tipos de contenido
            tipos_contenido = [
                meta['tipo_contenido_estimado'] 
                for meta in self.embeddings_metadata
            ]
            
            from collections import Counter
            distribucion_tipos = Counter(tipos_contenido)
            
            # Análisis de densidad de información por página
            densidades_pagina = defaultdict(list)
            for meta in self.embeddings_metadata:
                pagina = meta['pagina']
                densidades_pagina[pagina].append({
                    'numerica': meta['densidad_numerica'],
                    'fechas': meta['densidad_fechas'],
                    'monetaria': meta['densidad_monetaria']
                })
            
            # Identificar páginas con alta densidad de información clave
            paginas_criticas = []
            for pagina, densidades in densidades_pagina.items():
                densidad_promedio = np.mean([
                    d['numerica'] + d['fechas'] + d['monetaria'] 
                    for d in densidades
                ])
                if densidad_promedio > 5.0:  # Umbral de criticidad
                    paginas_criticas.append(pagina)
            
            self.estructura_documental = {
                'distribucion_tipos_contenido': dict(distribucion_tipos),
                'paginas_alta_densidad_info': paginas_criticas,
                'total_segmentos': len(self.embeddings_metadata),
                'promedio_palabras_segmento': np.mean([
                    meta['longitud_palabras'] for meta in self.embeddings_metadata
                ]) if self.embeddings_metadata else 0
            }
            
        except Exception as e:
            log_warning_with_text(self.logger, f"⚠️ Error analizando estructura: {e}")
            self.estructura_documental = {}
    
    def _calcular_densidad_causal_avanzada(self, texto: str) -> Dict[str, float]:
        """Cálculo avanzado de densidad causal con múltiples indicadores."""
        try:
            # Patrones causales básicos
            patrones_causales = [
                r'\b(porque|debido a|como consecuencia de|en razón de|a causa de|por efecto de)\b',
                r'\b(genera|produce|causa|determina|influye en|afecta a|resulta en|conlleva)\b',
                r'\b(impacto|efecto|resultado|consecuencia|repercusión|derivación)\b',
                r'\b(mejora|aumenta|incrementa|reduce|disminuye|fortalece|debilita|optimiza)\b',
                r'\b(siempre que|cuando|si|en caso de que)\b.*\b(entonces|por lo tanto|en consecuencia|se logra)\b'
            ]
            
            # Patrones de correlación
            patrones_correlacion = [
                r'\b(correlación|asociación|relación|vínculo)\b.*\b(con|entre|hacia)\b',
                r'\b(mientras|a medida que|conforme|en tanto que)\b.*\b(más|menos|mayor|menor)\b'
            ]
            
            # Patrones temporales
            patrones_temporales = [
                r'\b(antes|después|durante|posteriormente|previamente|inicialmente|finalmente)\b',
                r'\b(primero|segundo|luego|seguido de|en paralelo|simultáneamente)\b'
            ]
            
            # Contar matches por categoría
            densidad_causal = sum(len(re.findall(p, texto.lower())) for p in patrones_causales)
            densidad_correlacional = sum(len(re.findall(p, texto.lower())) for p in patrones_correlacion)
            densidad_temporal = sum(len(re.findall(p, texto.lower())) for p in patrones_temporales)
            
            # Normalización por longitud del texto
            num_palabras = len(texto.split())
            factor_normalizacion = max(1, num_palabras / 100)
            
            return {
                'densidad_causal': min(1.0, densidad_causal / factor_normalizacion),
                'densidad_correlacional': min(1.0, densidad_correlacional / factor_normalizacion),
                'densidad_temporal': min(1.0, densidad_temporal / factor_normalizacion),
                'densidad_causal_agregada': min(1.0, (
                    densidad_causal * 0.5 + 
                    densidad_correlacional * 0.3 + 
                    densidad_temporal * 0.2
                ) / factor_normalizacion)
            }
        
        except Exception:
            return {
                'densidad_causal': 0.0,
                'densidad_correlacional': 0.0,
                'densidad_temporal': 0.0,
                'densidad_causal_agregada': 0.0
            }
    
    def _analizar_sentimientos_texto(self, texto: str) -> Dict[str, float]:
        """Análisis de sentimientos del texto si está disponible."""
        if not self.sentiment_analyzer:
            return {'sentimiento_positivo': 0.5, 'sentimiento_negativo': 0.5, 'sentimiento_neutral': 0.5}
        
        try:
            # Fragmentar texto si es muy largo
            max_length = 512  # Límite típico de modelos de transformers
            fragmentos = [texto[i:i+max_length] for i in range(0, len(texto), max_length)]
            
            sentimientos_fragmentos = []
            for fragmento in fragmentos[:3]:  # Máximo 3 fragmentos para eficiencia
                if len(fragmento.strip()) > 10:
                    resultado = self.sentiment_analyzer(fragmento)
                    sentimientos_fragmentos.append(resultado)
            
            if sentimientos_fragmentos:
                # Agregar sentimientos de todos los fragmentos
                sentimientos_agregados = defaultdict(float)
                for resultado_fragmento in sentimientos_fragmentos:
                    for item in resultado_fragmento:
                        sentimientos_agregados[item['label']] += item['score']
                
                # Normalizar
                total_fragmentos = len(sentimientos_fragmentos)
                for key in sentimientos_agregados:
                    sentimientos_agregados[key] /= total_fragmentos
                
                return {
                    'sentimiento_positivo': sentimientos_agregados.get('POSITIVE', sentimientos_agregados.get('LABEL_2', 0.5)),
                    'sentimiento_negativo': sentimientos_agregados.get('NEGATIVE', sentimientos_agregados.get('LABEL_0', 0.5)),
                    'sentimiento_neutral': sentimientos_agregados.get('NEUTRAL', sentimientos_agregados.get('LABEL_1', 0.5))
                }
            
        except Exception as e:
            log_debug_with_text(self.logger, f"Error en análisis de sentimientos: {e}")
        
        return {'sentimiento_positivo': 0.5, 'sentimiento_negativo': 0.5, 'sentimiento_neutral': 0.5}
    
    def buscar_evidencia_causal_avanzada(
        self, 
        query: str, 
        conceptos_clave: List[str], 
        top_k: int = 10, 
        umbral_certeza: float = 0.7,
        filtros_tipo_contenido: List[str] = None,
        pesos_criterios: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """Búsqueda avanzada de evidencia causal con múltiples criterios."""
        
        if (self.embeddings_doc is None) or (self.embeddings_doc.numel() == 0):
            log_warning_with_text(self.logger, "⚠️ Embeddings no disponibles. Usando fallback.")
            return self._buscar_evidencia_fallback(query, conceptos_clave, top_k, umbral_certeza)
        
        try:
            # Pesos por defecto
            if pesos_criterios is None:
                pesos_criterios = {
                    'similitud_semantica': 0.4,
                    'relevancia_conceptual': 0.25,
                    'densidad_causal': 0.2,
                    'calidad_contenido': 0.15
                }
            
            # Embedding de la query
            q_emb = EMBEDDING_MODEL.encode(query, convert_to_tensor=True)
            
            # Calcular similitudes semánticas
            similitudes = util.pytorch_cos_sim(q_emb, self.embeddings_doc)[0]
            
            resultados = []
            
            # Procesar cada documento
            for idx, sim_score in enumerate(similitudes):
                if idx >= len(self.textos_originales) or idx >= len(self.embeddings_metadata):
                    continue
                
                texto = self.textos_originales[idx]
                metadata = self.embeddings_metadata[idx]
                
                # Filtrar por tipo de contenido si se especifica
                if (filtros_tipo_contenido and 
                    metadata['tipo_contenido_estimado'] not in filtros_tipo_contenido):
                    continue
                
                # Relevancia conceptual mejorada
                relevancia_conceptual = self._calcular_relevancia_conceptual_avanzada(
                    texto, conceptos_clave
                )
                
                # Densidad causal avanzada
                densidad_causal_info = self._calcular_densidad_causal_avanzada(texto)
                
                # Calidad del contenido
                calidad_contenido = self._evaluar_calidad_contenido(texto, metadata)
                
                # Análisis de sentimientos
                sentimientos = self._analizar_sentimientos_texto(texto)
                
                # Score final ponderado
                score_final = (
                    float(sim_score) * pesos_criterios['similitud_semantica'] +
                    relevancia_conceptual * pesos_criterios['relevancia_conceptual'] +
                    densidad_causal_info['densidad_causal_agregada'] * pesos_criterios['densidad_causal'] +
                    calidad_contenido * pesos_criterios['calidad_contenido']
                )
                
                # Crear resultado enriquecido
                if score_final >= umbral_certeza:
                    resultado = {
                        'texto': texto,
                        'pagina': metadata['pagina'],
                        'similitud_semantica': float(sim_score),
                        'relevancia_conceptual': relevancia_conceptual,
                        'densidad_causal_agregada': densidad_causal_info['densidad_causal_agregada'],
                        'densidad_causal_detalle': densidad_causal_info,
                        'calidad_contenido': calidad_contenido,
                        'score_final': score_final,
                        'tipo_contenido': metadata['tipo_contenido_estimado'],
                        'caracteristicas_texto': {
                            'longitud_palabras': metadata['longitud_palabras'],
                            'densidad_numerica': metadata['densidad_numerica'],
                            'densidad_fechas': metadata['densidad_fechas'],
                            'densidad_monetaria': metadata['densidad_monetaria'],
                            'complejidad_sintactica': metadata['complejidad_sintactica']
                        },
                        'analisis_sentimientos': sentimientos,
                        'hash_segmento': metadata['hash_contenido'],
                        'timestamp_extraccion': datetime.now().isoformat(),
                        'confianza_global': min(1.0, score_final * 1.2)  # Factor de ajuste de confianza
                    }
                    resultados.append(resultado)
            
            # Ordenar por score final y aplicar post-procesamiento
            resultados_ordenados = sorted(resultados, key=lambda x: x['score_final'], reverse=True)
            
            # Diversificación de resultados (evitar resultados muy similares)
            resultados_diversificados = self._diversificar_resultados(resultados_ordenados, top_k)
            
            return resultados_diversificados[:top_k]
        
        except Exception as e:
            log_error_with_text(self.logger, f"❌ Error en búsqueda avanzada: {e}")
            return self._buscar_evidencia_fallback(query, conceptos_clave, top_k, umbral_certeza)
    
    def _calcular_relevancia_conceptual_avanzada(
        self, texto: str, conceptos_clave: List[str]
    ) -> float:
        """Cálculo avanzado de relevancia conceptual."""
        if not conceptos_clave:
            return 0.0
        
        try:
            texto_lower = texto.lower()
            
            # Coincidencias exactas
            coincidencias_exactas = sum(1 for concepto in conceptos_clave if concepto.lower() in texto_lower)
            
            # Coincidencias parciales (stemming básico)
            coincidencias_parciales = 0
            for concepto in conceptos_clave:
                raiz_concepto = concepto[:max(4, len(concepto) - 2)]  # Stemming básico
                if raiz_concepto.lower() in texto_lower:
                    coincidencias_parciales += 0.7  # Peso menor para coincidencias parciales
            
            # Coincidencias semánticas usando ontología
            coincidencias_semanticas = 0
            for categoria, terminos in self.ontologia.vocabulario_especializado.items():
                conceptos_categoria = set(concepto.lower()# -------------------- Sistema de carga del decálogo avanzado --------------------
def cargar_decalogo_industrial_avanzado() -> List[DimensionDecalogoAvanzada]:
    """Carga el decálogo industrial con capacidades avanzadas."""
    json_path = Path("decalogo_industrial_avanzado.json")
    
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not isinstance(data, list) or len(data) != 10:
                raise ValueError("Decálogo debe contener exactamente 10 dimensiones")
            
            decalogos = []
            for i, item in enumerate(data):
                # Validación de estructura
                campos_requeridos = ["id", "nombre", "cluster", "teoria_cambio", "eslabones"]
                if not all(k in item for k in campos_requeridos):
                    raise ValueError(f"Dimensión {i+1} incompleta")
                
                if item["id"] != i + 1:
                    raise ValueError(f"ID incorrecto en dimensión {i+1}")
                
                # Construcción de teoría de cambio avanzada
                tc_data = item["teoria_cambio"]
                teoria_cambio = TeoriaCambioAvanzada(
                    supuestos_causales=tc_data["supuestos_causales"],
                    mediadores=tc_data["mediadores"],
                    resultados_intermedios=tc_data["resultados_intermedios"],
                    precondiciones=tc_data["precondiciones"],
                    moderadores=tc_data.get("moderadores", []),
                    variables_contextuales=tc_data.get("variables_contextuales", []),
                    mecanismos_causales=tc_data.get("mecanismos_causales", [])
                )
                
                # Validación de identificabilidad
                identificabilidad = teoria_cambio.verificar_identificabilidad_avanzada()
                if identificabilidad['puntaje_global_identificabilidad'] < 0.4:
                    raise ValueError(f"Teoría de cambio no suficientemente identificable en dimensión {i+1}")
                
                # Construcción de eslabones avanzados
                eslabones = []
                for j, ed in enumerate(item["eslabones"]):
                    eslabon = EslabonCadenaAvanzado(
                        id=ed["id"],
                        tipo=TipoCadenaValor[ed["tipo"]],
                        indicadores=ed["indicadores"],
                        capacidades_requeridas=ed["capacidades_requeridas"],
                        puntos_criticos=ed["puntos_criticos"],
                        ventana_temporal=tuple(ed["ventana_temporal"]),
                        kpi_ponderacion=float(ed.get("kpi_ponderacion", 1.0)),
                        riesgos_especificos=ed.get("riesgos_especificos", []),
                        dependencias=ed.get("dependencias", []),
                        stakeholders=ed.get("stakeholders", []),
                        recursos_estimados=ed.get("recursos_estimados", {})
                    )
                    eslabones.append(eslabon)
                
                # Construcción de dimensión avanzada
                dimension = DimensionDecalogoAvanzada(
                    id=item["id"],
                    nombre=item["nombre"],
                    cluster=item["cluster"],
                    teoria_cambio=teoria_cambio,
                    eslabones=eslabones,
                    prioridad_estrategica=float(item.get("prioridad_estrategica", 1.0)),
                    complejidad_implementacion=float(item.get("complejidad_implementacion", 0.5)),
                    interdependencias=item.get("interdependencias", []),
                    contexto_territorial=item.get("contexto_territorial", {})
                )
                
                decalogos.append(dimension)
            
            LOGGER.info(f"✅ Decálogo avanzado cargado y validado: {len(decalogos)} dimensiones")
            return decalogos
        
        except Exception as e:
            LOGGER.error(f"❌ Error cargando decálogo avanzado: {e}")
            raise SystemExit("Fallo en carga de decálogo avanzado")
    
    # Generar template avanzado si no existe
    LOGGER.info("⚙️ Generando template avanzado de decálogo estructurado")
    
    template_avanzado = []
    for dim_id in range(1, 11):
        # Nombres más específicos y descriptivos
        nombres_dimensiones = {
            1: "Dimensión 1: Paz Territorial y Seguridad Humana Integral",
            2: "Dimensión 2: Derechos de Grupos Poblacionales Vulnerables",
            3: "Dimensión 3: Territorio Sostenible y Gestión Ambiental",
            4: "Dimensión 4: Derechos Sociales Fundamentales y Servicios Públicos",
            5: "Dimensión 5: Protección de Defensores de Derechos Humanos",
            6: "Dimensión 6: Equidad de Género y Diversidad Sexual",
            7: "Dimensión 7: Desarrollo Rural y Soberanía Alimentaria",
            8: "Dimensión 8: Justicia Transicional y Memoria Histórica",
            9: "Dimensión 9: Participación Ciudadana y Democracia Participativa",
            10: "Dimensión 10: Crisis Humanitarias y Gestión del Riesgo"
        }
        
        # Clusters más específicos
        clusters_avanzados = {
            1: "CLUSTER 1: PAZ, SEGURIDAD Y PROTECCIÓN INTEGRAL",
            2: "CLUSTER 2: DERECHOS DE GRUPOS POBLACIONALES Y EQUIDAD",
            3: "CLUSTER 3: TERRITORIO SOSTENIBLE Y DESARROLLO RURAL",
            4: "CLUSTER 4: DERECHOS SOCIALES Y GESTIÓN HUMANITARIA"
        }
        
        cluster_id = ((dim_id - 1) // 3) + 1 if dim_id <= 9 else 4
        
        dim = {
            "id": dim_id,
            "nombre": nombres_dimensiones.get(dim_id, f"Dimensión {dim_id} del Decálogo Industrial"),
            "cluster": clusters_avanzados.get(cluster_id, f"Cluster {cluster_id}"),
            "prioridad_estrategica": 1.0 + (dim_id % 3) * 0.3,  # Variación en prioridades
            "complejidad_implementacion": 0.4 + (dim_id % 4) * 0.15,  # Variación en complejidad
            "interdependencias": [(dim_id % 10) + 1] if dim_id < 10 else [1],
            "contexto_territorial": {
                "ambito_aplicacion": "municipal",
                "poblacion_objetivo": "general",
                "sector_prioritario": "mixto"
            },
            "teoria_cambio": {
                "supuestos_causales": [
                    f"La implementación efectiva de la dimensión {dim_id} genera cambios sostenibles en el territorio",
                    f"Los actores territoriales tienen capacidad de apropiación de los procesos de la dimensión {dim_id}",
                    f"Existe voluntad política e institucional para sostener las intervenciones de la dimensión {dim_id}"
                ],
                "mediadores": {
                    "institucionales": [
                        f"fortalecimiento_institucional_dim_{dim_id}",
                        f"coordinacion_intersectorial_dim_{dim_id}",
                        f"capacidades_tecnicas_dim_{dim_id}"
                    ],
                    "comunitarios": [
                        f"participacion_comunitaria_dim_{dim_id}",
                        f"empoderamiento_ciudadano_dim_{dim_id}",
                        f"capital_social_dim_{dim_id}"
                    ],
                    "territoriales": [
                        f"articulacion_territorial_dim_{dim_id}",
                        f"identidad_territorial_dim_{dim_id}"
                    ]
                },
                "resultados_intermedios": [
                    f"resultado_intermedio_institucional_dim_{dim_id}",
                    f"resultado_intermedio_social_dim_{dim_id}",
                    f"resultado_intermedio_territorial_dim_{dim_id}"
                ],
                "precondiciones": [
                    f"precondicion_normativa_dim_{dim_id}",
                    f"precondicion_presupuestal_dim_{dim_id}",
                    f"precondicion_tecnica_dim_{dim_id}"
                ],
                "moderadores": [
                    f"contexto_politico_dim_{dim_id}",
                    f"condiciones_economicas_dim_{dim_id}"
                ],
                "variables_contextuales": [
                    f"variable_demografica_dim_{dim_id}",
                    f"variable_geografica_dim_{dim_id}",
                    f"variable_cultural_dim_{dim_id}"
                ],
                "mecanismos_causales": [
                    f"mecanismo_incentivos_dim_{dim_id}",
                    f"mecanismo_capacitacion_dim_{dim_id}",
                    f"mecanismo_coordinacion_dim_{dim_id}"
                ]
            },
            "eslabones": []
        }
        
        # Generar eslabones más sofisticados
        tipos_eslabon = [
            ("INSUMOS", "Recursos financieros, humanos y normativos"),
            ("PROCESOS", "Gestión, coordinación e implementación"),
            ("PRODUCTOS", "Bienes y servicios específicos entregados"),
            ("RESULTADOS", "Cambios medibles en la población objetivo"),
            ("IMPACTOS", "Transformaciones territoriales sostenibles"),
            ("OUTCOMES", "Efectos de largo plazo en desarrollo humano")
        ]
        
        for tipo_idx, (tipo_nombre, descripcion) in enumerate(tipos_eslabon):
            eslabon_data = {
                "id": f"{tipo_nombre.lower()[:3]}_{dim_id}",
                "tipo": tipo_nombre,
                "descripcion": descripcion,
                "indicadores": [
                    f"indicador_{tipo_nombre.lower()}_cuantitativo_{dim_id}_{i+1}" for i in range(3)
                ] + [
                    f"indicador_{tipo_nombre.lower()}_cualitativo_{dim_id}_{i+1}" for i in range(2)
                ],
                "capacidades_requeridas": [
                    f"capacidad_tecnica_{tipo_nombre.lower()}_{dim_id}_{i+1}" for i in range(3)
                ] + [
                    f"capacidad_institucional_{tipo_nombre.lower()}_{dim_id}_{i+1}" for i in range(2)
                ],
                "puntos_criticos": [
                    f"punto_critico_operativo_{tipo_nombre.lower()}_{dim_id}_{i+1}" for i in range(2)
                ] + [
                    f"punto_critico_estrategico_{tipo_nombre.lower()}_{dim_id}_{i+1}" for i in range(1)
                ],
                "ventana_temporal": [
                    tipo_idx * 4 + 1,  # Inicio
                    (tipo_idx + 1) * 4 + 8   # Fin con superposición
                ],
                "kpi_ponderacion": 1.0 + (tipo_idx * 0.2) + ((dim_id % 3) * 0.1),
                "riesgos_especificos": [
                    f"riesgo_presupuestal_{tipo_nombre.lower()}_{dim_id}",
                    f"riesgo_temporal_{tipo_nombre.lower()}_{dim_id}",
                    f"riesgo_institucional_{tipo_nombre.lower()}_{dim_id}"
                ],
                "dependencias": [
                    f"{tipos_eslabon[max(0, tipo_idx-1)][0].lower()[:3]}_{dim_id}"
                ] if tipo_idx > 0 else [],
                "stakeholders": [
                    f"stakeholder_institucional_{tipo_nombre.lower()}_{dim_id}",
                    f"stakeholder_comunitario_{tipo_nombre.lower()}_{dim_id}",
                    f"stakeholder_sectorial_{tipo_nombre.lower()}_{dim_id}"
                ],
                "recursos_estimados": {
                    "financiero": float(tipo_idx * 50000000 + dim_id * 10000000),  # En pesos colombianos
                    "humano": float(tipo_idx * 2 + 3),  # Número de personas
                    "tiempo": float((tipo_idx + 1) * 6)  # Meses
                }
            }
            dim["eslabones"].append(eslabon_data)
        
        template_avanzado.append(dim)
    
    # Guardar template avanzado
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(template_avanzado, f, indent=2, ensure_ascii=False)
    
    LOGGER.info(f"✅ Template avanzado de decálogo generado: {json_path}")
    LOGGER.warning("⚠️ COMPLETE Y VALIDE MANUALMENTE 'decalogo_industrial_avanzado.json'")
    
    return cargar_decalogo_industrial_avanzado()

# Cargar decálogo avanzado
DECALOGO_INDUSTRIAL_AVANZADO = cargar_decalogo_industrial_avanzado()

# -------------------- Metadatos avanzados de clusters --------------------
@dataclass(frozen=True)
class ClusterMetadataAvanzada:
    """Metadatos avanzados de clusters con análisis profundo."""
    cluster_id: str
    titulo: str
    descripcion_extendida: str
    puntos: List[int]
    logica_agrupacion: str
    teoria_cambio_cluster: Dict[str, Any]
    interconexiones: Dict[str, float]
    complejidad_agregada: float
    prioridad_politica: float
    
    def calcular_metricas_cluster(self) -> Dict[str, float]:
        """Calcula métricas avanzadas del cluster."""
        try:
            # Métricas básicas
            num_puntos = len(self.puntos)
            densidad_interconexion = len(self.interconexiones) / max(1, num_puntos * (num_puntos - 1) / 2)
            
            # Fuerza de interconexión promedio
            fuerza_promedio = np.mean(list(self.interconexiones.values())) if self.interconexiones else 0.5
            
            # Factor de complejidad balanceada
            factor_complejidad = min(1.0, self.complejidad_agregada / num_puntos)
            
            return {
                'densidad_interconexion': densidad_interconexion,
                'fuerza_interconexion_promedio': fuerza_promedio,
                'factor_complejidad_balanceada': factor_complejidad,
                'puntaje_cohesion_cluster': (densidad_interconexion + fuerza_promedio + 
                                           (1 - factor_complejidad)) / 3,
                'implementabilidad_cluster': min(1.0, self.prioridad_politica / self.complejidad_agregada),
                'num_puntos': num_puntos
            }
        except Exception:
            return {
                'densidad_interconexion': 0.5,
                'fuerza_interconexion_promedio': 0.5,
                'factor_complejidad_balanceada': 0.5,
                'puntaje_cohesion_cluster': 0.5,
                'implementabilidad_cluster': 0.5,
                'num_puntos': len(self.puntos)
            }

@dataclass(frozen=True)
class DecalogoContextoAvanzado:
    """Contexto avanzado del decálogo con capacidades analíticas superiores."""
    dimensiones_por_id: Dict[int, DimensionDecalogoAvanzada]
    clusters_por_id: Dict[str, ClusterMetadataAvanzada]
    cluster_por_dimension: Dict[int, ClusterMetadataAvanzada]
    matriz_interdependencias: np.ndarray
    ontologia: OntologiaPoliticasAvanzada
    
    def calcular_interdependencias_avanzadas(self) -> Dict[str, Any]:
        """Calcula interdependencias avanzadas entre dimensiones."""
        try:
            n_dims = len(self.dimensiones_por_id)
            matriz_sim = np.zeros((n_dims, n_dims))
            
            # Calcular similaridad entre dimensiones
            for i in range(n_dims):
                for j in range(n_dims):
                    if i != j:
                        dim_i = self.dimensiones_por_id[i+1]
                        dim_j = self.dimensiones_por_id[j+1]
                        
                        # Similaridad basada en eslabones
                        tipos_i = {e.tipo for e in dim_i.eslabones}
                        tipos_j = {e.tipo for e in dim_j.eslabones}
                        sim_tipos = len(tipos_i.intersection(tipos_j)) / len(tipos_i.union(tipos_j))
                        
                        # Similaridad basada en interdependencias declaradas
                        sim_interdep = 1.0 if j+1 in dim_i.interdependencias else 0.0
                        
                        # Similaridad de cluster
                        sim_cluster = 1.0 if dim_i.cluster == dim_j.cluster else 0.3
                        
                        matriz_sim[i, j] = (sim_tipos * 0.4 + sim_interdep * 0.4 + sim_cluster * 0.2)
            
            # Análisis de componentes principales
            eigenvals, eigenvecs = np.linalg.eigh(matriz_sim)
            varianza_explicada = eigenvals / np.sum(eigenvals)
            
            # Detección de comunidades (clustering)
            from sklearn.cluster import SpectralClustering
            if n_dims > 3:
                clustering = SpectralClustering(n_clusters=4, affinity='precomputed', random_state=42)
                cluster_labels = clustering.fit_predict(matriz_sim)
            else:
                cluster_labels = np.arange(n_dims)
            
            return {
                'matriz_similaridad': matriz_sim,
                'eigenvalues': eigenvals,
                'varianza_explicada': varianza_explicada,
                'comunidades_detectadas': cluster_labels,
                'densidad_red': np.mean(matriz_sim[matriz_sim > 0]),
                'centralidad_dimensiones': np.sum(matriz_sim, axis=1),
                'modularidad': self._calcular_modularidad(matriz_sim, cluster_labels)
            }
        
        except Exception as e:
            LOGGER.warning(f"Error calculando interdependencias avanzadas: {e}")
            return {'error': str(e)}
    
    def _calcular_modularidad(self, matriz_adj: np.ndarray, clusters: np.ndarray) -> float:
        """Calcula modularidad de la red de interdependencias."""
        try:
            m = np.sum(matriz_adj) / 2  # Número total de aristas
            if m == 0:
                return 0.0
            
            modularidad = 0.0
            n = len(matriz_adj)
            
            for i in range(n):
                for j in range(n):
                    if clusters[i] == clusters[j]:
                        ki = np.sum(matriz_adj[i])
                        kj = np.sum(matriz_adj[j])
                        expected = (ki * kj) / (2 * m)
                        modularidad += matriz_adj[i, j] - expected
            
            return modularidad / (2 * m)
        except Exception:
            return 0.5

# Definiciones avanzadas de clusters
_CLUSTER_DEFINITIONS_AVANZADAS = {
    "CLUSTER_1": {
        "titulo": "CLUSTER 1: PAZ TERRITORIAL Y SEGURIDAD HUMANA INTEGRAL",
        "descripcion_extendida": """
        Este cluster agrupa las dimensiones relacionadas con la construcción de paz territorial,
        la protección de defensores de derechos humanos y la implementación de justicia transicional.
        Se enfoca en crear condiciones de seguridad humana integral que permitan el desarrollo
        sostenible del territorio y la participación ciudadana efectiva.
        """,
        "puntos": [1, 5, 8],
        "logica": """
        La lógica de agrupación se basa en la interconexión entre paz territorial, protección
        de líderes sociales y construcción de memoria histórica como elementos fundamentales
        para la consolidación democrática y el desarrollo humano integral.
        """,
        "teoria_cambio_cluster": {
            "hipotesis_principal": "La consolidación de la paz territorial requiere protección efectiva de defensores y construcción de memoria histórica",
            "supuestos_criticos": [
                "Existe voluntad política para la protección efectiva",
                "Las comunidades participan activamente en procesos de memoria",
                "Los mecanismos institucionales funcionan coordinadamente"# -------------------- Decálogo avanzado con capacidades de frontera --------------------
@dataclass(frozen=True)
class DimensionDecalogoAvanzada:
    """Dimensión del decálogo con capacidades matemáticas de frontera."""
    id: int
    nombre: str
    cluster: str
    teoria_cambio: TeoriaCambioAvanzada
    eslabones: List[EslabonCadenaAvanzado]
    prioridad_estrategica: float = 1.0
    complejidad_implementacion: float = 0.5
    interdependencias: List[int] = field(default_factory=list)
    contexto_territorial: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (1 <= self.id <= 10):
            raise ValueError("ID de dimensión debe estar entre 1 y 10")
        if len(self.nombre) < 10:
            raise ValueError("Nombre de dimensión debe ser más descriptivo")
        if len(self.eslabones) < 4:
            raise ValueError("Debe haber al menos 4 eslabones por dimensión para análisis robusto")
        if not (0.1 <= self.prioridad_estrategica <= 3.0):
            raise ValueError("Prioridad estratégica debe estar entre 0.1 y 3.0")
    
    def evaluar_coherencia_causal_avanzada(self) -> Dict[str, float]:
        """Evaluación avanzada de coherencia causal con múltiples métricas."""
        try:
            # Verificación de identificabilidad
            identificabilidad = self.teoria_cambio.verificar_identificabilidad_avanzada()
            puntaje_identificabilidad = identificabilidad['puntaje_global_identificabilidad']
            
            # Análisis de cobertura de tipos de eslabones
            tipos_presentes = {e.tipo for e in self.eslabones}
            tipos_criticos = {TipoCadenaValor.INSUMOS, TipoCadenaValor.PROCESOS, TipoCadenaValor.PRODUCTOS}
            cobertura_critica = len(tipos_criticos.intersection(tipos_presentes)) / len(tipos_criticos)
            
            # Bonus por cobertura completa
            tiene_resultados = any(e.tipo == TipoCadenaValor.RESULTADOS for e in self.eslabones)
            tiene_impactos = any(e.tipo == TipoCadenaValor.IMPACTOS for e in self.eslabones)
            bonus_cobertura = 0.3 if (tiene_resultados and tiene_impactos) else 0.1
            
            # Análisis de complejidad vs manejabilidad
            complejidades = [e.calcular_metricas_avanzadas()['complejidad_operativa'] for e in self.eslabones]
            complejidad_promedio = np.mean(complejidades)
            factor_manejabilidad = max(0.3, 1.0 - (complejidad_promedio - 0.5) * 0.8)
            
            # Coherencia temporal
            ventanas_temporales = [e.ventana_temporal for e in self.eslabones]
            coherencia_temporal = self._evaluar_coherencia_temporal(ventanas_temporales)
            
            # Análisis de dependencias circulares
            factor_dependencias = self._evaluar_dependencias_circulares()
            
            # Cálculo agregado con pesos sofisticados
            coherencia_global = (
                puntaje_identificabilidad * 0.25 +
                cobertura_critica * 0.20 +
                bonus_cobertura * 0.15 +
                factor_manejabilidad * 0.15 +
                coherencia_temporal * 0.15 +
                factor_dependencias * 0.10
            )
            
            return {
                'coherencia_global': coherencia_global,
                'identificabilidad_causal': puntaje_identificabilidad,
                'cobertura_eslabones': cobertura_critica + bonus_cobertura / 3,
                'manejabilidad_complejidad': factor_manejabilidad,
                'coherencia_temporal': coherencia_temporal,
                'dependencias_circulares': factor_dependencias,
                'nivel_coherencia': self._clasificar_coherencia(coherencia_global)
            }
        
        except Exception as e:
            LOGGER.warning(f"Error evaluando coherencia causal avanzada: {e}")
            return {
                'coherencia_global': 0.4,
                'identificabilidad_causal': 0.4,
                'cobertura_eslabones': 0.4,
                'manejabilidad_complejidad': 0.4,
                'coherencia_temporal': 0.4,
                'dependencias_circulares': 0.4,
                'nivel_coherencia': 'BAJA'
            }
    
    def _evaluar_coherencia_temporal(self, ventanas: List[Tuple[int, int]]) -> float:
        """Evalúa coherencia temporal entre eslabones."""
        if not ventanas:
            return 0.5
        
        try:
            # Ordenar por inicio de ventana
            ventanas_ordenadas = sorted(ventanas, key=lambda x: x[0])
            
            # Verificar superposiciones lógicas
            superposiciones_logicas = 0
            total_comparaciones = 0
            
            for i in range(len(ventanas_ordenadas) - 1):
                for j in range(i + 1, len(ventanas_ordenadas)):
                    v1, v2 = ventanas_ordenadas[i], ventanas_ordenadas[j]
                    
                    # Hay superposición?
                    hay_superposicion = not (v1[1] < v2[0] or v2[1] < v1[0])
                    if hay_superposicion:
                        superposiciones_logicas += 1
                    
                    total_comparaciones += 1
            
            # Ratio de superposiciones (esperado en procesos paralelos)
            ratio_superposicion = superposiciones_logicas / max(1, total_comparaciones)
            
            # Evaluar dispersión temporal
            inicios = [v[0] for v in ventanas]
            fines = [v[1] for v in ventanas]
            
            dispersiom_inicios = np.std(inicios) if len(inicios) > 1 else 0
            dispersiom_fines = np.std(fines) if len(fines) > 1 else 0
            
            # Coherencia basada en dispersión controlada
            factor_dispersion = max(0.2, 1.0 - (dispersiom_inicios + dispersiom_fines) / 48)  # 48 meses max
            
            return (ratio_superposicion * 0.4 + factor_dispersion * 0.6)
        
        except Exception:
            return 0.5
    
    def _evaluar_dependencias_circulares(self) -> float:
        """Evalúa si existen dependencias circulares problemáticas."""
        try:
            # Construir grafo de dependencias entre eslabones
            G = nx.DiGraph()
            
            for eslabon in self.eslabones:
                G.add_node(eslabon.id)
                for dep in eslabon.dependencias:
                    if any(e.id == dep for e in self.eslabones):
                        G.add_edge(dep, eslabon.id)
            
            # Detectar ciclos
            try:
                ciclos = list(nx.simple_cycles(G))
                factor_ciclos = max(0.3, 1.0 - len(ciclos) * 0.2)
                return factor_ciclos
            except Exception:
                return 0.7  # Asume pocas dependencias circulares
        
        except Exception:
            return 0.6
    
    def _clasificar_coherencia(self, coherencia: float) -> str:
        """Clasifica nivel de coherencia."""
        if coherencia >= 0.9: return "EXCELENTE"
        if coherencia >= 0.8: return "ALTA"
        if coherencia >= 0.65: return "MEDIA-ALTA"
        if coherencia >= 0.5: return "MEDIA"
        if coherencia >= 0.35: return "BAJA"
        return "CRITICA"
    
    def calcular_kpi_global_avanzado(self) -> Dict[str, float]:
        """Cálculo avanzado de KPI global con múltiples dimensiones."""
        try:
            metricas_eslabones = [e.calcular_metricas_avanzadas() for e in self.eslabones]
            
            # KPI básico ponderado
            kpi_basico = sum(e.kpi_ponderacion for e in self.eslabones) / len(self.eslabones)
            
            # Factor de complejidad agregada
            complejidades = [m['complejidad_operativa'] for m in metricas_eslabones]
            factor_complejidad = 1.0 - (np.mean(complejidades) * 0.3)
            
            # Factor de riesgo agregado
            riesgos = [m['riesgo_agregado'] for m in metricas_eslabones]
            factor_riesgo = 1.0 - (np.mean(riesgos) * 0.4)
            
            # Factor de recursos
            recursos = [m['intensidad_recursos'] for m in metricas_eslabones]
            factor_recursos = np.mean(recursos) * 0.8 + 0.2  # Base mínima del 20%
            
            # KPI global ajustado
            kpi_global_ajustado = kpi_basico * factor_complejidad * factor_riesgo * factor_recursos
            
            # Métricas adicionales
            lead_times = [m['lead_time_normalizado'] for m in metricas_eslabones]
            criticidades = [m['criticidad_global'] for m in metricas_eslabones]
            
            return {
                'kpi_basico': kpi_basico,
                'kpi_global_ajustado': kpi_global_ajustado,
                'factor_complejidad': factor_complejidad,
                'factor_riesgo': factor_riesgo,
                'factor_recursos': factor_recursos,
                'lead_time_promedio': np.mean(lead_times),
                'criticidad_promedio': np.mean(criticidades),
                'prioridad_estrategica': self.prioridad_estrategica,
                'score_implementabilidad': min(1.0, kpi_global_ajustado / self.complejidad_implementacion)
            }
        
        except Exception:
            return {
                'kpi_basico': 1.0,
                'kpi_global_ajustado': 0.8,
                'factor_complejidad': 0.7,
                'factor_riesgo': 0.7,
                'factor_recursos': 0.6,
                'lead_time_promedio': 0.5,
                'criticidad_promedio': 0.5,
                'prioridad_estrategica': self.prioridad_estrategica,
                'score_implementabilidad': 0.6
            }
    
    def generar_matriz_riesgos_avanzada(self) -> Dict[str, Dict[str, Any]]:
        """Generación de matriz de riesgos avanzada con análisis probabilístico."""
        matriz_riesgos = {}
        
        try:
            for eslabon in self.eslabones:
                riesgos_eslabon = []
                probabilidades = []
                impactos = []
                
                # Riesgos específicos del eslabón
                for riesgo in eslabon.riesgos_especificos:
                    prob_base = 0.3  # Probabilidad base
                    impacto_base = 0.5  # Impacto base
                    
                    # Ajustes por contexto
                    if "presupuestal" in riesgo.lower():
                        prob_base += 0.2
                        impacto_base += 0.3
                    elif "temporal" in riesgo.lower():
                        prob_base += 0.15
                        impacto_base += 0.2
                    elif "institucional" in riesgo.lower():
                        prob_base += 0.1
                        impacto_base += 0.25
                    
                    riesgos_eslabon.append(riesgo)
                    probabilidades.append(min(0.95, prob_base))
                    impactos.append(min(0.95, impacto_base))
                
                # Riesgos sistémicos identificados automáticamente
                metricas = eslabon.calcular_metricas_avanzadas()
                
                # Riesgo por alta complejidad
                if metricas['complejidad_operativa'] > 0.7:
                    riesgos_eslabon.append("RIESGO SISTÉMICO: Alta complejidad operativa")
                    probabilidades.append(0.4)
                    impactos.append(0.6)
                
                # Riesgo por recursos insuficientes
                if metricas['intensidad_recursos'] < 0.3:
                    riesgos_eslabon.append("RIESGO SISTÉMICO: Recursos insuficientes identificados")
                    probabilidades.append(0.5)
                    impactos.append(0.7)
                
                # Riesgo por lead time extenso
                if metricas['lead_time_normalizado'] > 0.8:
                    riesgos_eslabon.append("RIESGO SISTÉMICO: Ventana temporal muy extensa")
                    probabilidades.append(0.3)
                    impactos.append(0.4)
                
                # Riesgo por dependencias múltiples
                if len(eslabon.dependencias) > 3:
                    riesgos_eslabon.append("RIESGO SISTÉMICO: Múltiples dependencias críticas")
                    probabilidades.append(0.35)
                    impactos.append(0.5)
                
                # Cálculo de riesgo agregado
                if riesgos_eslabon and probabilidades and impactos:
                    # Riesgo = Probabilidad × Impacto para cada riesgo
                    riesgos_individuales = [p * i for p, i in zip(probabilidades, impactos)]
                    riesgo_agregado = 1.0 - np.prod([1.0 - r for r in riesgos_individuales])
                else:
                    riesgo_agregado = 0.2
                
                # Clasificación de riesgo
                if riesgo_agregado >= 0.7:
                    clasificacion = "CRÍTICO"
                    color = "🔴"
                elif riesgo_agregado >= 0.5:
                    clasificacion = "ALTO"
                    color = "🟠"
                elif riesgo_agregado >= 0.3:
                    clasificacion = "MEDIO"
                    color = "🟡"
                else:
                    clasificacion = "BAJO"
                    color = "🟢"
                
                matriz_riesgos[eslabon.id] = {
                    'riesgos_especificos': riesgos_eslabon,
                    'probabilidades': probabilidades,
                    'impactos': impactos,
                    'riesgo_agregado': riesgo_agregado,
                    'clasificacion': clasificacion,
                    'color_indicator': color,
                    'medidas_mitigacion': self._generar_medidas_mitigacion(eslabon, riesgos_eslabon),
                    'monitoreo_indicadores': self._generar_indicadores_monitoreo(eslabon)
                }
            
            return matriz_riesgos
        
        except Exception as e:
            LOGGER.warning(f"Error generando matriz de riesgos avanzada: {e}")
            return {e.id: {'riesgos_especificos': ['Error en análisis de riesgos'], 
                          'riesgo_agregado': 0.5, 'clasificacion': 'MEDIO'} 
                    for e in self.eslabones}
    
    def _generar_medidas_mitigacion(self, eslabon: EslabonCadenaAvanzado, riesgos: List[str]) -> List[str]:
        """Genera medidas de mitigación específicas por tipo de riesgo."""
        medidas = []
        
        # Análisis de riesgos para generar medidas específicas
        for riesgo in riesgos:
            if "presupuestal" in riesgo.lower():
                medidas.append("Establecer fondos de contingencia (mínimo 5% del presupuesto)")
                medidas.append("Diversificar fuentes de financiación")
            elif "temporal" in riesgo.lower():
                medidas.append("Implementar metodologías ágiles de gestión")
                medidas.append("Establecer hitos de control mensual")
            elif "institucional" in riesgo.lower():
                medidas.append("Fortalecer capacidades del equipo técnico")
                medidas.append("Establecer comité directivo de alto nivel")
            elif "complejidad" in riesgo.lower():
                medidas.append("Implementar enfoque de implementación por fases")
                medidas.append("Establecer quick wins tempranos")
            elif "dependencias" in riesgo.lower():
                medidas.append("Mapear y gestionar dependencias críticas")
                medidas.append("Establecer acuerdos de nivel de servicio (SLA)")
        
        # Medidas generales si no hay medidas específicas
        if not medidas:
            medidas.extend([
                "Implementar sistema de monitoreo continuo",
                "Establecer plan de contingencia operativo",
                "Fortalecer coordinación interinstitucional"
            ])
        
        return list(set(medidas))  # Eliminar duplicados
    
    def _generar_indicadores_monitoreo(self, eslabon: EslabonCadenaAvanzado) -> List[str]:
        """Genera indicadores de monitoreo específicos."""
        indicadores = []
        
        # Indicadores por tipo de eslabón
        if eslabon.tipo == TipoCadenaValor.INSUMOS:
            indicadores.extend([
                "Porcentaje de presupuesto ejecutado",
                "Número de recursos humanos asignados vs planificados",
                "Disponibilidad de infraestructura requerida (%)"
            ])
        elif eslabon.tipo == TipoCadenaValor.PROCESOS:
            indicadores.extend([
                "Porcentaje de procesos implementados según cronograma",
                "Tiempo promedio de ejecución de procesos críticos",
                "Número de cuellos de botella identificados y resueltos"
            ])
        elif eslabon.tipo == TipoCadenaValor.PRODUCTOS:
            indicadores.extend([
                "Porcentaje de productos entregados según especificaciones",
                "Índice de calidad de productos (escala 1-10)",
                "Tiempo de entrega promedio vs planificado"
            ])
        elif eslabon.tipo == TipoCadenaValor.RESULTADOS:
            indicadores.extend([
                "Porcentaje de población objetivo alcanzada",
                "Nivel de satisfacción de beneficiarios (%)",
                "Cambios medibles en indicadores de resultado"
            ])
        elif eslabon.tipo == TipoCadenaValor.IMPACTOS:
            indicadores.extend([
                "Variación en indicadores de desarrollo territorial",
                "Índice de sostenibilidad de resultados",
                "Contribución a objetivos de desarrollo sostenible"
            ])
        
        # Indicadores transversales
        indicadores.extend([
            "Nivel de riesgo agregado (mensual)",
            "Índice de coordinación interinstitucional",
            "Porcentaje de hitos cumplidos en tiempo"
        ])
        
        return indicadores# -*- coding: utf-8 -*-
"""
Sistema Integral de Evaluación de Cadenas de Valor en Planes de Desarrollo Municipal
Versión: 9.0 — Marco Teórico-Institucional con Análisis Causal Multinivel, Frontier AI Capabilities,
Mathematical Innovation, Sophisticated Evidence Processing y Reporting Industrial.
Framework basado en IAD + Theory of Change, con triangulación cuali-cuantitativa,
verificación causal, certeza probabilística y capacidades de frontera.
Autor: Dr. en Políticas Públicas
Enfoque: Evaluación estructural con econometría de políticas, minería causal avanzada,
procesamiento paralelo industrial y reportes masivos granulares.
"""

import argparse
import atexit
import hashlib
import heapq
import json
import logging
import re
import signal
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn.functional as F

# -------------------- Dependencias avanzadas --------------------
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    pdfplumber = None

import spacy
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer, util

# Módulos matemáticos avanzados
try:
    from scipy.spatial.distance import cdist
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.stats import entropy, chi2_contingency, pearsonr, spearmanr
    ADVANCED_STATS_AVAILABLE = True
except ImportError:
    ADVANCED_STATS_AVAILABLE = False

# Capacidades de frontera en NLP
try:
    import transformers
    from transformers import pipeline, AutoTokenizer, AutoModel
    FRONTIER_NLP_AVAILABLE = True
except ImportError:
    FRONTIER_NLP_AVAILABLE = False

try:
    from decalogo_loader import get_decalogo_industrial
except ImportError:
    def get_decalogo_industrial():
        return "Fallback: Decálogo industrial para desarrollo municipal con 10 dimensiones estratégicas."

# Device configuration avanzada
try:
    from device_config import add_device_args, configure_device_from_args, get_device_config, to_device
except ImportError:
    def add_device_args(parser):
        parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
        parser.add_argument('--precision', default='float32', choices=['float16', 'float32'], help='Precision')
        parser.add_argument('--batch_size', default=16, type=int, help='Batch size for processing')
        return parser

    def configure_device_from_args(args):
        return AdvancedDeviceConfig(
            args.device if hasattr(args, 'device') else 'cpu',
            args.precision if hasattr(args, 'precision') else 'float32',
            args.batch_size if hasattr(args, 'batch_size') else 16
        )

    def get_device_config():
        return AdvancedDeviceConfig('cpu', 'float32', 16)

    def to_device(model):
        return model

    class AdvancedDeviceConfig:
        def __init__(self, device='cpu', precision='float32', batch_size=16):
            self.device = device
            self.precision = precision
            self.batch_size = batch_size
        
        def get_device(self):
            return self.device
        
        def get_precision(self):
            return torch.float16 if self.precision == 'float16' else torch.float32
        
        def get_batch_size(self):
            return self.batch_size
        
        def get_device_info(self):
            return {
                'device_type': self.device,
                'precision': self.precision,
                'batch_size': self.batch_size,
                'num_threads': torch.get_num_threads(),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'memory_info': self._get_memory_info()
            }
        
        def _get_memory_info(self):
            if torch.cuda.is_available():
                return {
                    'allocated': torch.cuda.memory_allocated() / 1024**3,
                    'reserved': torch.cuda.memory_reserved() / 1024**3,
                    'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
                }
            return {'cpu_memory': 'N/A'}

# Text processing avanzado
try:
    from text_truncation_logger import (
        get_truncation_logger, log_debug_with_text, log_error_with_text,
        log_info_with_text, log_warning_with_text, truncate_text_for_log,
    )
except ImportError:
    def get_truncation_logger(name): 
        return logging.getLogger(name)
    def log_debug_with_text(logger, text): 
        logger.debug(truncate_text_for_log(text, 500))
    def log_error_with_text(logger, text): 
        logger.error(truncate_text_for_log(text, 500))
    def log_info_with_text(logger, text): 
        logger.info(truncate_text_for_log(text, 500))
    def log_warning_with_text(logger, text): 
        logger.warning(truncate_text_for_log(text, 500))
    def truncate_text_for_log(text, max_len=500): 
        return text[:max_len] + "..." if len(text) > max_len else text

# Requerimiento de versión
assert sys.version_info >= (3, 11), "Python 3.11 or higher is required"

# Suprimir warnings innecesarios
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- Logging industrial avanzado --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"evaluacion_politicas_industrial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
    ],
)
LOGGER = logging.getLogger("EvaluacionPoliticasPublicasIndustrial")

# -------------------- Carga de modelos con capacidades de frontera --------------------
try:
    NLP = spacy.load("es_core_news_lg")
    log_info_with_text(LOGGER, "✅ Modelo SpaCy avanzado cargado (es_core_news_lg)")
except OSError:
    try:
        NLP = spacy.load("es_core_news_sm")
        log_warning_with_text(LOGGER, "⚠️ Usando modelo SpaCy básico (es_core_news_sm)")
    except OSError as e:
        log_error_with_text(LOGGER, f"❌ Error cargando SpaCy: {e}")
        raise SystemExit("Modelo SpaCy no disponible. Ejecute: python -m spacy download es_core_news_lg")

try:
    EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    EMBEDDING_MODEL = to_device(EMBEDDING_MODEL)
    log_info_with_text(LOGGER, "✅ Modelo de embeddings multilingual cargado")
    log_info_with_text(LOGGER, f"✅ Dispositivo: {get_device_config().get_device()}")
except Exception as e:
    log_error_with_text(LOGGER, f"❌ Error cargando embeddings: {e}")
    raise SystemExit(f"Error cargando modelo de embeddings: {e}")

# Carga de modelos de frontera para análisis avanzado
ADVANCED_NLP_PIPELINE = None
if FRONTIER_NLP_AVAILABLE:
    try:
        ADVANCED_NLP_PIPELINE = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True
        )
        log_info_with_text(LOGGER, "✅ Pipeline NLP avanzado cargado para análisis de sentimientos")
    except Exception as e:
        log_warning_with_text(LOGGER, f"⚠️ Pipeline NLP avanzado no disponible: {e}")

# -------------------- Innovaciones matemáticas --------------------
class MathematicalInnovations:
    """Clase con innovaciones matemáticas para análisis de políticas públicas."""
    
    @staticmethod
    def calculate_causal_strength(graph: nx.DiGraph, source: str, target: str) -> float:
        """Calcula la fuerza causal entre dos nodos usando innovaciones en teoría de grafos."""
        try:
            if not nx.has_path(graph, source, target):
                return 0.0
            
            # Innovación: Combinación de múltiples métricas de centralidad
            paths = list(nx.all_simple_paths(graph, source, target, cutoff=5))
            if not paths:
                return 0.0
            
            # Cálculo de fuerza causal ponderada
            total_strength = 0.0
            for path in paths:
                path_strength = 1.0
                for i in range(len(path) - 1):
                    edge_weight = graph.get_edge_data(path[i], path[i+1], {}).get('weight', 0.5)
                    path_strength *= edge_weight
                
                # Penalización por longitud de camino
                length_penalty = 0.8 ** (len(path) - 2)
                total_strength += path_strength * length_penalty
            
            # Normalización basada en la centralidad de los nodos
            source_centrality = nx.betweenness_centrality(graph).get(source, 0.1)
            target_centrality = nx.betweenness_centrality(graph).get(target, 0.1)
            centrality_factor = (source_centrality + target_centrality) / 2
            
            return min(1.0, total_strength * (1 + centrality_factor))
        
        except Exception:
            return 0.3
    
    @staticmethod
    def bayesian_evidence_integration(evidences: List[float], priors: List[float]) -> float:
        """Integración bayesiana de evidencias para cálculo de certeza probabilística."""
        if not evidences or not priors:
            return 0.5
        
        try:
            # Innovación: Actualización bayesiana iterativa
            posterior = priors[0] if priors else 0.5
            
            for i, evidence in enumerate(evidences):
                likelihood = evidence
                prior = posterior
                
                # Aplicación del teorema de Bayes
                numerator = likelihood * prior
                denominator = likelihood * prior + (1 - likelihood) * (1 - prior)
                posterior = numerator / denominator if denominator > 0 else prior
                
                # Regularización para evitar valores extremos
                posterior = max(0.01, min(0.99, posterior))
            
            return posterior
        
        except Exception:
            return np.mean(evidences) if evidences else 0.5
    
    @staticmethod
    def entropy_based_complexity(elements: List[str]) -> float:
        """Calcula complejidad basada en entropía de elementos."""
        if not elements:
            return 0.0
        
        try:
            # Distribución de frecuencias
            from collections import Counter
            freq_dist = Counter(elements)
            total = sum(freq_dist.values())
            probabilities = [count / total for count in freq_dist.values()]
            
            # Cálculo de entropía de Shannon
            entropy_val = -sum(p * np.log2(p) for p in probabilities if p > 0)
            
            # Normalización por máxima entropía posible
            max_entropy = np.log2(len(freq_dist))
            normalized_entropy = entropy_val / max_entropy if max_entropy > 0 else 0.0
            
            return normalized_entropy
        
        except Exception:
            return 0.5
    
    @staticmethod
    def fuzzy_logic_aggregation(values: List[float], weights: List[float] = None) -> Dict[str, float]:
        """Agregación difusa avanzada de valores con múltiples operadores."""
        if not values:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'fuzzy_and': 0.0, 'fuzzy_or': 0.0}
        
        values = np.array(values)
        weights = np.array(weights) if weights else np.ones(len(values))
        weights = weights / np.sum(weights)  # Normalización
        
        try:
            # Operadores difusos clásicos
            fuzzy_and = np.min(values)  # T-norma mínima
            fuzzy_or = np.max(values)   # T-conorma máxima
            
            # Operadores avanzados
            weighted_mean = np.sum(values * weights)
            geometric_mean = np.exp(np.sum(weights * np.log(np.maximum(values, 1e-10))))
            harmonic_mean = 1.0 / np.sum(weights / np.maximum(values, 1e-10))
            
            # Agregación OWA (Ordered Weighted Averaging)
            sorted_values = np.sort(values)[::-1]  # Orden descendente
            owa_weights = np.array([0.4, 0.3, 0.2, 0.1])[:len(sorted_values)]
            owa_weights = owa_weights / np.sum(owa_weights)
            owa_result = np.sum(sorted_values[:len(owa_weights)] * owa_weights)
            
            return {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(weighted_mean),
                'geometric_mean': float(geometric_mean),
                'harmonic_mean': float(harmonic_mean),
                'fuzzy_and': float(fuzzy_and),
                'fuzzy_or': float(fuzzy_or),
                'owa': float(owa_result),
                'std': float(np.std(values)),
                'entropy': MathematicalInnovations.entropy_based_complexity([str(v) for v in values])
            }
        
        except Exception:
            return {
                'min': float(np.min(values)) if len(values) > 0 else 0.0,
                'max': float(np.max(values)) if len(values) > 0 else 0.0,
                'mean': float(np.mean(values)) if len(values) > 0 else 0.0,
                'fuzzy_and': 0.0,
                'fuzzy_or': 0.0,
                'owa': 0.0,
                'std': 0.0,
                'entropy': 0.0
            }

# -------------------- Marco teórico avanzado --------------------
class NivelAnalisis(Enum):
    MACRO = "Institucional-Sistémico"
    MESO = "Organizacional-Sectorial"
    MICRO = "Operacional-Territorial"
    META = "Meta-Evaluativo"

class TipoCadenaValor(Enum):
    INSUMOS = "Recursos financieros, humanos y físicos"
    PROCESOS = "Transformación institucional y gestión"
    PRODUCTOS = "Bienes/servicios entregables medibles"
    RESULTADOS = "Cambios conductuales/institucionales"
    IMPACTOS = "Bienestar y desarrollo humano sostenible"
    OUTCOMES = "Efectos de largo plazo y sostenibilidad"

class TipoEvidencia(Enum):
    CUANTITATIVA = "Datos numéricos y estadísticas"
    CUALITATIVA = "Narrativas y descripciones"
    MIXTA = "Combinación cuanti-cualitativa"
    DOCUMENTAL = "Evidencia documental y normativa"
    TESTIMONIAL = "Testimonios y entrevistas"

@dataclass(frozen=True)
class TeoriaCambioAvanzada:
    """Teoría de cambio avanzada con capacidades matemáticas de frontera."""
    supuestos_causales: List[str]
    mediadores: Dict[str, List[str]]
    resultados_intermedios: List[str]
    precondiciones: List[str]
    moderadores: List[str] = field(default_factory=list)
    variables_contextuales: List[str] = field(default_factory=list)
    mecanismos_causales: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if len(self.supuestos_causales) == 0:
            raise ValueError("Supuestos causales no pueden estar vacíos")
        if len(self.mediadores) == 0:
            raise ValueError("Mediadores no pueden estar vacíos")
    
    def verificar_identificabilidad_avanzada(self) -> Dict[str, float]:
        """Verificación avanzada de identificabilidad causal."""
        criterios = {
            'supuestos_suficientes': len(self.supuestos_causales) >= 2,
            'mediadores_diversificados': len(self.mediadores) >= 2,
            'resultados_especificos': len(self.resultados_intermedios) >= 1,
            'precondiciones_definidas': len(self.precondiciones) >= 1,
            'moderadores_identificados': len(self.moderadores) >= 1,
            'mecanismos_explicitos': len(self.mecanismos_causales) >= 1
        }
        
        puntajes = {k: 1.0 if v else 0.0 for k, v in criterios.items()}
        puntaje_global = np.mean(list(puntajes.values()))
        
        return {
            'puntaje_global_identificabilidad': puntaje_global,
            'criterios_individuales': puntajes,
            'nivel_identificabilidad': self._clasificar_identificabilidad(puntaje_global)
        }
    
    def _clasificar_identificabilidad(self, puntaje: float) -> str:
        if puntaje >= 0.9: return "EXCELENTE"
        if puntaje >= 0.75: return "ALTA"
        if puntaje >= 0.6: return "MEDIA"
        if puntaje >= 0.4: return "BAJA"
        return "INSUFICIENTE"
    
    def construir_grafo_causal_avanzado(self) -> nx.DiGraph:
        """Construcción de grafo causal con propiedades avanzadas."""
        G = nx.DiGraph()
        
        # Nodos básicos
        G.add_node("insumos", tipo="nodo_base", nivel="input", centralidad=1.0)
        G.add_node("impactos", tipo="nodo_base", nivel="outcome", centralidad=1.0)
        
        # Adición de nodos con atributos enriquecidos
        for categoria, lista in self.mediadores.items():
            for i, mediador in enumerate(lista):
                G.add_node(
                    mediador, 
                    tipo="mediador", 
                    categoria=categoria,
                    orden=i,
                    peso_teorico=0.8 + (i * 0.1)
                )
                G.add_edge("insumos", mediador, weight=0.9, tipo="causal_directa")
        
        # Resultados intermedios con conexiones complejas
        for i, resultado in enumerate(self.resultados_intermedios):
            G.add_node(
                resultado, 
                tipo="resultado_intermedio",
                orden=i,
                criticidad=0.7 + (i * 0.1)
            )
            
            # Conexiones desde mediadores
            mediadores_disponibles = [n for n in G.nodes if G.nodes[n].get("tipo") == "mediador"]
            for mediador in mediadores_disponibles:
                G.add_edge(
                    mediador, 
                    resultado, 
                    weight=0.8 - (i * 0.1), 
                    tipo="causal_mediada"
                )
            
            # Conexión al impacto final
            G.add_edge(
                resultado, 
                "impactos", 
                weight=0.9 - (i * 0.05), 
                tipo="causal_final"
            )
        
        # Moderadores como nodos especiales
        for moderador in self.moderadores:
            G.add_node(moderador, tipo="moderador", influencia="contextual")
            # Los moderadores influencian las relaciones, no son parte del flujo directo
        
        # Precondiciones como requisitos
        for precond in self.precondiciones:
            G.add_node(precond, tipo="precondicion", necesidad="critica")
            G.add_edge(precond, "insumos", weight=1.0, tipo="prerequisito")
        
        return G
    
    def calcular_coeficiente_causal_avanzado(self) -> Dict[str, float]:
        """Cálculo avanzado de coeficientes causales."""
        G = self.construir_grafo_causal_avanzado()
        
        if len(G.nodes) < 3:
            return {'coeficiente_global': 0.3, 'robustez_estructural': 0.2, 'complejidad_causal': 0.1}
        
        try:
            # Métricas estructurales
            density = nx.density(G)
            avg_clustering = nx.average_clustering(G.to_undirected())
            
            # Análisis de caminos causales
            mediadores = [n for n in G.nodes if G.nodes[n].get("tipo") == "mediador"]
            resultados = [n for n in G.nodes if G.nodes[n].get("tipo") == "resultado_intermedio"]
            
            # Innovación: Cálculo de fuerza causal usando la clase MathematicalInnovations
            fuerza_causal = MathematicalInnovations.calculate_causal_strength(G, "insumos", "impactos")
            
            # Robustez estructural
            robustez = self._calcular_robustez_estructural(G, mediadores, resultados)
            
            # Complejidad causal
            elementos_causales = (self.supuestos_causales + 
                                list(self.mediadores.keys()) + 
                                self.resultados_intermedios + 
                                self.moderadores)
            complejidad = MathematicalInnovations.entropy_based_complexity(elementos_causales)
            
            return {
                'coeficiente_global': fuerza_causal,
                'robustez_estructural': robustez,
                'complejidad_causal': complejidad,
                'densidad_grafo': density,
                'clustering_promedio': avg_clustering,
                'nodos_totales': len(G.nodes),
                'aristas_totales': len(G.edges)
            }
        
        except Exception as e:
            LOGGER.warning(f"Error en cálculo causal avanzado: {e}")
            return {'coeficiente_global': 0.5, 'robustez_estructural': 0.4, 'complejidad_causal': 0.3}
    
    def _calcular_robustez_estructural(self, G: nx.DiGraph, mediadores: List[str], resultados: List[str]) -> float:
        """Cálculo de robustez estructural del grafo causal."""
        try:
            # Simulación de perturbaciones
            robustez_scores = []
            
            for _ in range(100):  # 100 simulaciones
                G_perturbed = G.copy()
                
                # Remover aleatoriamente algunos nodos mediadores
                nodes_to_remove = np.random.choice(
                    mediadores, 
                    size=min(len(mediadores) // 3, 2), 
                    replace=False
                ) if len(mediadores) > 2 else []
                
                for node in nodes_to_remove:
                    if G_perturbed.has_node(node):
                        G_perturbed.remove_node(node)
                
                # Verificar si aún existe camino causal principal
                if nx.has_path(G_perturbed, "insumos", "impactos"):
                    robustez_scores.append(1.0)
                else:
                    robustez_scores.append(0.0)
            
            return np.mean(robustez_scores)
        
        except Exception:
            return 0.5

@dataclass(frozen=True)
class EslabonCadenaAvanzado:
    """Eslabón de cadena de valor con capacidades avanzadas."""
    id: str
    tipo: TipoCadenaValor
    indicadores: List[str]
    capacidades_requeridas: List[str]
    puntos_criticos: List[str]
    ventana_temporal: Tuple[int, int]
    kpi_ponderacion: float = 1.0
    riesgos_especificos: List[str] = field(default_factory=list)
    dependencias: List[str] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    recursos_estimados: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0 <= self.kpi_ponderacion <= 3.0):
            raise ValueError("KPI ponderación debe estar entre 0 y 3.0")
        if self.ventana_temporal[0] > self.ventana_temporal[1]:
            raise ValueError("Ventana temporal inválida")
        if len(self.indicadores) == 0:
            raise ValueError("Debe tener al menos un indicador")
    
    def calcular_metricas_avanzadas(self) -> Dict[str, float]:
        """Cálculo de métricas avanzadas del eslabón."""
        try:
            # Complejidad operativa
            complejidad_operativa = (
                len(self.capacidades_requeridas) * 0.3 +
                len(self.puntos_criticos) * 0.4 +
                len(self.dependencias) * 0.3
            ) / 10.0  # Normalización
            
            # Riesgo agregado
            riesgo_agregado = min(1.0, len(self.riesgos_especificos) * 0.2)
            
            # Intensidad de recursos
            intensidad_recursos = sum(self.recursos_estimados.values()) / max(1, len(self.recursos_estimados))
            intensidad_recursos = min(1.0, intensidad_recursos / 1000000)  # Normalización por millones
            
            # Lead time normalizado
            lead_time = self.calcular_lead_time()
            lead_time_normalizado = min(1.0, lead_time / 24)  # Normalización por 24 meses
            
            # Factor de stakeholders
            factor_stakeholders = min(1.0, len(self.stakeholders) * 0.15)
            
            return {
                'complejidad_operativa': complejidad_operativa,
                'riesgo_agregado': riesgo_agregado,
                'intensidad_recursos': intensidad_recursos,
                'lead_time_normalizado': lead_time_normalizado,
                'factor_stakeholders': factor_stakeholders,
                'kpi_ponderado': self.kpi_ponderacion / 3.0,  # Normalización
                'criticidad_global': (complejidad_operativa + riesgo_agregado + lead_time_normalizado) / 3
            }
        
        except Exception:
            return {
                'complejidad_operativa': 0.5,
                'riesgo_agregado': 0.5,
                'intensidad_recursos': 0.5,
                'lead_time_normalizado': 0.5,
                'factor_stakeholders': 0.3,
                'kpi_ponderado': self.kpi_ponderacion / 3.0,
                'criticidad_global': 0.5
            }
    
    def calcular_lead_time(self) -> float:
        """Cálculo optimizado del lead time."""
        return (self.ventana_temporal[0] + self.ventana_temporal[1]) / 2.0
    
    def generar_hash_avanzado(self) -> str:
        """Generación de hash avanzado del eslabón."""
        data = (f"{self.id}|{self.tipo.value}|{sorted(self.indicadores)}|"
                f"{sorted(self.capacidades_requeridas)}|{sorted(self.riesgos_especificos)}|"
                f"{self.ventana_temporal}|{self.kpi_ponderacion}")
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

# -------------------- Ontología avanzada --------------------
@dataclass
class OntologiaPoliticasAvanzada:
    """Ontología avanzada para políticas públicas con capacidades de frontera."""
    dimensiones: Dict[str, List[str]]
    relaciones_causales: Dict[str, List[str]]
    indicadores_ods: Dict[str, List[str]]
    taxonomia_evidencia: Dict[str, List[str]]
    patrones_linguisticos: Dict[str, List[str]]
    vocabulario_especializado: Dict[str, List[str]]
    fecha_creacion: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "3.0-industrial-frontier"
    
    @classmethod
    def cargar_ontologia_avanzada(cls) -> 'OntologiaPoliticasAvanzada':
        """Carga ontología avanzada con capacidades de frontera."""
        try:
            # Dimensiones expandidas con granularidad superior
            dimensiones_frontier = {
                "social_avanzado": [
                    "salud_preventiva", "educacion_calidad", "vivienda_digna", "proteccion_social_integral",
                    "equidad_genero", "inclusion_diversidad", "cohesion_social", "capital_social",
                    "bienestar_subjetivo", "calidad_vida_urbana", "seguridad_ciudadana", "participacion_comunitaria"
                ],
                "economico_transformacional": [
                    "empleo_decente", "productividad_sectorial", "innovacion_tecnologica", "infraestructura_inteligente",
                    "competitividad_territorial", "emprendimiento_social", "economia_circular", "finanzas_sostenibles",
                    "comercio_justo", "turismo_sostenible", "agroindustria_sustentable", "servicios_avanzados"
                ],
                "ambiental_regenerativo": [
                    "sostenibilidad_integral", "biodiversidad_conservacion", "mitigacion_climatica", "adaptacion_climatica",
                    "gestion_integral_residuos", "gestion_hidrica", "energia_renovable", "movilidad_sostenible",
                    "construccion_verde", "agricultura_regenerativa", "bosques_urbanos", "economia_verde"
                ],
                "institucional_transformativo": [
                    "gobernanza_multinivel", "transparencia_activa", "participacion_ciudadana", "rendicion_cuentas",
                    "eficiencia_administrativa", "innovacion_publica", "gobierno_abierto", "justicia_social",
                    "estado_derecho", "capacidades_institucionales", "coordinacion_intersectorial", "planificacion_estrategica"
                ],
                "territorial_inteligente": [
                    "ordenamiento_territorial", "planificacion_urbana", "conectividad_digital", "logistica_territorial",
                    "patrimonio_cultural", "identidad_territorial", "resiliencia_territorial", "sistemas_urbanos"
                ]
            }
            
            # Relaciones causales avanzadas con múltiples niveles
            relaciones_causales_avanzadas = {
                "inversion_publica_inteligente": [
                    "crecimiento_economico_sostenible", "empleo_formal_calidad", "infraestructura_resiliente",
                    "capacidades_institucionales", "innovacion_territorial", "equidad_espacial"
                ],
                "educacion_transformacional": [
                    "productividad_laboral_avanzada", "innovacion_social", "reduccion_desigualdades",
                    "cohesion_social", "capital_humano_especializado", "emprendimiento_innovador"
                ],
                "salud_integral": [
                    "productividad_economica", "calidad_vida_poblacional", "equidad_social_territorial",
                    "resilienza_comunitaria", "capital_social_saludable"
                ],
                "gobernanza_inteligente": [
                    "transparencia_institucional", "eficiencia_publica", "confianza_ciudadana",
                    "participacion_democratica", "legitimidad_estatal", "capacidad_adaptativa"
                ],
                "sostenibilidad_regenerativa": [
                    "resiliencia_climatica", "economia_circular_territorial", "bienestar_ecosistemico",
                    "salud_ambiental", "prosperidad_sostenible", "justicia_intergeneracional"
                ]
            }
            
            # Taxonomía de evidencia sofisticada
            taxonomia_evidencia_avanzada = {
                "cuantitativa_robusta": [
                    "estadisticas_oficiales", "encuestas_representativas", "censos_poblacionales",
                    "registros_administrativos", "indicadores_desempeño", "metricas_impacto",
                    "series_temporales", "analisis_econometricos", "evaluaciones_impacto"
                ],
                "cualitativa_profunda": [
                    "entrevistas_profundidad", "grupos_focales", "observacion_participante",
                    "etnografia_institucional", "narrativas_territoriales", "historias_vida",
                    "analisis_discurso", "mapeo_actores", "analisis_redes_sociales"
                ],
                "mixta_integrativa": [
                    "triangulacion_metodologica", "evaluacion_realista", "analisis_configuracional",
                    "metodos_participativos", "investigacion_accion", "evaluacion_desarrollo"
                ],
                "documental_normativa": [
                    "planes_desarrollo", "politicas_publicas", "normatividad_vigente",
                    "reglamentaciones_tecnicas", "lineamientos_sectoriales", "directrices_internacionales"
                ]
            }
            
            # Patrones lingüísticos avanzados para detección de evidencia
            patrones_linguisticos_especializados = {
                "indicadores_desempeño": [
                    r"\b(?:indicador|metric|medidor|parametro|kpi)\b.*\b(?:de|para|del)\b.*\b(?:desempeño|resultado|impacto|logro)\b",
                    r"\b(?:medir|evaluar|monitorear|seguir|rastrear)\b.*\b(?:progreso|avance|cumplimiento|efectividad)\b",
                    r"\b(?:linea\s+base|baseline|situacion\s+inicial|punto\s+partida)\b.*\d+",
                    r"\b(?:meta|objetivo|target|proposito)\b.*\d+.*\b(?:2024|2025|2026|2027|2028)\b"
                ],
                "recursos_financieros": [
                    r"\$\s*[\d,.]+(?: millones?| mil(?:es)?| billones?)?\b",
                    r"\bpresupuesto\b.*\$?[\d,.]+(?: millones?| mil(?:es)?| billones?)?",
                    r"\b(?:inversion|asignacion|destinacion|cofinanciacion)\b.*\$?[\d,.]+(?: millones?| mil(?:es)?)?",
                    r"\b(?:recursos|fondos|capital|financiacion)\b.*\$?[\d,.]+(?: millones?| mil(?:es)?)?",
                    r"\bCOP\s*[\d,.]+(?: millones?| mil(?:es)?| billones?)?\b"
                ],
                "responsabilidades_institucionales": [
                    r"\b(?:responsable|encargado|lidera|coordina|gestiona|ejecuta)\b:\s*\w+",
                    r"\b(?:secretaria|ministerio|departamento|entidad|institucion)\b.*\b(?:responsable|cargo|funcion)\b",
                    r"\b(?:quien|que)\b.*\b(?:lidera|coordina|ejecuta|implementa)\b",
                    r"\brol\b.*\b(?:de|del|para)\b.*\b(?:secretaria|ministerio|entidad)\b"
                ],
                "temporalidad_plazos": [
                    r"\b(?:plazo|cronograma|calendario|programacion|tiempo)\b.*\b(?:de|para|del)\b.*\b(?:implementacion|ejecucion|desarrollo)\b",
                    r"\b(?:inicio|comienzo|arranque)\b.*\b(?:en|el|durante)\b.*\b(?:20\d{2}|primer|segundo|tercer|cuarto)\b.*\b(?:trimestre|semestre|año)\b",
                    r"\b(?:duracion|periodo|etapa|fase)\b.*\b(?:de|del)\b.*\b(?:\d+)\b.*\b(?:meses|años|trimestres)\b",
                    r"\b(?:hasta|para|antes|durante)\b.*\b(?:20\d{2}|diciembre|final|culminacion)\b"
                ],
                "impactos_resultados": [
                    r"\b(?:impacto|efecto|resultado|consecuencia|cambio)\b.*\b(?:en|sobre|para)\b.*\b(?:poblacion|comunidad|territorio)\b",
                    r"\b(?:beneficio|mejora|incremento|reduccion|disminucion)\b.*\b(?:del|de la|en el|en la)\b.*\b(?:\d+%|\d+ puntos)\b",
                    r"\b(?:transformacion|cambio|modificacion)\b.*\b(?:social|economica|ambiental|institucional|territorial)\b"
                ]
            }
            
            # Vocabulario especializado expandido
            vocabulario_especializado_ampliado = {
                "planificacion_territorial": [
                    "ordenamiento_territorial", "zonificacion", "uso_suelo", "plan_ordenamiento",
                    "esquema_ordenamiento", "plan_basico_ordenamiento", "pot", "eot", "pbot",
                    "suelo_urbano", "suelo_rural", "suelo_expansion", "suelo_proteccion"
                ],
                "desarrollo_sostenible": [
                    "objetivos_desarrollo_sostenible", "ods", "agenda_2030", "sostenibilidad",
                    "desarrollo_humano", "crecimiento_verde", "economia_circular", "resilencia_climatica"
                ],
                "gobernanza_publica": [
                    "participacion_ciudadana", "transparencia", "rendicion_cuentas", "gobierno_abierto",
                    "cocreacion", "corresponsabilidad", "veeduria_ciudadana", "control_social"
                ],
                "gestion_publica": [
                    "meci", "modelo_integrado_planeacion_gestion", "sistema_gestion_calidad",
                    "plan_desarrollo_territorial", "pdt", "plan_accion", "seguimiento_evaluacion"
                ]
            }
            
            # Carga de indicadores ODS especializados
            indicadores_ods_especializados = cls._cargar_indicadores_ods_avanzados()
            
            return cls(
                dimensiones=dimensiones_frontier,
                relaciones_causales=relaciones_causales_avanzadas,
                indicadores_ods=indicadores_ods_especializados,
                taxonomia_evidencia=taxonomia_evidencia_avanzada,
                patrones_linguisticos=patrones_linguisticos_especializados,
                vocabulario_especializado=vocabulario_especializado_ampliado
            )
            
        except Exception as e:
            log_error_with_text(LOGGER, f"❌ Error cargando ontología avanzada: {e}")
            raise SystemExit("Fallo en carga de ontología avanzada")
    
    @staticmethod
    def _cargar_indicadores_ods_avanzados() -> Dict[str, List[str]]:
        """Carga indicadores ODS con granularidad avanzada."""
        indicadores_path = Path("indicadores_ods_avanzados.json")
        
        # Indicadores base expandidos y especializados
        indicadores_especializados = {
            "ods1_pobreza": [
                "tasa_pobreza_monetaria", "tasa_pobreza_extrema", "indice_pobreza_multidimensional",
                "coeficiente_gini", "proteccion_social_cobertura", "acceso_servicios_basicos",
                "vulnerabilidad_economica", "resiliencia_economica_hogares", "activos_productivos_acceso"
            ],
            "ods3_salud": [
                "mortalidad_infantil", "mortalidad_materna", "esperanza_vida_nacimiento",
                "acceso_servicios_salud", "cobertura_vacunacion", "prevalencia_enfermedades_cronicas",
                "salud_mental_indicadores", "seguridad_alimentaria", "agua_potable_saneamiento_acceso"
            ],
            "ods4_educacion": [
                "tasa_alfabetizacion", "matriucla_educacion_basica", "permanencia_educativa",
                "calidad_educativa_pruebas", "acceso_educacion_superior", "formacion_tecnica_profesional",
                "educacion_digital_competencias", "infraestructura_educativa_calidad"
            ],
            "ods5_genero": [
                "participacion_politica_mujeres", "brecha_salarial_genero", "violencia_genero_prevalencia",
                "acceso_credito_mujeres", "liderazgo_empresarial_femenino", "uso_tiempo_trabajo_cuidado",
                "educacion_ciencia_tecnologia_mujeres", "derechos_reproductivos_acceso"
            ],
            "ods8_trabajo": [
                "tasa_empleo", "tasa_desempleo", "empleo_informal", "trabajo_decente_indicadores",
                "productividad_laboral", "crecimiento_economico_pib", "diversificacion_economica",
                "emprendimiento_formal", "inclusion_financiera", "innovacion_empresarial"
            ],
            "ods11_ciudades": [
                "vivienda_adecuada_acceso", "transporte_publico_acceso", "espacios_publicos_calidad",
                "gestion_residuos_solidos", "calidad_aire", "planificacion_urbana_participativa",
                "patrimonio_cultural_proteccion", "resiliencia_desastres", "conectividad_urbana"
            ],
            "ods13_clima": [
                "emisiones_gei_per_capita", "vulnerabilidad_climatica", "adaptacion_climatica_medidas",
                "educacion_ambiental", "energia_renovable_uso", "eficiencia_energetica",
                "conservacion_ecosistemas", "reforestacion_restauracion", "economia_baja_carbono"
            ],
            "ods16_paz": [
                "indice_transparencia", "percepcion_corrupcion", "acceso_justicia",
                "participacion_decisiones_publicas", "libertad_expresion", "seguridad_ciudadana",
                "confianza_instituciones", "estado_derecho_fortalecimiento", "inclusion_social_politica"
            ],
            "ods17_alianzas": [
                "cooperacion_internacional", "transferencia_tecnologia", "capacitacion_institucional",
                "movilizacion_recursos_domesticos", "comercio_internacional", "acceso_mercados",
                "sostenibilidad_deuda", "sistemas_monitoreo_datos", "alianzas_publico_privadas"
            ]
        }
        
        # Intentar cargar desde archivo si existe
        if indicadores_path.exists():
            try:
                with open(indicadores_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and len(data) >= 8:
                    LOGGER.info("✅ Indicadores ODS avanzados cargados desde archivo")
                    return data
                else:
                    LOGGER.warning("⚠️ Indicadores ODS avanzados inválidos, usando base especializada")
            except Exception as e:
                LOGGER.warning(f"⚠️ Error leyendo indicadores avanzados {indicadores_path}: {e}")
        
        # Guardar template avanzado
        try:
            with open(indicadores_path, "w", encoding="utf-8") as f:
                json.dump(indicadores_especializados, f, indent=2, ensure_ascii=False)
            LOGGER.info(f"✅ Template ODS avanzado generado: {indicadores_path}")
        except Exception as e:
            LOGGER.error(f"❌ Error generando template ODS avanzado: {e}")
        
        return indicadores_especializados
    
    def buscar_patrones_avanzados(self, texto: str, categoria: str) -> List[Dict[str, Any]]:
        """Búsqueda avanzada de patrones lingüísticos en texto."""
        if categoria not in self.patrones_linguisticos:
            return []
        
        patrones = self.patrones_linguisticos[categoria]
        resultados = []
        
        for i, patron in enumerate(patrones):
            try:
                matches = re.finditer(patron, texto, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    resultado = {
                        'texto_encontrado': match.group(),
                        'posicion_inicio': match.start(),
                        'posicion_fin': match.end(),
                        'patron_id': i,
                        'categoria': categoria,
                        'confianza': self._calcular_confianza_patron(match.group(), patron),
                        'contexto': texto[max(0, match.start()-50):match.end()+50]
                    }
                    resultados.append(resultado)
            except re.error:
                continue
        
        return sorted(resultados, key=lambda x: x['confianza'], reverse=True)
    
    def _calcular_confianza_patron(self, texto_match: str, patron: str) -> float:
        """Calcula confianza del patrón encontrado."""
        try:
            # Factores de confianza
            longitud_factor = min(1.0, len(texto_match) / 50)  # Textos más largos = mayor confianza
            complejidad_patron = min(1.0, len(patron) / 100)  # Patrones más complejos = mayor precisión
            
            # Verificar presencia de números (para indicadores cuantitativos)
            tiene_numeros = bool(re.search(r'\d+', texto_match))
            factor_numerico = 0.2 if tiene_numeros else 0.0
            
            # Verificar presencia de fechas
            tiene_fechas = bool(re.search(r'20\d{2}', texto_match))
            factor_temporal = 0.15 if tiene_fechas else 0.0
            
            # Confianza base
            confianza_base = 0.6
            
            return min(1.0, confianza_base + longitud_factor * 0.2 + complejidad_patron * 0.1 + 
                      factor_numerico + factor_temporal)
        
        except Exception:
            return 0.5