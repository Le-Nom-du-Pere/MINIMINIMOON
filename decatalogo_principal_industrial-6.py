# -------------------- Sistema de carga del decálogo avanzado --------------------
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