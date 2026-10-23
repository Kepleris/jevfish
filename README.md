# JevFish

Simula cómo reaccionaría una población ante algo que la afecta (una ley, un aumento de precio, una
decisión de empresa) y estima, con ponderación por grupos, **qué tan probable es cada decisión**.

Se inspira en [MiroFish](https://github.com/666ghj/MiroFish), pero se queda con lo esencial y lo
hace gratis:

| | MiroFish | JevFish |
|---|---|---|
| Motor | LLMs grandes de chat que generan texto | **Modelo de decisión** `baseten/mammoth-1:free` vía OpenRouter, que devuelve directamente probabilidades por opción |
| Personas | Generadas por LLM con GraphRAG y memoria | Muestreadas de distribuciones del `.json` (sin coste de LLM) |
| Simulación | Red social con publicaciones | Rondas con **influencia social**: cada agente ve qué opinan sus vecinos |
| Resultado | Informe redactado por un agente | Probabilidades ponderadas por segmento, IC 95%, desglose y evolución entre rondas |
| Infraestructura | Frontend, backend, Zep, Docker | Una CLI de Python y una caché SQLite |

## Cómo funciona

1. **Escenario** (`escenarios/*.json`): el contexto, los segmentos de población con su peso real y
   atributos, y las preguntas.
2. **Agentes**: se reparten por segmento con un mínimo por segmento y se muestrean sus atributos
   (edad, ingresos, etc.). Si un segmento queda sobrerrepresentado, la agregación lo corrige.
3. **Ronda 1**: cada agente recibe la situación y su perfil, y mammoth-1 responde **todas las
   preguntas en una sola llamada** con una distribución de probabilidad por pregunta.
4. **Rondas 2+**: cada agente tiene `vecinos` contactos, la mayoría de su mismo grupo según
   `homofilia`. Ve qué eligieron en la ronda anterior y vuelve a decidir.
5. **Agregación**: cada agente pesa `peso_segmento / agentes_del_segmento`. Por cada opción se
   calcula la probabilidad media ponderada, su IC 95% por bootstrap estratificado, el % de agentes
   que la prefieren, el desglose por segmento y el cambio entre la primera y la última ronda.

## Instalación

```bash
uv sync
cp .env.example .env   # y pega tu OPENROUTER_API_KEY (https://openrouter.ai/keys)
```

## Uso

```bash
uv run jevfish validate escenarios/tarifa_transporte.json
uv run jevfish personas escenarios/tarifa_transporte.json --show 2 --state   # ver qué recibe el modelo
uv run jevfish simulate escenarios/tarifa_transporte.json -b mock                 # probar sin API
uv run jevfish simulate escenarios/tarifa_transporte.json --agents 10            # primera prueba real
uv run jevfish simulate escenarios/tarifa_transporte.json                         # mammoth-1 vía OpenRouter
uv run jevfish simulate escenarios/tarifa_transporte.json --model elephant --agents 300 --rounds 3
```

Cada ejecución crea `resultados/<escenario>-<fecha>/` con:

- `reporte.md`: tablas legibles.
- `resultados.json`: todas las métricas por ronda.
- `agentes.jsonl`: cada agente con sus atributos, vecinos y respuestas por ronda.

Tests: `uv run python -m unittest discover -s tests`.

Las respuestas se guardan en `.jevfish_cache/`. Repetir la misma simulación (mismo escenario y
semilla) no vuelve a llamar a la API, y si una ejecución se corta, al relanzarla continúa donde iba.

## Backends

| Backend | Modelo | Coste | Notas |
|---|---|---|---|
| `openrouter` (por defecto) | `baseten/mammoth-1:free` (27B). Alias: `mammoth`, `elephant` (`baseten/elephant-1:free`, 9B) o cualquier id de modelo de decisión | Gratis | Usa la [Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request) (`POST /api/alpha/decisions`), no la de chat. Contexto de 16K tokens. Requiere `OPENROUTER_API_KEY`. |
| `mock` | — | Gratis | Respuestas aleatorias, solo para probar el flujo. |

El modelo también se puede fijar con `JEVFISH_MODEL` en `.env`. Para ver todos los modelos de
decisión disponibles: `curl -s "https://openrouter.ai/api/v1/models?output_modalities=decisions"`.
`resultados.json` guarda la versión exacta del modelo que respondió y el coste (debería ser 0).

## Formato del escenario

```jsonc
{
  "tema": "Título corto",
  "contexto": "La situación que afecta a la población, con datos concretos.",
  "instrucciones_generales": "(opcional) Encuadre que se antepone a cada pregunta.",
  "poblacion": {
    "atributos_comunes": {
      "edad": {"18-29": 0.3, "30-59": 0.5, "60+": 0.2},  // valor: probabilidad
      "zona": ["urbana", "rural"]                         // lista = uniforme
    },
    "segmentos": [
      {
        "id": "afectados",
        "nombre": "Nombre legible",
        "peso": 0.4,                      // proporción real en la población (se normaliza)
        "descripcion": "Quiénes son",
        "afectacion": "Cómo les afecta esto en concreto",
        "atributos": {"edad": {"18-29": 1.0}}   // reemplaza el atributo común para este segmento
      }
    ]
  },
  "preguntas": [
    {"id": "postura", "tipo": "opcion", "pregunta": "¿...?",
     "opciones": {"a_favor": "Descripción", "en_contra": "Descripción"}},   // 1–255 opciones
    {"id": "protesta", "tipo": "si_no", "pregunta": "¿...?"},
    {"id": "intensidad", "tipo": "escala", "pregunta": "¿...?",
     "niveles": ["Nada", "Poco", "Mucho"]}                                  // 1–10, de menor a mayor
  ],
  "simulacion": {
    "agentes": 100, "rondas": 2, "vecinos": 5, "homofilia": 0.7,
    "min_por_segmento": 10, "semilla": 42
  }
}
```

Límites de la API: hasta 512 opciones sumando todas las preguntas (cada `si_no` cuenta 2) y un
estado de hasta 50.000 caracteres.

## Cómo leer los resultados

- **Probabilidad ponderada**: la mejor estimación de la proporción de la población que tomaría esa
  decisión. Se usa la distribución completa de cada agente, no solo su opción favorita.
- **Agentes que la prefieren**: % ponderado de agentes cuya opción más probable es esa. Útil para
  ver mayorías claras.
- **Indecisión media**: 0% significa agentes muy seguros; 100%, agentes que reparten la probabilidad
  por igual entre las opciones.
- **IC 95%**: refleja la variación por muestreo de agentes, **no** el error del modelo ni el de los
  supuestos del escenario.

Esto es una simulación con agentes sintéticos, no una encuesta. Sirve para explorar escenarios y
comparar alternativas, no para sustituir datos reales.
