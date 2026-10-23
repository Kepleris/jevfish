"""Carga y validación de escenarios (.json)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Tipos aceptados en el JSON -> tipos de pregunta de la API System One de Baseten.
TYPE_ALIASES = {
    "opcion": "choice",
    "opciones": "choice",
    "choice": "choice",
    "si_no": "noul",
    "sino": "noul",
    "booleano": "noul",
    "noul": "noul",
    "escala": "score",
    "score": "score",
}
NOUL_KEYS = ("si", "no")

# Límites de la API System One.
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
MAX_TOTAL_OPTIONS = 512

DEFAULT_FRAMING = (
    "Ponte en el lugar de la persona descrita. Teniendo en cuenta su perfil, "
    "cómo le afecta la situación y lo que opina su entorno, indica qué es lo más "
    "probable que esta persona piense o haga."
)


class ScenarioError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(f"- {e}" for e in errors))


@dataclass
class Question:
    id: str
    type: str  # choice | noul | score
    text: str
    options: dict[str, str | None] = field(default_factory=dict)
    levels: list[str] = field(default_factory=list)

    @property
    def keys(self) -> list[str]:
        if self.type == "choice":
            return list(self.options)
        if self.type == "noul":
            return list(NOUL_KEYS)
        return [str(i) for i in range(len(self.levels))]

    def label(self, key: str) -> str:
        if self.type == "choice":
            return key
        if self.type == "noul":
            return "sí" if key == "si" else "no"
        return self.levels[int(key)]

    @property
    def option_cost(self) -> int:
        return 2 if self.type == "noul" else len(self.keys)


@dataclass
class Segment:
    id: str
    nombre: str
    peso: float
    descripcion: str = ""
    afectacion: str = ""
    atributos: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    agentes: int = 100
    rondas: int = 1
    vecinos: int = 5
    homofilia: float = 0.7
    min_por_segmento: int = 5
    semilla: int = 42


@dataclass
class Scenario:
    tema: str
    contexto: str
    segmentos: list[Segment]
    preguntas: list[Question]
    atributos_comunes: dict[str, dict[str, float]] = field(default_factory=dict)
    simulacion: SimulationConfig = field(default_factory=SimulationConfig)
    marco: str = DEFAULT_FRAMING

    def segment(self, segment_id: str) -> Segment:
        return next(s for s in self.segmentos if s.id == segment_id)


def _parse_distribution(name: str, raw, errors: list[str], where: str) -> dict[str, float]:
    """Acepta {"valor": prob} o ["valor", ...] (uniforme) y normaliza a suma 1."""
    if isinstance(raw, list) and raw:
        return {str(v): 1 / len(raw) for v in raw}
    if isinstance(raw, dict) and raw:
        try:
            dist = {str(k): float(v) for k, v in raw.items()}
        except (TypeError, ValueError):
            errors.append(f"{where}: el atributo '{name}' debe tener probabilidades numéricas")
            return {}
        total = sum(dist.values())
        if total <= 0 or any(v < 0 for v in dist.values()):
            errors.append(f"{where}: el atributo '{name}' tiene probabilidades inválidas")
            return {}
        return {k: v / total for k, v in dist.items()}
    errors.append(f"{where}: el atributo '{name}' debe ser una lista o un objeto no vacío")
    return {}


def _parse_attributes(raw, errors: list[str], where: str) -> dict[str, dict[str, float]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        errors.append(f"{where}: 'atributos' debe ser un objeto")
        return {}
    return {name: _parse_distribution(name, dist, errors, where) for name, dist in raw.items()}


def _parse_question(raw: dict, index: int, errors: list[str]) -> Question | None:
    where = f"preguntas[{index}]"
    if not isinstance(raw, dict):
        errors.append(f"{where}: debe ser un objeto")
        return None
    qid = raw.get("id")
    text = raw.get("pregunta")
    qtype = TYPE_ALIASES.get(str(raw.get("tipo", "")).lower())
    if not qid or not isinstance(qid, str):
        errors.append(f"{where}: falta 'id' (texto)")
    if not text:
        errors.append(f"{where}: falta 'pregunta'")
    if not qtype:
        errors.append(f"{where}: 'tipo' debe ser uno de: opcion, si_no, escala")
        return None
    where = f"pregunta '{qid}'"

    question = Question(id=str(qid), type=qtype, text=str(text))
    if qtype == "choice":
        options = raw.get("opciones")
        if isinstance(options, list):
            options = {str(o): None for o in options}
        if not isinstance(options, dict) or not options:
            errors.append(f"{where}: 'opciones' debe ser un objeto {{clave: descripción}} o una lista")
        elif len(options) > MAX_CHOICE_OPTIONS:
            errors.append(f"{where}: máximo {MAX_CHOICE_OPTIONS} opciones")
        else:
            question.options = {str(k): (str(v) if v else None) for k, v in options.items()}
    elif qtype == "score":
        levels = raw.get("niveles")
        if not isinstance(levels, list) or not levels:
            errors.append(f"{where}: 'niveles' debe ser una lista ordenada de menor a mayor")
        elif len(levels) > MAX_SCORE_LEVELS:
            errors.append(f"{where}: máximo {MAX_SCORE_LEVELS} niveles")
        else:
            question.levels = [str(level) for level in levels]
    return question


def parse_scenario(data: dict) -> Scenario:
    errors: list[str] = []
    if not isinstance(data, dict):
        raise ScenarioError(["el escenario debe ser un objeto JSON"])

    tema = data.get("tema")
    contexto = data.get("contexto")
    if not tema:
        errors.append("falta 'tema'")
    if not contexto:
        errors.append("falta 'contexto' (la situación que afecta a la población)")

    poblacion = data.get("poblacion") or {}
    common = _parse_attributes(poblacion.get("atributos_comunes"), errors, "poblacion")

    segments: list[Segment] = []
    for i, raw in enumerate(poblacion.get("segmentos") or []):
        where = f"segmentos[{i}]"
        sid = raw.get("id") if isinstance(raw, dict) else None
        if not sid:
            errors.append(f"{where}: falta 'id'")
            continue
        try:
            peso = float(raw.get("peso", 0))
        except (TypeError, ValueError):
            peso = 0
        if peso <= 0:
            errors.append(f"segmento '{sid}': 'peso' debe ser un número mayor que 0")
        segments.append(
            Segment(
                id=str(sid),
                nombre=str(raw.get("nombre") or sid),
                peso=peso,
                descripcion=str(raw.get("descripcion") or ""),
                afectacion=str(raw.get("afectacion") or ""),
                atributos=_parse_attributes(raw.get("atributos"), errors, f"segmento '{sid}'"),
            )
        )
    if not segments:
        errors.append("'poblacion.segmentos' debe tener al menos un segmento")
    if len({s.id for s in segments}) != len(segments):
        errors.append("hay ids de segmento repetidos")

    questions = [
        q for i, raw in enumerate(data.get("preguntas") or []) if (q := _parse_question(raw, i, errors))
    ]
    if not questions:
        errors.append("'preguntas' debe tener al menos una pregunta")
    if len({q.id for q in questions}) != len(questions):
        errors.append("hay ids de pregunta repetidos")
    total_options = sum(q.option_cost for q in questions)
    if total_options > MAX_TOTAL_OPTIONS:
        errors.append(
            f"las preguntas suman {total_options} opciones; la API admite {MAX_TOTAL_OPTIONS} "
            "por solicitud (cada si_no cuenta como 2)"
        )

    sim_raw = data.get("simulacion") or {}
    defaults = SimulationConfig()
    try:
        sim = SimulationConfig(
            agentes=int(sim_raw.get("agentes", defaults.agentes)),
            rondas=int(sim_raw.get("rondas", defaults.rondas)),
            vecinos=int(sim_raw.get("vecinos", defaults.vecinos)),
            homofilia=float(sim_raw.get("homofilia", defaults.homofilia)),
            min_por_segmento=int(sim_raw.get("min_por_segmento", defaults.min_por_segmento)),
            semilla=int(sim_raw.get("semilla", defaults.semilla)),
        )
    except (TypeError, ValueError):
        errors.append("'simulacion' tiene valores no numéricos")
        sim = defaults
    if sim.rondas < 1:
        errors.append("'simulacion.rondas' debe ser al menos 1")
    if not 0 <= sim.homofilia <= 1:
        errors.append("'simulacion.homofilia' debe estar entre 0 y 1")

    if errors:
        raise ScenarioError(errors)

    total_weight = sum(s.peso for s in segments)
    for s in segments:
        s.peso /= total_weight

    return Scenario(
        tema=str(tema),
        contexto=str(contexto),
        segmentos=segments,
        preguntas=questions,
        atributos_comunes=common,
        simulacion=sim,
        marco=str(data.get("instrucciones_generales") or DEFAULT_FRAMING),
    )


def load_scenario(path: str | Path) -> Scenario:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ScenarioError([f"JSON inválido: {e}"]) from e
    return parse_scenario(data)
