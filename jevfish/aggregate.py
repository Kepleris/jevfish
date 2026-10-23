"""Agregación ponderada (post-estratificación por segmento) con intervalos bootstrap."""

from __future__ import annotations

import math
import random

from .personas import Persona
from .scenario import Question, Scenario
from .simulate import RoundAnswers, top_option


def agent_weights(scenario: Scenario, personas: list[Persona], answered: set[str]) -> dict[str, float]:
    """Cada agente pesa peso_segmento / agentes_con_respuesta_en_el_segmento."""
    counts: dict[str, int] = {}
    for p in personas:
        if p.id in answered:
            counts[p.segment.id] = counts.get(p.segment.id, 0) + 1
    covered = sum(s.peso for s in scenario.segmentos if counts.get(s.id))
    return {
        p.id: p.segment.peso / counts[p.segment.id] / covered
        for p in personas
        if p.id in answered
    }


def _weighted_mean(question: Question, ids: list[str], weights: dict[str, float], answers: RoundAnswers) -> dict[str, float]:
    total = sum(weights[i] for i in ids)
    return {
        k: sum(weights[i] * answers[i][question.id][k] for i in ids) / total
        for k in question.keys
    }


def _entropy(dist: dict[str, float]) -> float:
    if len(dist) < 2:
        return 0.0
    return -sum(p * math.log(p) for p in dist.values() if p > 0) / math.log(len(dist))


def _expected_level(dist: dict[str, float]) -> float:
    return sum(int(k) * p for k, p in dist.items())


def aggregate_question(
    scenario: Scenario,
    question: Question,
    personas: list[Persona],
    answers: RoundAnswers,
    bootstrap: int = 500,
    seed: int = 0,
) -> dict:
    ids = [p.id for p in personas if p.id in answers]
    weights = agent_weights(scenario, personas, set(ids))

    mean = _weighted_mean(question, ids, weights, answers)
    votes = {k: 0.0 for k in question.keys}
    for i in ids:
        votes[top_option(answers[i][question.id])] += weights[i]

    # Bootstrap estratificado: se remuestrea dentro de cada segmento para mantener los pesos.
    rng = random.Random(seed)
    by_segment: dict[str, list[str]] = {}
    for p in personas:
        if p.id in answers:
            by_segment.setdefault(p.segment.id, []).append(p.id)
    samples: dict[str, list[float]] = {k: [] for k in question.keys}
    for _ in range(bootstrap):
        resampled = [pid for seg_ids in by_segment.values() for pid in rng.choices(seg_ids, k=len(seg_ids))]
        total = sum(weights[pid] for pid in resampled)
        for k in question.keys:
            samples[k].append(sum(weights[pid] * answers[pid][question.id][k] for pid in resampled) / total)

    def interval(values: list[float]) -> list[float]:
        values = sorted(values)
        return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]

    segments = {}
    for segment in scenario.segmentos:
        seg_ids = by_segment.get(segment.id, [])
        if not seg_ids:
            continue
        seg_mean = _weighted_mean(question, seg_ids, weights, answers)
        segments[segment.id] = {
            "nombre": segment.nombre,
            "peso": segment.peso,
            "agentes": len(seg_ids),
            "probabilidades": seg_mean,
            **({"nivel_esperado": _expected_level(seg_mean)} if question.type == "score" else {}),
        }

    stats = {
        "id": question.id,
        "tipo": question.type,
        "pregunta": question.text,
        "agentes": len(ids),
        "opciones": [
            {
                "clave": k,
                "etiqueta": question.label(k),
                "descripcion": question.options.get(k) if question.type == "choice" else None,
                "probabilidad": mean[k],
                "ic95": interval(samples[k]) if bootstrap else None,
                "voto": votes[k],
            }
            for k in question.keys
        ],
        "opcion_mas_probable": top_option(mean),
        "indecision_media": sum(weights[i] * _entropy(answers[i][question.id]) for i in ids),
        "por_segmento": segments,
    }
    if question.type == "score":
        stats["nivel_esperado"] = _expected_level(mean)
    return stats


def aggregate(scenario: Scenario, personas: list[Persona], rounds: list[RoundAnswers], bootstrap: int = 500) -> dict:
    per_round = [
        {q.id: aggregate_question(scenario, q, personas, answers, bootstrap, seed=r) for q in scenario.preguntas}
        for r, answers in enumerate(rounds)
    ]
    evolution = {}
    if len(per_round) > 1:
        first, last = per_round[0], per_round[-1]
        for q in scenario.preguntas:
            start = {o["clave"]: o["probabilidad"] for o in first[q.id]["opciones"]}
            end = {o["clave"]: o["probabilidad"] for o in last[q.id]["opciones"]}
            evolution[q.id] = {k: end[k] - start[k] for k in q.keys}
    return {"rondas": per_round, "final": per_round[-1], "evolucion": evolution}
