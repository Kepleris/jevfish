"""Simulación por rondas: en cada ronda cada agente decide; desde la segunda ve lo que opina su entorno."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import httpx

from .backends import Backend, BackendError, Distribution
from .cache import Cache
from .personas import Persona, build_network
from .scenario import Question, Scenario

RoundAnswers = dict[str, dict[str, Distribution]]  # persona_id -> pregunta_id -> distribución


@dataclass
class SimulationResult:
    rounds: list[RoundAnswers]
    network: dict[str, list[str]]
    failures: list[tuple[int, str, str]] = field(default_factory=list)  # (ronda, persona, error)
    api_calls: int = 0
    cache_hits: int = 0


def top_option(dist: Distribution) -> str:
    return max(dist, key=dist.get)


def peer_summary(questions: list[Question], neighbors: list[str], previous: RoundAnswers) -> list[str]:
    known = [previous[n] for n in neighbors if n in previous]
    if not known:
        return []
    lines = []
    for q in questions:
        counts = Counter(q.label(top_option(answers[q.id])) for answers in known)
        parts = ", ".join(f"{n} se inclinan por «{label}»" for label, n in counts.most_common())
        lines.append(f"Sobre «{q.text}»: de {len(known)} personas cercanas, {parts}.")
    return lines


def build_state(scenario: Scenario, persona: Persona, peers: list[str] | None = None) -> dict:
    state = {"tema": scenario.tema, "situacion": scenario.contexto, "persona": persona.profile()}
    if peers:
        state["lo_que_opina_su_entorno"] = peers
    return state


async def simulate(
    scenario: Scenario,
    personas: list[Persona],
    backend: Backend,
    *,
    rounds: int | None = None,
    concurrency: int | None = None,
    cache: Cache | None = None,
    progress: Callable[[int, int, int], None] | None = None,
) -> SimulationResult:
    sim = scenario.simulacion
    rounds = rounds or sim.rondas
    concurrency = concurrency or backend.default_concurrency
    cache = cache or Cache(None)
    network = build_network(personas, sim.vecinos, sim.homofilia, sim.semilla)
    result = SimulationResult(rounds=[], network=network)
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=20)) as client:
        for r in range(rounds):
            previous = result.rounds[-1] if result.rounds else None
            answers: RoundAnswers = {}
            done = 0

            async def run_one(persona: Persona) -> None:
                nonlocal done
                peers = peer_summary(scenario.preguntas, network[persona.id], previous) if previous else None
                state = build_state(scenario, persona, peers)
                questions_spec = [q.__dict__ for q in scenario.preguntas]
                key = Cache.key(backend.identity, scenario.marco, state, questions_spec)
                cached = cache.get(key)
                try:
                    if cached is None:
                        async with semaphore:
                            cached = await backend.decide(client, state, scenario.preguntas, scenario.marco)
                        result.api_calls += 1
                        cache.set(key, cached)
                    else:
                        result.cache_hits += 1
                    answers[persona.id] = cached
                except BackendError as e:
                    result.failures.append((r + 1, persona.id, str(e)))
                done += 1
                if progress:
                    progress(r + 1, done, len(personas))

            await asyncio.gather(*(run_one(p) for p in personas))
            if not answers:
                first_error = result.failures[-1][2] if result.failures else "sin respuestas"
                raise BackendError(f"ningún agente obtuvo respuesta en la ronda {r + 1}: {first_error}")
            result.rounds.append(answers)
    return result
