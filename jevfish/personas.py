"""Generación de agentes sintéticos a partir de los segmentos del escenario (sin LLM)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .scenario import Scenario, Segment


@dataclass
class Persona:
    id: str
    segment: Segment
    attrs: dict[str, str]

    def profile(self) -> dict:
        profile = {"grupo": self.segment.nombre}
        if self.segment.descripcion:
            profile["descripcion_del_grupo"] = self.segment.descripcion
        if self.segment.afectacion:
            profile["como_le_afecta"] = self.segment.afectacion
        profile.update(self.attrs)
        return profile


def allocate(total: int, segments: list[Segment], minimum: int) -> dict[str, int]:
    """Reparte agentes por segmento según su peso, garantizando un mínimo por segmento.

    El mínimo sobrerrepresenta a los segmentos pequeños; la agregación lo corrige
    ponderando cada agente por peso_del_segmento / agentes_del_segmento.
    """
    if total < len(segments):
        raise ValueError(f"se necesitan al menos {len(segments)} agentes (uno por segmento)")
    minimum = max(1, min(minimum, total // len(segments)))
    remaining = total - minimum * len(segments)
    quotas = {s.id: s.peso * remaining for s in segments}
    counts = {sid: minimum + int(q) for sid, q in quotas.items()}
    leftover = total - sum(counts.values())
    for sid in sorted(quotas, key=lambda k: quotas[k] - int(quotas[k]), reverse=True)[:leftover]:
        counts[sid] += 1
    return counts


def generate_personas(scenario: Scenario, total: int | None = None, seed: int | None = None) -> list[Persona]:
    sim = scenario.simulacion
    total = total or sim.agentes
    rng = random.Random(sim.semilla if seed is None else seed)
    counts = allocate(total, scenario.segmentos, sim.min_por_segmento)

    personas = []
    for segment in scenario.segmentos:
        attributes = {**scenario.atributos_comunes, **segment.atributos}
        for i in range(counts[segment.id]):
            attrs = {
                name: rng.choices(list(dist), weights=list(dist.values()))[0]
                for name, dist in attributes.items()
            }
            personas.append(Persona(id=f"{segment.id}-{i + 1:03d}", segment=segment, attrs=attrs))
    return personas


def build_network(personas: list[Persona], k: int, homophily: float, seed: int) -> dict[str, list[str]]:
    """Red social simple: cada agente tiene k vecinos, con preferencia por su propio segmento."""
    rng = random.Random(seed)
    by_segment: dict[str, list[str]] = {}
    for p in personas:
        by_segment.setdefault(p.segment.id, []).append(p.id)
    all_ids = [p.id for p in personas]

    network = {}
    for p in personas:
        k_eff = min(k, len(all_ids) - 1)
        neighbors: set[str] = set()
        attempts = 0
        while len(neighbors) < k_eff and attempts < k_eff * 20:
            attempts += 1
            pool = by_segment[p.segment.id] if rng.random() < homophily else all_ids
            candidate = rng.choice(pool)
            if candidate != p.id:
                neighbors.add(candidate)
        network[p.id] = sorted(neighbors)
    return network
