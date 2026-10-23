"""Línea de comandos: jevfish validate | personas | simulate."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from .aggregate import aggregate
from .backends import BACKENDS, BackendError, make_backend
from .cache import Cache
from .personas import generate_personas
from .report import write_outputs
from .scenario import ScenarioError, load_scenario
from .simulate import build_state, simulate


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def cmd_validate(args) -> int:
    scenario = load_scenario(args.scenario)
    options = sum(q.option_cost for q in scenario.preguntas)
    print(f"✓ Escenario válido: {scenario.tema}")
    print(f"  {len(scenario.segmentos)} segmentos, {len(scenario.preguntas)} preguntas ({options} opciones en total)")
    sim = scenario.simulacion
    print(f"  Llamadas previstas: {sim.agentes} agentes × {sim.rondas} rondas = {sim.agentes * sim.rondas}")
    return 0


def cmd_personas(args) -> int:
    scenario = load_scenario(args.scenario)
    personas = generate_personas(scenario, args.agents)
    print(f"{len(personas)} agentes: {dict(Counter(p.segment.id for p in personas))}\n")
    for p in personas[: args.show]:
        print(json.dumps(build_state(scenario, p), ensure_ascii=False, indent=2) if args.state else f"{p.id}: {p.attrs}")
    return 0


def cmd_simulate(args) -> int:
    scenario = load_scenario(args.scenario)
    sim = scenario.simulacion
    if args.seed is not None:
        sim.semilla = args.seed
    rounds = args.rounds or sim.rondas
    personas = generate_personas(scenario, args.agents)
    backend = make_backend(args.backend, args.model)
    cache = Cache(None if args.no_cache else Path(".jevfish_cache") / "respuestas.sqlite")

    print(f"Tema: {scenario.tema}")
    print(f"Backend: {backend.name} ({backend.model}) · {len(personas)} agentes × {rounds} rondas", file=sys.stderr)
    if backend.name == "mock":
        print("⚠ Backend mock: respuestas aleatorias, solo para probar el flujo.", file=sys.stderr)

    interactive = sys.stderr.isatty()

    def progress(r: int, done: int, total: int) -> None:
        if interactive:
            print(f"\r  Ronda {r}/{rounds}: {done}/{total} agentes", end="" if done < total else "\n", file=sys.stderr)
        elif done == total or done % max(1, total // 10) == 0:
            print(f"  Ronda {r}/{rounds}: {done}/{total} agentes", file=sys.stderr)

    try:
        result = asyncio.run(
            simulate(scenario, personas, backend, rounds=rounds, concurrency=args.concurrency, cache=cache, progress=progress)
        )
    finally:
        cache.close()

    summary = aggregate(scenario, personas, result.rounds, bootstrap=args.bootstrap)
    now = datetime.now()
    meta = {
        "escenario": str(args.scenario),
        "backend": backend.name,
        "modelo": backend.model,
        "versiones_que_respondieron": sorted(backend.served_models),
        "coste_usd": backend.cost,
        "agentes": len(personas),
        "agentes_por_segmento": dict(Counter(p.segment.id for p in personas)),
        "rondas": rounds,
        "semilla": sim.semilla,
        "fecha": now.isoformat(timespec="seconds"),
        "llamadas_api": result.api_calls,
        "respuestas_en_cache": result.cache_hits,
        "fallos": len(result.failures),
    }
    slug = re.sub(r"[^a-z0-9]+", "-", Path(args.scenario).stem.lower()).strip("-")
    out_dir = Path(args.output) if args.output else Path("resultados") / f"{slug}-{now:%Y%m%d-%H%M%S}"
    report = write_outputs(out_dir, scenario, personas, result, summary, meta)

    print(f"\nLlamadas a la API: {result.api_calls} · desde caché: {result.cache_hits} · fallos: {len(result.failures)} · coste: ${backend.cost:.4f}")
    for q in scenario.preguntas:
        stats = summary["final"][q.id]
        best = max(stats["opciones"], key=lambda o: o["probabilidad"])
        print(f"  {q.id}: {best['etiqueta']} ({best['probabilidad'] * 100:.1f}%)")
    print(f"\nInforme: {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="jevfish", description="Simula decisiones de grupos de población.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="valida un escenario .json")
    p.add_argument("scenario", type=Path)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("personas", help="muestra los agentes que se generarían")
    p.add_argument("scenario", type=Path)
    p.add_argument("-n", "--agents", type=int)
    p.add_argument("--show", type=int, default=5)
    p.add_argument("--state", action="store_true", help="muestra el estado completo que recibe el modelo")
    p.set_defaults(func=cmd_personas)

    p = sub.add_parser("simulate", help="ejecuta la simulación")
    p.add_argument("scenario", type=Path)
    p.add_argument("-b", "--backend", choices=list(BACKENDS), default="openrouter")
    p.add_argument("-m", "--model", help="mammoth | elephant | id de un modelo de decisión de OpenRouter")
    p.add_argument("-n", "--agents", type=int)
    p.add_argument("-r", "--rounds", type=int)
    p.add_argument("-c", "--concurrency", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--bootstrap", type=int, default=500, help="remuestreos para el IC 95%% (0 = sin IC)")
    p.add_argument("-o", "--output", help="carpeta de salida")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=cmd_simulate)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ScenarioError as e:
        print(f"Escenario inválido:\n{e}", file=sys.stderr)
    except (BackendError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
