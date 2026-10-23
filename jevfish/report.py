"""Escritura de resultados: resultados.json, agentes.jsonl y reporte.md."""

from __future__ import annotations

import json
from pathlib import Path

from .personas import Persona
from .scenario import Scenario
from .simulate import SimulationResult


def _bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(scenario: Scenario, summary: dict, meta: dict) -> str:
    lines = [
        f"# {scenario.tema}",
        "",
        f"- Backend: `{meta['backend']}` · modelo: `{meta['modelo']}`"
        + (f" ({', '.join(meta['versiones_que_respondieron'])})" if meta.get("versiones_que_respondieron") else ""),
        f"- Agentes: {meta['agentes']} · rondas: {meta['rondas']} · semilla: {meta['semilla']}",
        f"- Fecha: {meta['fecha']}",
        "",
        "## Población",
        "",
        "| Segmento | Peso real | Agentes simulados |",
        "|---|---:|---:|",
    ]
    for s in scenario.segmentos:
        lines.append(f"| {s.nombre} | {_pct(s.peso)} | {meta['agentes_por_segmento'].get(s.id, 0)} |")

    final = summary["final"]
    for q in scenario.preguntas:
        stats = final[q.id]
        lines += ["", f"## {q.text}", ""]
        lines += [
            "| Opción | Probabilidad ponderada | IC 95% | Agentes que la prefieren | |",
            "|---|---:|---:|---:|---|",
        ]
        for o in sorted(stats["opciones"], key=lambda o: o["probabilidad"], reverse=True):
            ci = f"{_pct(o['ic95'][0])} – {_pct(o['ic95'][1])}" if o["ic95"] else "—"
            lines.append(
                f"| {o['etiqueta']} | {_pct(o['probabilidad'])} | {ci} | {_pct(o['voto'])} | `{_bar(o['probabilidad'])}` |"
            )
        if q.type == "score":
            lines.append(f"\nNivel esperado: **{stats['nivel_esperado']:.2f}** (escala 0 – {len(q.levels) - 1})")
        lines.append(f"\nIndecisión media de los agentes: {_pct(stats['indecision_media'])}")

        header = [q.label(k) for k in q.keys]
        lines += ["", "**Por segmento**", "", "| Segmento | " + " | ".join(header) + " |"]
        lines.append("|---|" + "---:|" * len(header))
        for seg in stats["por_segmento"].values():
            cells = " | ".join(_pct(seg["probabilidades"][k]) for k in q.keys)
            lines.append(f"| {seg['nombre']} | {cells} |")

        if q.id in summary["evolucion"]:
            changes = ", ".join(
                f"{q.label(k)} {delta * 100:+.1f} pp" for k, delta in summary["evolucion"][q.id].items()
            )
            lines.append(f"\nCambio entre la ronda 1 y la {meta['rondas']} (influencia social): {changes}")

    if meta["fallos"]:
        lines += ["", f"> {meta['fallos']} respuestas fallaron y se excluyeron (ver resultados.json)."]
    lines += [
        "",
        "---",
        "",
        "*Simulación con agentes sintéticos, no una encuesta. La «probabilidad ponderada» es la "
        "media, ponderada por el peso real de cada segmento, de la probabilidad que el modelo asigna "
        "a que cada agente elija esa opción. Los resultados dependen de los supuestos del escenario.*",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    out_dir: Path,
    scenario: Scenario,
    personas: list[Persona],
    result: SimulationResult,
    summary: dict,
    meta: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "resumen": summary,
        "fallos": [{"ronda": r, "agente": a, "error": e} for r, a, e in result.failures],
    }
    (out_dir / "resultados.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "agentes.jsonl").open("w", encoding="utf-8") as f:
        for p in personas:
            record = {
                "id": p.id,
                "segmento": p.segment.id,
                "atributos": p.attrs,
                "vecinos": result.network.get(p.id, []),
                "respuestas_por_ronda": [answers.get(p.id) for answers in result.rounds],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = out_dir / "reporte.md"
    report.write_text(render_markdown(scenario, summary, meta), encoding="utf-8")
    return report
