import asyncio
import json
import unittest
from pathlib import Path

from jevfish.aggregate import aggregate, agent_weights
from jevfish.backends import MockBackend, OpenRouterBackend
from jevfish.personas import allocate, generate_personas
from jevfish.scenario import ScenarioError, load_scenario, parse_scenario
from jevfish.simulate import simulate

EXAMPLE = Path(__file__).parent.parent / "escenarios" / "tarifa_transporte.json"


class ScenarioTests(unittest.TestCase):
    def test_example_is_valid(self):
        scenario = load_scenario(EXAMPLE)
        self.assertAlmostEqual(sum(s.peso for s in scenario.segmentos), 1.0)
        self.assertEqual([q.type for q in scenario.preguntas], ["choice", "noul", "choice", "score"])

    def test_errors_are_collected(self):
        with self.assertRaises(ScenarioError) as ctx:
            parse_scenario({"poblacion": {"segmentos": [{"id": "a", "peso": 0}]}, "preguntas": [{"id": "x", "tipo": "raro"}]})
        self.assertGreaterEqual(len(ctx.exception.errors), 4)


class PersonaTests(unittest.TestCase):
    def test_allocation_respects_total_and_minimum(self):
        scenario = load_scenario(EXAMPLE)
        counts = allocate(100, scenario.segmentos, 10)
        self.assertEqual(sum(counts.values()), 100)
        self.assertTrue(all(c >= 10 for c in counts.values()))

    def test_weights_follow_real_population_not_sample(self):
        scenario = load_scenario(EXAMPLE)
        personas = generate_personas(scenario, 100)
        weights = agent_weights(scenario, personas, {p.id for p in personas})
        for segment in scenario.segmentos:
            total = sum(w for pid, w in weights.items() if pid.startswith(segment.id + "-"))
            self.assertAlmostEqual(total, segment.peso)


class PipelineTests(unittest.TestCase):
    def test_mock_simulation_end_to_end(self):
        scenario = load_scenario(EXAMPLE)
        personas = generate_personas(scenario, 40)
        result = asyncio.run(simulate(scenario, personas, MockBackend(), rounds=2))
        self.assertEqual(len(result.rounds), 2)
        summary = aggregate(scenario, personas, result.rounds, bootstrap=50)
        for q in scenario.preguntas:
            options = summary["final"][q.id]["opciones"]
            self.assertAlmostEqual(sum(o["probabilidad"] for o in options), 1.0)
            self.assertAlmostEqual(sum(o["voto"] for o in options), 1.0)
            for o in options:
                self.assertLessEqual(o["ic95"][0], o["probabilidad"] + 1e-9)
                self.assertGreaterEqual(o["ic95"][1], o["probabilidad"] - 1e-9)
        self.assertIn("postura", summary["evolucion"])


class BackendFormatTests(unittest.TestCase):
    def test_payload_matches_decisions_api_schema(self):
        scenario = load_scenario(EXAMPLE)
        payload = OpenRouterBackend(api_key="x").build_payload({"a": 1}, scenario.preguntas, "marco")
        self.assertEqual(payload["model"], "baseten/mammoth-1:free")
        questions = payload["questions"]
        self.assertEqual(questions["ira_a_la_marcha"]["type"], "noul")
        self.assertNotIn("criteria", questions["ira_a_la_marcha"])
        self.assertIsInstance(questions["postura"]["criteria"], dict)
        self.assertIsInstance(questions["impacto_en_voto"]["criteria"], list)

    def test_parses_decisions_response(self):
        scenario = load_scenario(EXAMPLE)
        backend = OpenRouterBackend(api_key="x")
        response = {
            "model": "baseten/mammoth-1-20261021:free",
            "answers": {
                "postura": {"type": "choice", "choice": "apoya", "probabilities": {"apoya": 0.6, "acepta_resignado": 0.4}},
                "ira_a_la_marcha": {"type": "noul", "noul": 0.25},
                "cambio_de_transporte": {"type": "choice", "choice": "sin_cambios"},
                "impacto_en_voto": {"type": "score", "score": 1.8, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.5, "3": 0.2}},
            },
            "usage": {"input_tokens": 400, "output_tokens": 4, "cost": 0},
        }

        async def fake_post(client, url, headers, payload):
            return response

        import jevfish.backends as backends
        original, backends.post_with_retry = backends.post_with_retry, fake_post
        try:
            result = asyncio.run(backend.decide(None, {}, scenario.preguntas, "marco"))
        finally:
            backends.post_with_retry = original
        self.assertAlmostEqual(result["postura"]["apoya"], 0.6)
        self.assertAlmostEqual(result["ira_a_la_marcha"]["si"], 0.25)
        self.assertEqual(result["cambio_de_transporte"]["sin_cambios"], 1.0)
        self.assertAlmostEqual(result["impacto_en_voto"]["2"], 0.5)
        self.assertEqual(backend.served_models, {"baseten/mammoth-1-20261021:free"})


if __name__ == "__main__":
    unittest.main()
