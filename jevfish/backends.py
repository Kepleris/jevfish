"""Backends de inferencia. Todos devuelven, por pregunta, una distribución {opción: probabilidad}."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random

import httpx

from .scenario import Question

Distribution = dict[str, float]

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class BackendError(RuntimeError):
    pass


def normalize(question: Question, probabilities: dict) -> Distribution:
    dist = {}
    for key in question.keys:
        try:
            dist[key] = max(0.0, float(probabilities.get(key, 0.0)))
        except (TypeError, ValueError):
            dist[key] = 0.0
    total = sum(dist.values())
    if total <= 0:
        raise BackendError(f"la pregunta '{question.id}' no recibió probabilidades válidas")
    return {k: v / total for k, v in dist.items()}


async def post_with_retry(client: httpx.AsyncClient, url: str, headers: dict, payload: dict, retries: int = 6) -> dict:
    last_error = ""
    for attempt in range(retries):
        try:
            response = await client.post(url, headers=headers, json=payload)
        except httpx.TransportError as e:
            last_error = f"error de red: {e!r}"
            retry_after = None
        else:
            if response.status_code < 400:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            if response.status_code not in RETRY_STATUS:
                raise BackendError(last_error)
            retry_after = response.headers.get("retry-after")
        try:
            delay = float(retry_after) if retry_after else min(60, 2**attempt)
        except ValueError:
            delay = min(60, 2**attempt)
        await asyncio.sleep(delay + random.random())
    raise BackendError(f"falló tras {retries} intentos: {last_error}")


class Backend:
    name = "base"
    model = ""
    default_concurrency = 4

    def __init__(self):
        self.served_models: set[str] = set()  # versiones exactas que respondieron
        self.cost = 0.0

    @property
    def identity(self) -> str:
        return f"{self.name}:{self.model}"

    async def decide(self, client: httpx.AsyncClient, state: dict, questions: list[Question], framing: str) -> dict[str, Distribution]:
        raise NotImplementedError


class OpenRouterBackend(Backend):
    """Modelos de decisión (mammoth-1, elephant-1, …) vía la Decisions API de OpenRouter."""

    name = "openrouter"
    url = "https://openrouter.ai/api/alpha/decisions"
    default_concurrency = 8
    default_model = "baseten/mammoth-1:free"
    aliases = {
        "mammoth": "baseten/mammoth-1:free",
        "mammoth-1": "baseten/mammoth-1:free",
        "elephant": "baseten/elephant-1:free",
        "elephant-1": "baseten/elephant-1:free",
    }

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise BackendError("falta OPENROUTER_API_KEY (ponla en .env o en el entorno)")
        model = model or os.environ.get("JEVFISH_MODEL") or self.default_model
        self.model = self.aliases.get(model, model)

    def build_payload(self, state: dict, questions: list[Question], framing: str) -> dict:
        payload_questions = {}
        for q in questions:
            item = {"type": q.type, "instructions": f"{framing}\n\n{q.text}"}
            if q.type == "choice":
                item["criteria"] = q.options
            elif q.type == "score":
                item["criteria"] = q.levels
            payload_questions[q.id] = item
        return {"model": self.model, "state": state, "questions": payload_questions}

    async def decide(self, client, state, questions, framing):
        headers = {"Authorization": f"Bearer {self.api_key}", "X-Title": "JevFish"}
        data = await post_with_retry(client, self.url, headers, self.build_payload(state, questions, framing))
        if data.get("model"):
            self.served_models.add(data["model"])
        self.cost += float((data.get("usage") or {}).get("cost") or 0)

        answers = data.get("answers") or {}
        result = {}
        for q in questions:
            answer = answers.get(q.id)
            if not answer or answer.get("type") != q.type:
                raise BackendError(f"respuesta inválida para la pregunta '{q.id}': {answer!r}")
            if q.type == "noul":
                p_yes = float(answer["noul"])
                result[q.id] = normalize(q, {"si": p_yes, "no": 1 - p_yes})
            elif answer.get("probabilities"):
                result[q.id] = normalize(q, answer["probabilities"])
            else:
                # El esquema marca 'probabilities' como opcional; sin él solo sabemos la opción elegida.
                key = answer["choice"] if q.type == "choice" else str(round(float(answer["score"])))
                result[q.id] = normalize(q, {key: 1.0})
        return result


class MockBackend(Backend):
    """Respuestas pseudoaleatorias deterministas. Solo para probar el flujo sin API ni coste."""

    name = "mock"
    model = "mock"
    default_concurrency = 32

    async def decide(self, client, state, questions, framing):
        result = {}
        for q in questions:
            seed = hashlib.sha256(json.dumps([state, q.id], sort_keys=True).encode()).hexdigest()
            rng = random.Random(seed)
            result[q.id] = normalize(q, {k: rng.gammavariate(0.8, 1) for k in q.keys})
        return result


BACKENDS = {"openrouter": OpenRouterBackend, "mock": MockBackend}


def make_backend(name: str, model: str | None = None) -> Backend:
    if name not in BACKENDS:
        raise BackendError(f"backend desconocido '{name}'; opciones: {', '.join(BACKENDS)}")
    if name == "mock":
        return MockBackend()
    return BACKENDS[name](model=model)
