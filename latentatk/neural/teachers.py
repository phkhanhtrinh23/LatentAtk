from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .records import LABEL_FIELDS, HybridRecord


PROMPT_VERSION = "hybrid-plan-v1"
SYSTEM_PROMPT = (
    "Plan a synthetic recommender robustness evaluation. Return JSON with role_mix "
    "[PR,CA,SN,IN], topology_probs [star,chain,forest,broadcast], topology index 0..3, "
    "depth 1..6, five bridge_weights, and four nonempty role_traces in order "
    "[PR,CA,SN,CO]. Probabilities must be nonnegative and sum to one. "
    "CO coordinates the campaign including the inactive IN share. Use only the supplied observation."
)


def load_env(path=".env"):
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.removeprefix("export ").partition("=")
        if not sep or not key.strip().isidentifier():
            raise ValueError(f"invalid environment assignment in {path}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def resolve_teacher(spec):
    resolved = dict(spec)
    if resolved.get("provider", "scripted") == "openrouter":
        model = resolved.get("model") or os.environ.get(resolved.get("model_env", ""))
        if not model or "/" not in model or model.endswith(":latest") or model == "openrouter/auto":
            raise ValueError(f"{resolved['name']}: configure an explicit versioned OpenRouter model ID")
        resolved["model"] = model
    return resolved


def label_schema():
    def array(size, kind):
        return {"type": "array", "items": {"type": kind}, "minItems": size, "maxItems": size}

    properties = {
        "role_mix": array(4, "number"), "topology_probs": array(4, "number"),
        "topology": {"type": "integer", "minimum": 0, "maximum": 3},
        "depth": {"type": "integer", "minimum": 1, "maximum": 6},
        "bridge_weights": array(5, "number"), "role_traces": array(4, "string"),
    }
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


class OpenRouterTeacher:
    def __init__(self, spec):
        self.spec = resolve_teacher(spec)
        self.model = self.spec["model"]
        self.api_key = os.environ.get(self.spec.get("api_key_env", "OPENROUTER_API_KEY"))
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required to generate API teacher labels")
        self.base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        catalogue = self._request("/models")["data"]
        entries = [entry for entry in catalogue if entry["id"] == self.model]
        if len(entries) != 1:
            raise ValueError(f"teacher {self.spec['name']}: OpenRouter model unavailable: {self.model}")
        self.catalogue_entry = entries[0]
        if "structured_outputs" not in entries[0].get("supported_parameters", []):
            raise ValueError(f"{self.model} does not advertise structured_outputs support")

    def _request(self, endpoint, payload=None):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for env, header in (("OPENROUTER_SITE_URL", "HTTP-Referer"),
                            ("OPENROUTER_APP_NAME", "X-OpenRouter-Title")):
            if os.environ.get(env):
                headers[header] = os.environ[env]
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = Request(self.base_url + endpoint, data=data, headers=headers)
        with urlopen(request, timeout=float(self.spec.get("timeout_s", 120))) as response:
            return json.load(response)

    def label(self, record, seed, audit_path):
        provider = {"allow_fallbacks": False, "require_parameters": True}
        if self.spec.get("endpoint_provider"):
            provider["only"] = [self.spec["endpoint_provider"]]
        payload = {
            "model": self.model, "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": record.prompt()}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "hybrid_plan", "strict": True, "schema": label_schema()}},
            "provider": provider, "stream": False,
        }
        supported = set(self.catalogue_entry.get("supported_parameters", []))
        if "seed" in supported:
            payload["seed"] = int(seed)
        for name, default, cast in (("temperature", 0.7, float), ("max_tokens", 2048, int)):
            if name in self.spec and name not in supported:
                raise ValueError(f"{self.model} does not advertise requested parameter {name}")
            if name in supported:
                payload[name] = cast(self.spec.get(name, default))
        max_retries = int(self.spec.get("max_retries", 3))
        for retry in range(max_retries + 1):
            response, error, result = None, None, None
            started = time.perf_counter()
            permanent = False
            try:
                response = self._request("/chat/completions", payload)
                if response.get("error"):
                    raise ValueError(f"OpenRouter error: {response['error']}")
                if response.get("model") != self.model:
                    raise ValueError(f"requested {self.model}, received {response.get('model')}")
                if "seed" not in payload and not response.get("id"):
                    raise ValueError("unseeded teacher response requires a request ID")
                choice = response["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError(f"incomplete teacher response: {choice.get('finish_reason')}")
                labels = json.loads(choice["message"]["content"])
                if set(labels) != set(LABEL_FIELDS):
                    raise ValueError("teacher response must provide exactly all six label fields")
                usage = response.get("usage")
                metadata = {
                    "provider": "openrouter", "source_id": self.spec["name"], "model": self.model,
                    "revision": self.spec.get("revision"),
                    "revision_reason": None if self.spec.get("revision") else "provider exposes model ID only",
                    "actual_model": response.get("model"),
                    "catalogue_model": {key: self.catalogue_entry.get(key) for key in
                                        ("id", "canonical_slug", "created", "name")},
                    "served_provider": response.get("provider"), "request_id": response.get("id"),
                    "seed": payload.get("seed"), "requested_seed": int(seed),
                    "seed_reason": None if "seed" in payload else "model does not advertise seed support",
                    "prompt_version": PROMPT_VERSION, "retry_count": retry,
                    "prompt": payload["messages"], "response": response,
                    "usage": usage, "usage_reason": None if usage else "not returned by provider",
                    "cost_usd": (usage or {}).get("cost"),
                    "cost_reason": None if (usage or {}).get("cost") is not None else "not returned by provider",
                }
                result = replace(record, **labels, teacher_metadata=metadata)
                result.validate()
            except HTTPError as exc:
                error = f"HTTP {exc.code}: {exc.read().decode(errors='replace')}"
                permanent = exc.code not in {408, 409, 429} and exc.code < 500
            except (URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError) as exc:
                error = str(exc)
            append_json(audit_path, {
                "qid": record.qid, "environment_id": record.environment_id,
                "prompt_version": PROMPT_VERSION, "request": payload, "response": response,
                "requested_seed": int(seed),
                "retry_count": retry, "error": error, "elapsed_s": time.perf_counter() - started,
            })
            if result is not None and error is None:
                return result
            if permanent or retry == max_retries:
                raise RuntimeError(f"{record.qid}: teacher generation failed: {error}")
            time.sleep(min(2 ** retry, 30))
