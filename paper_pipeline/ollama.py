"""
ollama.py — Ollama VRAM manager and streaming generation client.

Two classes, two jobs:
  OllamaVRAM   — Ensures mutually exclusive models don't collide in GPU memory.
  OllamaClient — Streaming /api/generate with shutdown-aware interruption.

Thread-safety: a reentrant lock serialises model transitions so that
concurrent workers (--workers N) cannot interleave ensure_ready/generate
across different model tiers.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from threading import Event, RLock
from typing import Optional

import requests

from . import config
from .errors import EmptyGenerationError, OllamaUnavailable, ShutdownRequested


# ── Exceptions ───────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# VRAM MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

class OllamaVRAM:
    """Manages GPU residency for mutually exclusive large models.

    The key insight: models that together exceed VRAM cannot coexist.
    Rather than letting Ollama fail with cudaMalloc → evict-all → retry
    (which produces crash logs and core dumps), we proactively evict
    the outgoing model before loading the incoming one.

    Thread-safety: all public methods acquire self._lock so that
    concurrent workers cannot interleave model transitions.
    """

    def __init__(
        self,
        base_url: str = config.OLLAMA_URL,
        exclusive: set[str] | None = None,
    ):
        self.url = base_url
        self.exclusive = exclusive if exclusive is not None else config.EXCLUSIVE_MODELS
        self._lock = RLock()

    @property
    def lock(self) -> RLock:
        """Expose the lock so OllamaClient can hold it across ensure_ready + generate."""
        return self._lock

    # ── Public interface ─────────────────────────────────────────────────

    def ensure_ready(self, model: str) -> None:
        """Evict any co-resident exclusive model before loading `model`.

        Caller must already hold self._lock (OllamaClient does this).
        """
        for loaded in self._get_loaded():
            if loaded != model and loaded in self.exclusive:
                print(f"     🔁  Evicting {loaded} to make room for {model} …")
                self._evict(loaded)

    def provision(self, verbose: bool = False) -> bool:
        """Aggressively clear all loaded models.  Escalates to service restart.

        Returns True when Ollama is up and VRAM appears free.
        """
        with self._lock:
            print("\n  🎯  Provisioning: clearing GPU VRAM …")

            loaded = self._get_loaded()
            if not loaded:
                print("  ✓   No models loaded — GPU is free")
                return True

            # Level 1: graceful eviction
            print(f"  ⚡  Evicting {len(loaded)} model(s): {', '.join(loaded)}")
            for model in loaded:
                self._evict(model)

            if self._wait_clean(timeout=config.EVICT_TIMEOUT):
                print("  ✅  All models evicted — GPU is free\n")
                return True

            # Level 2: service restart
            remaining = self._get_loaded()
            print(f"  ⚠️   {len(remaining)} model(s) still loaded — restarting service")
            if not self._restart_service():
                print("  ⚠️   Restart failed — proceeding anyway\n")
                return False

            if self._wait_clean(timeout=15):
                print("  ✅  Ollama restarted clean\n")
                return True

            print("  ⚠️   Could not confirm clean VRAM — proceeding anyway\n")
            return False

    def restart_service(self) -> bool:
        """Thread-safe wrapper for _restart_service."""
        with self._lock:
            return self._restart_service()

    def force_reload(self, model: str) -> bool:
        """Evict `model` so the next generate call loads it fresh.

        Used to recover from a suspected corrupted context/session (e.g.
        an empty generation), not for routine VRAM management — ensure_ready
        skips eviction when `model` is already the resident model, since
        normally that's exactly the model we want to keep.
        """
        with self._lock:
            return self._evict(model)

    # ── Internals ────────────────────────────────────────────────────────

    def _restart_service(self) -> bool:
        """Restart the ollama systemd service and wait for it to respond."""
        print("  🔄  Restarting ollama service …")
        try:
            r = subprocess.run(
                ["sudo", "systemctl", "restart", "ollama"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                print(f"  ⚠️   systemctl restart failed: {r.stderr.strip()}")
                return False
        except Exception as exc:
            print(f"  ❌  Could not restart ollama: {exc}")
            return False

        print("  ⏳  Waiting for Ollama …", end="", flush=True)
        deadline = time.time() + config.SERVICE_RESTART_TIMEOUT
        while time.time() < deadline:
            time.sleep(2)
            try:
                r = requests.get(f"{self.url}/api/tags", timeout=3)
                if r.status_code == 200:
                    print(" up!")
                    return True
            except Exception:
                pass
            print(".", end="", flush=True)
        print(" timed out")
        return False

    def _get_loaded(self) -> list[str]:
        try:
            r = requests.get(f"{self.url}/api/ps", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def _evict(self, model: str) -> bool:
        try:
            r = requests.post(
                f"{self.url}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=config.EVICT_TIMEOUT,
            )
            return r.status_code == 200
        except Exception:
            return False

    def _wait_clean(self, timeout: int = 30, interval: float = 2.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._get_loaded():
                return True
            time.sleep(interval)
        return False


@dataclass
class GenerationResult:
    """Text plus the terminal metadata from one /api/generate stream.

    Kept even when `text` is empty — `done_reason`, `thinking_len`, and the
    token/duration counters are exactly what's needed to tell "the model
    burned its budget on hidden thinking" apart from "context checkpoint
    came back corrupted" apart from "the model just stopped immediately",
    without having to reproduce the failure by hand.
    """
    text: str
    thinking_len: int
    done_reason: str | None
    eval_count: int | None
    eval_duration: int | None
    prompt_eval_count: int | None
    prompt_eval_duration: int | None
    total_duration: int | None


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class OllamaClient:
    """Streaming Ollama /api/generate with shutdown-aware interruption.

    Thread-safety: generate() holds the VRAM lock across the full
    ensure_ready → stream cycle, so concurrent workers cannot evict
    each other's models mid-generation.
    """

    def __init__(
        self,
        vram: OllamaVRAM,
        shutdown: Event,
        ctx_tokens: int = config.DEFAULT_CTX_TOKENS,
    ):
        self.vram = vram
        self.shutdown = shutdown
        self.ctx_tokens = ctx_tokens

    def generate(
        self,
        model: str,
        prompt: str,
        ctx_tokens: int | None = None,
    ) -> str:
        """Send a prompt and return the full response text.

        Raises:
            ShutdownRequested: if the shutdown event fires at any point.
            OllamaUnavailable: if Ollama cannot be reached.
            RuntimeError: on timeout after retry, or other Ollama errors.

        Never returns empty string as a cancellation sentinel.
        """
        if self.shutdown.is_set():
            raise ShutdownRequested()

        ctx = ctx_tokens if ctx_tokens is not None else self.ctx_tokens
        options: dict = {
            "num_ctx": ctx,
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
            "repeat_penalty": config.REPEAT_PENALTY,
        }
        if model in config.MODEL_GPU_LAYERS:
            options["num_gpu"] = config.MODEL_GPU_LAYERS[model]

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": options,
        }

        url = f"{self.vram.url}/api/generate"

        for attempt in range(1, 3):
            if self.shutdown.is_set():
                raise ShutdownRequested()
            try:
                # Acquire the VRAM lock with a poll loop so shutdown can
                # interrupt a worker waiting behind a long generation.
                self._acquire_lock_or_shutdown()
                try:
                    # Recheck after acquiring — another worker may have set
                    # shutdown while we were waiting.
                    if self.shutdown.is_set():
                        raise ShutdownRequested()
                    self.vram.ensure_ready(model)
                    result = self._stream(url, payload)
                finally:
                    self.vram.lock.release()

                # Validate: a successful Ollama response with no text is a
                # protocol-level problem, not a valid empty answer.
                if not result.text:
                    self._log_empty_generation(model, result)
                    if attempt == 1 and not self.shutdown.is_set():
                        print(
                            f"     🔁  Empty generation — forcing clean reload "
                            f"and retrying (model={model}) …"
                        )
                        self.vram.force_reload(model)
                        continue
                    raise EmptyGenerationError(
                        f"Ollama returned an empty response after reload retry "
                        f"(model={model}, done_reason={result.done_reason}, "
                        f"thinking_chars={result.thinking_len}, "
                        f"eval_count={result.eval_count})"
                    )
                return result.text

            except requests.exceptions.ConnectionError as exc:
                raise OllamaUnavailable(
                    f"Cannot reach Ollama at {self.vram.url} — is `ollama serve` running?"
                ) from exc
            except requests.exceptions.Timeout:
                if attempt == 1 and not self.shutdown.is_set():
                    print(f"     ⏱  Timeout — restarting Ollama and retrying (model={model}) …")
                    self.vram.restart_service()
                    continue
                raise RuntimeError(f"Ollama timed out after retry (model={model})")
            except ShutdownRequested:
                raise
            except EmptyGenerationError:
                raise
            except Exception as exc:
                raise RuntimeError(f"Ollama error (model={model}): {exc}") from exc

        raise RuntimeError(f"Ollama generate exhausted retries (model={model})")

    def _acquire_lock_or_shutdown(self) -> None:
        """Acquire the VRAM lock, polling so shutdown can interrupt the wait.

        Raises ShutdownRequested if the event fires while waiting.
        On success, the caller is responsible for releasing the lock.
        """
        while not self.vram.lock.acquire(timeout=0.5):
            if self.shutdown.is_set():
                raise ShutdownRequested()

    def _stream(self, url: str, payload: dict) -> GenerationResult:
        """Stream tokens from Ollama.  Caller must hold the VRAM lock.

        Collects `response` text as the answer and counts `thinking`
        characters separately without ever treating them as the answer —
        a model that spends its whole context budget "thinking" and never
        reaches a response must still surface as an empty result, not a
        silently truncated one.
        """
        with requests.post(url, json=payload, timeout=config.GENERATE_TIMEOUT, stream=True) as r:
            if not r.ok:
                try:
                    body = r.json().get("error", r.text[:200])
                except Exception:
                    body = r.text[:200]
                raise RuntimeError(f"Ollama HTTP {r.status_code}: {body}")

            chunks: list[str] = []
            thinking_len = 0
            done_meta: dict = {}
            for line in r.iter_lines():
                if self.shutdown.is_set():
                    raise ShutdownRequested()
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        chunks.append(data["response"])
                    if data.get("thinking"):
                        thinking_len += len(data["thinking"])
                    if data.get("done"):
                        done_meta = data
                        break

            return GenerationResult(
                text="".join(chunks).strip(),
                thinking_len=thinking_len,
                done_reason=done_meta.get("done_reason"),
                eval_count=done_meta.get("eval_count"),
                eval_duration=done_meta.get("eval_duration"),
                prompt_eval_count=done_meta.get("prompt_eval_count"),
                prompt_eval_duration=done_meta.get("prompt_eval_duration"),
                total_duration=done_meta.get("total_duration"),
            )

    @staticmethod
    def _log_empty_generation(model: str, result: GenerationResult) -> None:
        eval_ms = (result.eval_duration or 0) // 1_000_000
        print(
            f"     ⚠️   Empty response (model={model}) — "
            f"done_reason={result.done_reason} thinking_chars={result.thinking_len} "
            f"eval_count={result.eval_count} eval_ms={eval_ms} "
            f"prompt_tokens={result.prompt_eval_count}"
        )

    def health_check(self) -> bool:
        """Verify Ollama is reachable and report available model count."""
        try:
            r = requests.get(
                f"{self.vram.url}/api/tags",
                timeout=config.HEALTH_CHECK_TIMEOUT,
            )
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"  🟢  Ollama reachable — {len(models)} models available")
            return True
        except Exception as exc:
            print(f"  ❌  Cannot reach Ollama at {self.vram.url}: {exc}")
            return False

    def check_required_models(self, models: list[str]) -> None:
        """Raise OllamaUnavailable if any model is missing locally."""
        try:
            r = requests.get(
                f"{self.vram.url}/api/tags",
                timeout=config.HEALTH_CHECK_TIMEOUT * 2,
            )
            r.raise_for_status()
            available = {m["name"] for m in r.json().get("models", [])}
        except Exception:
            return  # health_check already reported the problem
        missing = [m for m in models if m not in available]
        if missing:
            raise OllamaUnavailable(
                "Required models not found — pull them first:\n"
                + "\n".join(f"    ollama pull {m}" for m in missing)
            )
