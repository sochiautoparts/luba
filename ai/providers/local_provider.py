"""Local LLM Provider — RuadaptQwen3-4B via llama-cpp-python (GGUF).

Same model & approach as Asya bot:
  - RuadaptQwen3-4B-Instruct Q4_K_M (~2.3GB)
  - CPU inference (GitHub Actions compatible, OpenBLAS)
  - ChatML template (Qwen3 instruct, non-thinking variant)
  - asyncio.Lock to prevent segfaults on concurrent llama-cpp calls
  - Graceful degradation: if model missing/unloadable, returns error → router falls back to cloud

Local model is PRIMARY for chat and group comments (saves cloud balance, protects privacy
in groups — group messages never leave the machine when local model answers).
"""

import asyncio
import logging
import os
import time
from typing import Optional, List, Dict

from ai.providers.base import BaseAIProvider, AIResponse
from bot.config import config

logger = logging.getLogger("luba.ai.local")

# ── Qwen3 ChatML template tokens ──
QWEN3_SYSTEM_START = "<|im_start|>system\n"
QWEN3_USER_START = "<|im_start|>user\n"
QWEN3_ASSISTANT_START = "<|im_start|>assistant\n"
QWEN3_END = "<|im_end|>\n"


class LocalProvider(BaseAIProvider):
    """llama-cpp-python local inference."""

    name = "local"

    def __init__(self):
        super().__init__()
        self._llm = None
        self._model_loaded = False
        self._lock = asyncio.Lock()
        self._load_attempted = False
        # ── Anti-segfault generation guard (ported from Asya) ──
        # llama-cpp's C object is NOT thread-safe: concurrent llama() calls
        # from different asyncio tasks (e.g. after a wait_for timeout cancels
        # the holding task but the thread-pool thread keeps running) cause
        # GGML_ASSERT failures and segmentation faults (exit 139).
        # The flag + Event ensure only ONE generation runs at a time, and a
        # cancelled caller waits for the in-flight thread to finish before
        # returning, so the next generation starts on a quiescent model.
        self._generating = False
        self._generation_done = asyncio.Event()
        self._generation_done.set()  # initially idle

    async def initialize(self) -> bool:
        """Try to load the local model. Returns True if loaded."""
        if self._load_attempted:
            return self._model_loaded
        self._load_attempted = True

        if not config.ENABLE_LOCAL_MODEL:
            logger.info("Local model disabled (ENABLE_LOCAL_MODEL=false)")
            return False

        # Auto-download if configured and missing
        model_path = config.MODEL_PATH
        if not os.path.exists(model_path):
            if config.MODEL_AUTO_DOWNLOAD:
                ok = await self._download_model()
                if not ok:
                    return False
            else:
                logger.warning(f"Model not found at {model_path} and auto-download disabled")
                return False

        try:
            # Import here so module imports cleanly even if llama-cpp-python not installed
            from llama_cpp import Llama
        except ImportError as e:
            logger.error(f"llama-cpp-python not installed: {e}")
            return False

        try:
            logger.info(f"Loading local model: {model_path}")
            t0 = time.time()
            self._llm = Llama(
                model_path=model_path,
                n_ctx=config.MODEL_N_CTX,
                n_threads=config.MODEL_N_THREADS,
                n_gpu_layers=0,  # CPU only
                verbose=False,
                use_mlock=False,
                use_mmap=True,
            )
            self._model_loaded = True
            logger.info(f"Local model loaded in {time.time()-t0:.1f}s "
                        f"(ctx={config.MODEL_N_CTX}, threads={config.MODEL_N_THREADS})")
            return True
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            self._llm = None
            self._model_loaded = False
            return False

    async def _download_model(self) -> bool:
        """Download the GGUF model file (with HF_TOKEN if available)."""
        model_path = config.MODEL_PATH
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        url = config.MODEL_DOWNLOAD_URL
        headers = {}
        if config.HF_TOKEN:
            headers["Authorization"] = f"Bearer {config.HF_TOKEN}"

        logger.info(f"Downloading local model from {url} (~2.3GB)…")
        try:
            # Prefer huggingface_hub for resumable download
            try:
                from huggingface_hub import hf_hub_download
                # Derive repo_id & filename from URL
                # https://huggingface.co/<repo>/resolve/main/<file>
                import re
                m = re.match(r"https?://huggingface\.co/([^/]+)/([^/]+)/resolve/[^/]+/(.+)", url)
                if m:
                    repo_id = f"{m.group(1)}/{m.group(2)}"
                    filename = m.group(3)
                    token = config.HF_TOKEN or None
                    path = hf_hub_download(repo_id=repo_id, filename=filename,
                                           local_dir=os.path.dirname(model_path) or ".",
                                           token=token)
                    # Rename to expected name if different
                    if path and path != model_path and not os.path.exists(model_path):
                        os.replace(path, model_path)
                    if os.path.exists(model_path):
                        size_mb = os.path.getsize(model_path) // (1024 * 1024)
                        logger.info(f"Model downloaded via hf_hub: {size_mb} MB")
                        return True
            except Exception as e:
                logger.warning(f"hf_hub download failed: {e}, trying curl")

            # Fallback: curl
            import subprocess
            cmd = ["curl", "-L", "--retry", "3", "--max-time", "900"]
            if headers:
                for k, v in headers.items():
                    cmd += ["-H", f"{k}: {v}"]
            cmd += ["-o", model_path, url]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) // (1024 * 1024)
                if size_mb > 100:
                    logger.info(f"Model downloaded via curl: {size_mb} MB")
                    return True
            logger.error(f"curl download failed: {r.stderr[:300]}")
            return False
        except Exception as e:
            logger.error(f"Model download exception: {e}")
            return False

    async def is_available(self) -> bool:
        if not self._model_loaded:
            if not self._load_attempted:
                await self.initialize()
        return self._model_loaded and self._llm is not None

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Render ChatML prompt for Qwen3 instruct."""
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"{QWEN3_SYSTEM_START}{content}{QWEN3_END}")
            elif role == "user":
                parts.append(f"{QWEN3_USER_START}{content}{QWEN3_END}")
            elif role == "assistant":
                parts.append(f"{QWEN3_ASSISTANT_START}{content}{QWEN3_END}")
        parts.append(QWEN3_ASSISTANT_START)
        return "".join(parts)

    async def chat(self, messages: List[Dict[str, str]],
                   temperature: float = 0.8, max_tokens: int = 512) -> AIResponse:
        if not await self.is_available():
            return self._fail("local model not available")

        # Circuit breaker: too many recent errors → skip local for a bit
        if self._consecutive_errors >= 5:
            return self._fail("local circuit breaker open (5+ consecutive errors)")

        # ── Wait for any in-flight generation to finish (prevents segfault) ──
        # A previous caller may have been cancelled by wait_for but its
        # thread-pool thread is still running llama(). We must NOT start a
        # new generation until that thread completes.
        await self._generation_done.wait()

        # Trim context to fit budget: keep system + last few turns
        sys_msgs = [m for m in messages if m["role"] == "system"]
        convo = [m for m in messages if m["role"] != "system"]
        convo = convo[-6:]
        trimmed = sys_msgs + convo
        for m in trimmed:
            if len(m["content"]) > 1200:
                m["content"] = m["content"][:1200]

        prompt = self._format_messages(trimmed)

        async with self._lock:
            # Mark generation as in-flight and clear the "done" event
            self._generating = True
            self._generation_done.clear()
            try:
                loop = asyncio.get_running_loop()
                # run_in_executor + shield: even if the caller is cancelled
                # (e.g. by asyncio.wait_for timeout), the C thread runs to
                # completion. We then mark "done" so the next caller can start.
                fut = loop.run_in_executor(
                    None, self._generate, prompt, temperature, max_tokens
                )
                cancelled = False
                while True:
                    try:
                        result = await asyncio.shield(fut)
                        break
                    except asyncio.CancelledError:
                        # Caller cancelled — but we MUST wait for the C thread
                        # to finish before allowing a new generation, otherwise
                        # the next llama() call segfaults.
                        if not cancelled:
                            cancelled = True
                            logger.warning(
                                "Local generation cancelled — waiting for "
                                "thread to complete safely (preventing segfault)"
                            )
                        try:
                            result = await asyncio.wait_for(asyncio.shield(fut), timeout=60.0)
                            break
                        except asyncio.TimeoutError:
                            # Thread is stuck — give up rather than hang forever
                            logger.error("Local generation thread stuck >60s after cancel")
                            return self._fail("generation thread stuck after cancel")
                return result
            except Exception as e:
                logger.error(f"Local generation error: {e}")
                return self._fail(f"generation error: {e}")
            finally:
                # ALWAYS release the generation slot, even on error/cancel
                self._generating = False
                self._generation_done.set()

    def _generate(self, prompt: str, temperature: float, max_tokens: int) -> AIResponse:
        try:
            out = self._llm(
                prompt,
                max_tokens=min(max_tokens, config.MODEL_MAX_TOKENS),
                temperature=temperature,
                top_p=0.9,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False,
            )
            text = ""
            if isinstance(out, dict):
                choices = out.get("choices", [])
                if choices:
                    text = choices[0].get("text", "")
            text = (text or "").strip()
            if not text:
                return self._fail("empty generation")
            # Strip stray ChatML tokens
            for tok in ["<|im_end|>", "<|im_start|>"]:
                text = text.replace(tok, "")
            return self._ok(text.strip(), model="ruadapt-qwen3-4b")
        except Exception as e:
            return self._fail(f"llama error: {e}")

    async def close(self) -> None:
        self._llm = None
        self._model_loaded = False
