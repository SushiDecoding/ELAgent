"""Configuration loading: defaults <- ~/.elagent/config.json <- env <- CLI."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def default_home() -> Path:
    override = os.environ.get("ELAGENT_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".elagent"


@dataclass
class Config:
    # --- endpoint -------------------------------------------------------
    base_url: str = "http://127.0.0.1:8138/v1"
    api_key: str = "sk-local"
    model: str = "auto"          # "auto" -> resolved from /v1/models
    # Generation parameters belong to llama.cpp.  When this is False (default)
    # ELAgent sends none of temperature/top_p/max_tokens and never changes them
    # at runtime; the server launch flags decide.  `max_tokens` stays here only
    # as a local context-budget reserve.
    send_generation_params: bool = False
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 2048
    request_timeout: float = 300.0
    max_retries: int = 3
    stream: bool = True
    # Local safety cap on streamed characters (0 = unlimited).  This never
    # changes a server parameter; it stops the client from reading a runaway
    # generation (reasoning models can loop for minutes without --n-predict).
    max_stream_chars: int = 8000

    # --- context --------------------------------------------------------
    context_limit: int = 32768
    compact_threshold: int = 12000
    keep_recent_messages: int = 8
    tool_output_limit: int = 6000

    # --- loop -----------------------------------------------------------
    max_steps: int = 24          # tool calls per round
    max_rounds: int = -1         # -1 = unlimited goal rounds
    judge_retries: int = 3
    judge_max_tokens: int = 700
    max_invalid_actions: int = 3
    max_repeat_actions: int = 3   # same tool+args: 3rd occurrence stops the round

    # --- sub-agents ------------------------------------------------------
    subagent_max_steps: int = 8    # steps per sub-agent run
    subagent_max_depth: int = 1    # 1 = a sub-agent cannot spawn another
    workflow_max_steps: int = 4    # sequential steps in one workflow call

    # --- tools ----------------------------------------------------------
    shell: str = "auto"          # auto | bash | cmd | powershell
    allow_outside_workspace: bool = False
    confirm_dangerous: bool = True
    yolo: bool = False

    # --- memory ---------------------------------------------------------
    memory_enabled: bool = True
    memory_auto_recall: bool = True
    memory_max_entries: int = 3
    memory_on_complete: bool = False   # only /memory save writes经验

    # --- ui -------------------------------------------------------------
    theme: str = "dark"
    verbose: bool = False
    workspace: str = field(default_factory=lambda: os.getcwd())

    # ------------------------------------------------------------------
    @property
    def home(self) -> Path:
        return default_home()

    @property
    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    @property
    def memory_file(self) -> Path:
        return self.home / "global.memory"

    @property
    def config_file(self) -> Path:
        return self.home / "config.json"

    def ensure_dirs(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def to_json(self) -> dict:
        data = asdict(self)
        data.pop("workspace", None)
        return data

    def save(self) -> None:
        self.ensure_dirs()
        tmp = self.config_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.config_file)

    @classmethod
    def load(cls, path: Path | None = None, workspace: str | None = None) -> "Config":
        cfg = cls()
        if workspace:
            cfg.workspace = workspace
        target = path or cfg.config_file
        if target.exists():
            try:
                raw = json.loads(target.read_text(encoding="utf-8"))
                known = {f.name for f in fields(cls)}
                for key, value in raw.items():
                    if key in known and value is not None:
                        setattr(cfg, key, value)
            except Exception:
                # A corrupt config must never prevent startup.
                pass
        if os.environ.get("ELAGENT_BASE_URL"):
            cfg.base_url = os.environ["ELAGENT_BASE_URL"]
        if os.environ.get("ELAGENT_MODEL"):
            cfg.model = os.environ["ELAGENT_MODEL"]
        if os.environ.get("ELAGENT_API_KEY"):
            cfg.api_key = os.environ["ELAGENT_API_KEY"]
        if os.environ.get("ELAGENT_MAX_ROUNDS"):
            try:
                cfg.max_rounds = int(os.environ["ELAGENT_MAX_ROUNDS"])
            except ValueError:
                pass
        return cfg


# Fields that the CLI is allowed to override from argv.
OVERRIDABLE = (
    "base_url", "api_key", "model", "max_rounds", "max_steps",
    "temperature", "max_tokens", "yolo", "verbose", "shell",
)
