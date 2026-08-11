from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class FlowConfig(BaseModel):
    id: str
    name: str
    description: str = ""
    knowledge_paths: list[Path] = Field(default_factory=list)
    system_prompt: str
    aliases: dict[str, list[str]] = Field(default_factory=dict)
    max_search_iterations: int = 2


class FlowRegistry:
    def __init__(self, directory: Path):
        self.directory = directory
        self._flows: dict[str, FlowConfig] = {}

    def load(self) -> None:
        self._flows.clear()
        for path in sorted(self.directory.glob("*/config.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            flow = FlowConfig.model_validate(raw)
            flow.knowledge_paths = [
                candidate if candidate.is_absolute() else (self.directory.parent / candidate).resolve()
                for candidate in flow.knowledge_paths
            ]
            self._flows[flow.id] = flow

    def get(self, flow_id: str) -> FlowConfig:
        try:
            return self._flows[flow_id]
        except KeyError as error:
            raise ValueError(f"Fluxo desconhecido: {flow_id}") from error

    def all(self) -> list[FlowConfig]:
        return list(self._flows.values())
