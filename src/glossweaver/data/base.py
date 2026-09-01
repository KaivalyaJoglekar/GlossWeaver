from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CanonicalRecord:
    id: str
    dataset: str
    gloss: str
    target: str
    split: str
    source_partition: str
    audio_filepath: str | None = None
    grammar_labels: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

