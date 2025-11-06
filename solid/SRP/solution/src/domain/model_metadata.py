"""Model metadata."""

from dataclasses import dataclass


@dataclass
class ModelMetadata:
    artifact_uri: str
    metrics: dict


