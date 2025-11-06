# Architecture

This SRP module intentionally omits a "core/ports" layer to keep focus on Single Responsibility.

- domain: pure types and config (e.g., `TrainingConfig`, `ModelMetadata`)
- services: concrete implementations with single responsibilities (data, training, storage, tracking)
- pipelines: orchestrate components (sequence), minimal logic

Note: a later module can introduce DIP by adding interfaces (ports) between pipelines/services.
See diagrams in `docs/diagrams`.


