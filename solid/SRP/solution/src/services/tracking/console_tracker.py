"""Console metrics tracker."""

import logging

log = logging.getLogger(__name__)


class ConsoleTracker:
    def log_param(self, key: str, value: object) -> None:
        log.info(f"param {key}={value}")

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if step is None:
            log.info(f"metric {key}={value}")
        else:
            log.info(f"metric {key}={value} step={step}")


