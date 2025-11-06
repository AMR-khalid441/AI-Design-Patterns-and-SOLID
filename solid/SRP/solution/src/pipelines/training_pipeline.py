"""Training pipeline orchestrator."""

from ..domain.model_metadata import ModelMetadata


class TrainingPipeline:
    def __init__(self, loader, model, trainer, evaluator, repo, tracker) -> None:
        self.loader = loader
        self.model = model
        self.trainer = trainer
        self.evaluator = evaluator
        self.repo = repo
        self.tracker = tracker

    def run(self) -> ModelMetadata:
        (train_x, train_y), (valid_x, valid_y) = self.loader.train_valid()
        self.tracker.log_param("model", type(self.model).__name__)
        self.trainer.train((train_x, train_y))
        metrics = self.evaluator.evaluate(self.model, (valid_x, valid_y))
        for k, v in metrics.items():
            self.tracker.log_metric(k, v)
        uri = self.repo.save(self.model, {"metrics": metrics})
        return ModelMetadata(artifact_uri=uri, metrics=metrics)


