from solid.SRP.violation.src.model_trainer import ModelTrainer


def test_model_trainer_smoke_one_epoch_no_stratify():
    mt = ModelTrainer(config_path=None, overrides={"epochs": 1, "batch_size": 16, "use_stratify": False})
    mt.load_data()
    mt.build_model()
    mt.train()
    metrics = mt.evaluate()
    assert "accuracy" in metrics


