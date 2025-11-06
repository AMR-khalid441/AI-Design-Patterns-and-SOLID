from solid.SRP.solution.src.main import load_config, build_pipeline


def test_pipeline_smoke_one_epoch():
    cfg = load_config(config_path=None, overrides={
        "epochs": 1,
        "batch_size": 16,
        "use_stratify": False,
    })
    pipeline = build_pipeline(cfg)
    meta = pipeline.run()
    assert "accuracy" in meta.metrics

