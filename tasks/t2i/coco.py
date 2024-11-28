dataset = dict(
    type="Text2ImageBaseDataset",
    data_root="/share/projset/mmdataset/t2i_json/coco.json",
    name="coco",
)

clip_evaluator = dict(
    type="CLIPScoreEvaluator",
    model_name_or_path="/share/projset/models/t2i/clip-vit-base-patch16",
)

image_quality_evaluator = dict(
    type="ImageQualityEvaluator",
    metrics=["IS", "FID"],
    example_dir="/share/projset/datasets_cv/coco/images/test_sample",
)

vqascore_evaluator = dict(
    type="VqascoreEvaluator",
    model="clip-flant5-xxl",
)

evaluator = dict(
    type="AggregationEvaluator",
    evaluators=[clip_evaluator, image_quality_evaluator, vqascore_evaluator],
)
