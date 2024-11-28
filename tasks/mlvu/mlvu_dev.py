register_dataset = {"mlvu_dataset.py": "MLVUDataset"}
register_evaluator = {"mlvu_dataset_evaluator.py": "MLUVDatasetEvaluator"}

dataset = dict(
    type="MLVUDataset",
    data_root="/share/projset/mmdataset/MLVU/MLVU",
    split="dev",
    name="mlvu_dev",
)

evaluator = dict(type="MLUVDatasetEvaluator")
