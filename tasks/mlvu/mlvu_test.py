register_dataset = {"mlvu_dataset.py": "MLVUDataset"}
register_evaluator = {"mlvu_dataset_evaluator.py": "MLUVDatasetEvaluator"}
dataset = dict(
    type="MLVUDataset",
    data_root="/share/projset/mmdataset/MLVU/MLVU_Test",
    split="test",
    name="mlvu_test",
)

evaluator = dict(type="MLUVDatasetEvaluator")