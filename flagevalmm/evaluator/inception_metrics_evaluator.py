import json
import os.path as osp
import torch_fidelity
from flagevalmm.registry import EVALUATORS


@EVALUATORS.register_module()
class InceptionMetricsEvaluator:
    def __init__(self, metrics=["IS", "FID"], example_dir=None, **kwargs) -> None:
        self.metrics = metrics
        if "FID" in self.metrics:
            assert example_dir is not None, "example_dir is required for FID"
        self.example_dir = example_dir

    def get_metric_results(self, output_dir, **kwargs):
        results = {}
        cal_isc = "IS" in self.metrics
        cal_fid = "FID" in self.metrics
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=output_dir,
            input2=self.example_dir,
            cuda=True,
            isc=cal_isc,
            fid=cal_fid,
            kid=False,
            prc=False,
            verbose=False,
        )
        if cal_fid:
            results["FID"] = metrics_dict["frechet_inception_distance"]
        if cal_isc:
            results["IS"] = metrics_dict["inception_score_mean"]
        return results

    def process(self, dataset, output_dir, **kwargs):
        """
        Args:
            dataset (Dataset): dataset instance
            answers (list): list of answers
        """
        dataset_name = dataset.name
        result_file = osp.join(output_dir, f"{dataset_name}.json")
        output_info = json.load(open(result_file))

        results = self.get_metric_results(output_dir=output_dir)
        json.dump(
            results, open(osp.join(output_dir, f"{dataset_name}_result.json"), "w")
        )
        # save evaluation results
        json.dump(
            output_info,
            open(osp.join(output_dir, f"{dataset_name}_evaluated.json"), "w"),
            ensure_ascii=False,
            indent=2,
        )
        return results
