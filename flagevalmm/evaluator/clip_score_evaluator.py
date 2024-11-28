import torch
import os.path as osp
import json
import numpy as np
from PIL import Image
from flagevalmm.registry import EVALUATORS

from torchmetrics.multimodal.clip_score import CLIPScore


@EVALUATORS.register_module()
class CLIPScoreEvaluator:
    def __init__(self, model_name_or_path, **kwargs) -> None:
        self.metric = CLIPScore(model_name_or_path=model_name_or_path).to("cuda")
        self.name = "clip_score"

    def get_metric_results(self, output_info, output_dir, annotations, **kwargs):
        score_sum = 0
        for info in output_info:
            image_path = osp.join(output_dir, info["image"])
            image = Image.open(image_path).convert("RGB")
            image = np.array(image) * 255
            # to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to("cuda")
            question_id = info["id"]
            prompt = annotations[question_id]["prompt"]
            clip_score = self.metric(image, prompt).item()
            score_sum += clip_score
            info["clip_score"] = clip_score
        return {"clip_score": score_sum / len(output_info)}

    def save_results(self, dataset_name, output_info, results, output_dir):
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

    def process(self, dataset, output_dir, **kwargs):
        """
        Args:
            dataset (Dataset): dataset instance
            answers (list): list of answers
        """
        annotations = dataset.get_annotation()
        dataset_name = dataset.name
        result_file = osp.join(output_dir, f"{dataset_name}.json")
        output_info = json.load(open(result_file))

        results = {}
        results["clip_score"] = self.get_metric_result(
            output_info=output_info, output_dir=output_dir, annotations=annotations
        )
        self.save_results(
            dataset_name=dataset_name,
            output_info=output_info,
            results=results,
            output_dir=output_dir,
        )
        return results