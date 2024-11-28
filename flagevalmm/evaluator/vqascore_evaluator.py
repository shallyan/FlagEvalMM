from flagevalmm.registry import EVALUATORS
import os.path as osp
import json


@EVALUATORS.register_module()
class VqascoreEvaluator:
    """
    The evaluation method is adapted from the VQAScore project:
    - Project: https://linzhiqiu.github.io/papers/vqascore/
    - Paper: "Evaluating Text-to-Visual Generation with Image-to-Text Generation" (https://arxiv.org/pdf/2404.01291)
    """

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.load_model()

    def load_model(self):
        import t2v_metrics

        self.clip_flant5 = t2v_metrics.VQAScore(model=self.model)

    def get_metric_results(self, output_info, output_dir, **kwargs):
        vqascore_sum = 0
        for info in output_info:
            image_path = osp.join(output_dir, info["image"])
            text = info["prompt"]

            clip_flant5_score = self.clip_flant5(images=[image_path], texts=[text])
            vqascore = float(clip_flant5_score.item())
            info["vqascore"] = vqascore
            vqascore_sum += vqascore

        vqascore_sum /= len(output_info)
        results = {"vqascore": vqascore_sum}
        return results

    def process(self, dataset, output_dir, **kwargs):
        dataset_name = dataset.name
        result_file = osp.join(output_dir, f"{dataset_name}.json")
        output_info = json.load(open(result_file))

        results = self.get_metric_results(
            output_info=output_info, output_dir=output_dir
        )
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
