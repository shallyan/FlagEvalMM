# -*- coding: utf-8 -*-
# @Time    : 2024/9/6 16:38
# @Author  : yesliu
# @File    : genai_bench.py
task_name = "t2i"

dataset = dict(
    type="Text2ImageBaseDataset",
    data_root="/share/projset/mmdataset/t2i_json/genai_bench.json",
    name="genaibench",
)

evaluator = dict(
    type="VqascoreEvaluator",
    model="clip-flant5-xxl",
    model_path="/share/projset/models/t2i/clip-flant5-xxl",
)
