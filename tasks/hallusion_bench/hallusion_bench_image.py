config = dict(
    dataset_path="/share/projset/mmdataset/huggingface_format/HallusionBench",
    split="image",
    processed_dataset_path="HallusionBench",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(type="PromptTemplate"),
    name="hallusion_bench",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")