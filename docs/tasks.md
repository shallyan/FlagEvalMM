# About Tasks

Task is the minimal unit of evaluation. A task contains a processor, evaluator and a config.

## Add a new task

1. Process the dataset to the standard format.

```bash
python flagevalmm/datasets/data_preprocessor.py -c {task_config.py} [-f]
# For example
python flagevalmm/datasets/data_preprocessor.py -c tasks/math_vista/math_vista_testmini.py
```

