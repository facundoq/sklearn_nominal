import os
import openml
from tqdm import tqdm
import subprocess

benchmark_suite = openml.study.get_suite('OpenML-CC18')  # obtain the benchmark suite

#pbar = tqdm(total=len(benchmark_suite.tasks))

for i,task_id in (pbar := tqdm(list(enumerate(benchmark_suite.tasks)))):  # iterate over all tasks
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    filepath = f"tests/outputs/{dataset.name}.dot"
    image_filepath = f"tests/outputs/{dataset.name}.png"
    pbar.update(i)
    if os.path.exists(filepath):
        subprocess.run(f"dot -Tpng {filepath} > {image_filepath}",shell=True)