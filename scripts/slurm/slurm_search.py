from slurm_utils import create_and_submit_batch_job
from datetime import datetime
import os
from pathlib import Path
import argparse

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[1]

parser = argparse.ArgumentParser()
parser.add_argument('--interactive', action='store_true')
args = parser.parse_args()

# DATA_DIR = 'uspto_full'
SCRIPT_DIR = 'scripts'
platform = 'lumi'
project = 'project_462000833'

slurm_args = {
    'job_dir': 'jobs',
    'job_ids_file': 'job_ids.txt',
    'output_dir': 'output',
    'platform': platform,
    'project': project,
    'time': '01:00:00',
    'partition': 'small-g',
    'nodes': 1,
    'gpus-per-node': 1,
    'ntasks-per-node': 1,
    'cpus-per-task': 1,
    'mem': '100G', # 50G not enough for uspto_full
    'num_arrays': 1,
    'with_containers': True,
    'container': 'syntheseus.sif',
    'venv_path': 'syntheseus-container',
    'start_array_job': 0,
    'end_array_job': 0
}
# ablations: autoregressive with 1,5,10,20 data augmentation
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#experiment_yml = 'conifg.yaml'
task = 'search'
# script_args = {"script_dir": SCRIPT_DIR,
#                "args": {"+experiment": experiment_yml}}
script_args = {"script_dir": SCRIPT_DIR,
               "args": {}}
augmentations = [1]
for augmentation in augmentations:
    #experiment_name = f'{experiment_yml.split(".")[0]}_aug{augmentation}_{time_stamp}'  
    experiment_name = f'test'  
    script_args['script_name'] = 'search.py'
    slurm_args['job_name'] = f'{task}_{experiment_name}'
    # experiment_dir = os.path.join(PROJECT_ROOT, 'experiments', experiment_name)
    # script_args['args']['training.debug'] = 'true'
    # script_args['args']['evaluation.experiment_dir'] = experiment_dir
    # script_args['args']['dataset.augmentation'] = augmentation
    # script_args['args']['wandb.name'] = experiment_name
    # script_args['args']['hydra.run.dir'] = experiment_dir
    # slurm_args['output_dir'] = experiment_dir
    # slurm_args['job_dir'] = experiment_dir
    output = create_and_submit_batch_job(slurm_args, script_args, interactive=args.interactive)
    #print(output)