module purge
module use /appl/local/training/modules/AI-20241126
module load cotainr

# $ 
# $ 
cotainr build syntheseus.sif --base-image=/appl/local/containers/sif-images/lumi-rocm-rocm-6.0.3.sif --conda-env=environment_full.yml --accept-license

# NOTE: important not to run bind/mount stuff before creating the container
# singularity exec syntheseus.sif bash -c "python -m venv syntheseus-container --system-site-packages
# source syntheseus-container/bin/activate pip install "syntheseus[all]""
