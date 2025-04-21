module purge
module load LUMI/24.03 cotainr   
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

cotainr build syntheseus.sif --base-image=/appl/local/containers/sif-images/lumi-rocm-rocm-6.0.3.sif --conda-env=environment_lumi.yml --accept-license

# NOTE: important not to run bind/mount stuff before creating the container
# singularity exec syntheseus.sif bash -c "python -m venv syntheseus-container --system-site-packages & source syntheseus-container/bin/activate pip install "syntheseus[all]""
