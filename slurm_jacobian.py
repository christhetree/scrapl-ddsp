import os
import slurmjobs

username = "vl1019"
jobs = slurmjobs.Singularity(
    "python script_jacobian.py",
    f"/scratch/{username}/ic24_overlay-15GB-500K.ext3",
    "/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif",
    email="",
    sbatch=dict(time="4:00:00", mem="32G"),
    template="""{% extends 'job.singularity.j2' %}
      
{% block main %}
echo "Computing JTFS Jacobians"

{{ super() }}

echo "Many thanks to Bea Steers, author of SLURMJOBS."
{% endblock %}
    """,
)

# generate jobs across parameter grid
sav_dir = f"/scratch/{username}/icassp2024_scrapl_data"
run_script, job_paths = jobs.generate(
    [
        ("density_idx", range(7)),
        ("slope_idx", range(7)),
        ("seed", range(10)),
    ],
    sav_dir=sav_dir,
)

slurmjobs.util.summary(run_script, job_paths)
