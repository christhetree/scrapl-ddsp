# SCRAPL DDSP
Scattering with Random Paths as Loss for Differentiable Digital Signal Processing

<hr>
<h2>Instructions for Reproducibility</h2>

<ol>
    <li>Clone this repository and open its directory.</li>
    <li>
    Install the requirements using <br><code>conda env create --file=conda_env_cpu.yml</code> or <br>
    <code>conda env create --file=conda_env.yml</code><br> for GPU acceleration. <br>
    <code>requirements_pipchill.txt</code> and <code>requirements_all.txt</code> are also provided as references, 
    but are not needed when using the <code>conda_env.yml</code> files.
    </li>
    <li>The source code can be explored in the <code>experiments/</code> directory.</li>
    <li>All models from the paper can be found in the <code>models/</code> directory.</li>
    <li>Create an out directory (<code>mkdir out</code>).</li>
    <li>
    All models can be evaluated by modifying <code>scripts/validate.py</code> and the corresponding 
    <code>configs/eval_ ... .yml</code> config file and then running <code>python scripts/validate.py</code>. <br>
    Make sure your PYTHONPATH has been set correctly by running a command like 
    <code>export PYTHONPATH=$PYTHONPATH:BASE_DIR/scrapl_ddsp/</code>.
    </li>
    <li>
    (Optional) All models can be trained by modifying <code>scripts/train.py</code> and the corresponding 
    <code>configs/train_ ... .yml</code> config file and then running <code>python scripts/train.py</code>.
    </li>
    <li>
    The source code is currently not documented, but don't hesitate to open an issue if you have any questions or 
    comments.
    </li>
</ol>
