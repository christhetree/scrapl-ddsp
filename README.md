# SCRAPL DDSP
Scattering with Random Paths as Loss for Differentiable Digital Signal Processing

<hr>
<h2>Instructions for Reproducibility</h2>

<ol>
    <li>Clone this repository and open its directory.</li>
    <li>
    Install the requirements:
    <br><code>conda env create --file=conda_env_gpu.yml</code>
    <br>or 
    <br><code>pip install uv</code>
    <br><code>uv pip install -r requirements_gpu.txt</code>
    <br>For posterity, <code>requirements_all_gpu.txt</code> and <code>requirements_all_cpu.txt</code> are also provided.
    </li>
    <li>The source code can be explored in the <code>experiments/</code>, <code>scrapl/</code>, and <code>eval_808/</code> directories.</li>
    <li>All experiment config files can be found in the <code>configs/</code> directory.</li>
    <li>The dataset for the Roland TR-808 sound matching task can be found <a href="https://samplesfrommars.com/products/tr-808-samples" target="_blank">here</a>.</li>
    <li>Create an out directory (<code>mkdir out</code>).</li>
    <li>
    All experiments can be run by modifying <code>scripts/train.py</code> and the corresponding 
    <code>configs/.../train_ ... .yml</code> config file and then running <code>python scripts/train.py</code>.
    <br>Make sure your PYTHONPATH has been set correctly by running commands like:
    <br><code>export PYTHONPATH=$PYTHONPATH:BASE_DIR/scrapl/</code>,
    <br><code>export PYTHONPATH=$PYTHONPATH:BASE_DIR/scrapl/kymatio/</code>,
    <br>and <code>export PYTHONPATH=$PYTHONPATH:BASE_DIR/scrapl/scrapl/</code>.
    </li>
    <li>
    The source code is currently not documented, but don't hesitate to open an issue if you have any questions or 
    comments.
    </li>
    <li>
    A <code>pip</code> installable Python package of SCRAPL for the JTFS is coming soon.
    </li>
</ol>
