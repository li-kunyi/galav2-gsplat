conda create -n galav2 python=3.10
conda activate galav2
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit

pip install -r requirements.txt
conda install -c conda-forge gxx=11.4.0
pip install "numpy<2"

pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation
pip install --no-build-isolation git+https://github.com/camenduru/simple-knn 