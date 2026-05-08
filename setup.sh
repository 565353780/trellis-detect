cd ..
git clone --depth 1 --recurse-submodules https://github.com/microsoft/TRELLIS.git
git clone --depth 1 https://github.com/EasternJournalist/utils3d.git
git clone --depth 1 -b v0.0.33 --recursive https://github.com/facebookresearch/xformers.git
git clone --depth 1 -b v2.8.3 https://github.com/Dao-AILab/flash-attention.git
git clone --depth 1 https://github.com/NVlabs/nvdiffrast.git
git clone --depth 1 --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git
git clone --depth 1 https://github.com/autonomousvision/mip-splatting.git
# git clone https://github.com/FindDefinition/cumm.git
# git clone https://github.com/traveller59/spconv.git
git clone https://github.com/565353780/dino-v2-detect.git

conda install -c conda-forge libstdcxx-ng libjpeg-turbo -y

pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

pip install imageio imageio-ffmpeg tqdm easydict \
  opencv-python-headless scipy ninja rembg onnxruntime \
  trimesh open3d xatlas pyvista pymeshfix igraph \
  transformers tensorboard pandas lpips pillow-simd

cd utils3d
git checkout 9a4eb15e4021b67b12c460c7057d642626897ec8
cd ..

pip install ./utils3d --no-build-isolation
pip install ./xformers --no-build-isolation
pip install ./flash-attention --no-build-isolation
pip install ./nvdiffrast --no-build-isolation
pip install ./mip-splatting/submodules/diff-gaussian-rasterization --no-build-isolation
pip install ./diffoctreerast --no-build-isolation
# cp -r ./extensions/vox2seq ./vox2seq
# pip install ./vox2seq
# pip install ./cumm
# SPCONV_DISABLE_JIT=1 pip install ./spconv
pip install spconv-cu120

pip install kaolin==0.18.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124.html

pip install gradio==4.44.1 gradio_litmodel3d==0.0.1
