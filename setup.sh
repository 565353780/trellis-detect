cd ..
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
git clone https://github.com/EasternJournalist/utils3d.git
git clone https://github.com/NVlabs/nvdiffrast.git
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git
git clone https://github.com/autonomousvision/mip-splatting.git
git clone https://github.com/FindDefinition/cumm.git
git clone https://github.com/traveller59/spconv.git

sudo apt install libjpeg-dev -y

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install imageio imageio-ffmpeg tqdm easydict \
  opencv-python-headless scipy ninja rembg onnxruntime \
  trimesh open3d xatlas pyvista pymeshfix igraph \
  transformers tensorboard pandas lpips pillow-simd

cd utils3d
git checkout 9a4eb15e4021b67b12c460c7057d642626897ec8
cd ..

pip install ./utils3d/

pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu128

pip install flash-attn --no-build-isolation

pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

pip install ./kaolin/
pip install ./nvdiffrast/
pip install ./diffoctreerast/
pip install ./mip-splatting/submodules/diff-gaussian-rasterization/
# cp -r ./extensions/vox2seq ./vox2seq
# pip install ./vox2seq
pip install ./cumm
pip install ./spconv

pip install gradio==4.44.1 gradio_litmodel3d==0.0.1
