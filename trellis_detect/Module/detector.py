import os
import torch
import imageio
import numpy as np
from PIL import Image
from typing import Union
from torchvision import transforms
from trellis.pipelines import samplers, TrellisImageTo3DPipeline
from trellis.pipelines.base import Pipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis_detect.Method.path import createFileFolder

from dino_v2_detect.Model.vision_transformer import vit_large

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.


class LocalTrellisImageTo3DPipeline(TrellisImageTo3DPipeline, Pipeline):
    dino_model_file_path: str

    @staticmethod
    def from_pretrained(path: str) -> "LocalTrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = Pipeline.from_pretrained(path)
        new_pipeline = LocalTrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(
            samplers, args["sparse_structure_sampler"]["name"]
        )(**args["sparse_structure_sampler"]["args"])
        new_pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"][
            "params"
        ]

        new_pipeline.slat_sampler = getattr(samplers, args["slat_sampler"]["name"])(
            **args["slat_sampler"]["args"]
        )
        new_pipeline.slat_sampler_params = args["slat_sampler"]["params"]

        new_pipeline.slat_normalization = args["slat_normalization"]

        new_pipeline._init_image_cond_model(args["image_cond_model"])

        return new_pipeline

    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        # dinov2_model = torch.hub.load(self.dino_model_file_path, name, pretrained=True)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        dinov2_model = vit_large(
            patch_size=14,
            num_register_tokens=4,
            img_size=518,
            ffn_layer="mlp",
            block_chunks=0,
            interpolate_antialias=True,
            interpolate_offset=0.0,
            init_values=1.0,
        ).to("cuda", dtype=dtype)
        model_state_dict = torch.load(self.dino_model_file_path, map_location="cpu")
        dinov2_model.load_state_dict(model_state_dict, strict=True)
        dinov2_model.eval()
        self.models["image_cond_model"] = dinov2_model
        transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.image_cond_model_transform = transform
        return


class Detector(object):
    def __init__(
        self,
        dino_model_file_path: Union[str, None] = None,
        model_folder_path: Union[str, None] = None,
    ) -> None:
        if dino_model_file_path is not None and model_folder_path is not None:
            self.loadModel(dino_model_file_path, model_folder_path)
        return

    def loadModel(self, dino_model_file_path: str, model_folder_path: str) -> bool:
        if not os.path.exists(dino_model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t dino model file not exist!")
            print("\t dino_model_file_path:", dino_model_file_path)
            return False

        if not os.path.exists(model_folder_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model folder not exist!")
            print("\t model_folder_path:", model_folder_path)
            return False

        # Load a pipeline from a model folder or a Hugging Face model hub.
        LocalTrellisImageTo3DPipeline.dino_model_file_path = dino_model_file_path
        self.pipeline = LocalTrellisImageTo3DPipeline.from_pretrained(model_folder_path)
        self.pipeline.cuda()
        return True

    def detect(
        self,
        image: np.ndarray,
        save_glb_file_path: str,
        save_ply_file_path: str,
        render: bool = False,
    ) -> bool:
        outputs = self.pipeline.run(
            image,
            seed=1,
            # Optional parameters
            # sparse_structure_sampler_params={
            #     "steps": 12,
            #     "cfg_strength": 7.5,
            # },
            # slat_sampler_params={
            #     "steps": 12,
            #     "cfg_strength": 3,
            # },
        )
        # outputs is a dictionary containing generated 3D assets in different formats:
        # - outputs['gaussian']: a list of 3D Gaussians
        # - outputs['radiance_field']: a list of radiance fields
        # - outputs['mesh']: a list of meshes

        if render:
            save_result_folder_path = "./output/"
            os.makedirs(save_result_folder_path)

            # Render the outputs
            video = render_utils.render_video(outputs["gaussian"][0])["color"]
            imageio.mimsave(save_result_folder_path + "sample_gs.mp4", video, fps=30)
            video = render_utils.render_video(outputs["radiance_field"][0])["color"]
            imageio.mimsave(save_result_folder_path + "sample_rf.mp4", video, fps=30)
            video = render_utils.render_video(outputs["mesh"][0])["normal"]
            imageio.mimsave(save_result_folder_path + "sample_mesh.mp4", video, fps=30)

        # GLB files can be extracted from the outputs
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            # Optional parameters
            simplify=0.95,  # Ratio of triangles to remove in the simplification process
            texture_size=1024,  # Size of the texture used for the GLB
        )

        createFileFolder(save_glb_file_path)
        glb.export(save_glb_file_path)

        # Save Gaussians as PLY files
        outputs["gaussian"][0].save_ply(save_ply_file_path)
        return True

    def detectImageFile(
        self,
        image_file_path: str,
        save_glb_file_path: str,
        save_ply_file_path: str,
        render: bool = False,
    ) -> bool:
        if not os.path.exists(image_file_path):
            print("[ERROR][Detector::detectImageFile]")
            print("\t image file not exist!")
            print("\t image_file_path:", image_file_path)
            return False

        image = Image.open(image_file_path)
        return self.detect(image, save_glb_file_path, save_ply_file_path, render)
