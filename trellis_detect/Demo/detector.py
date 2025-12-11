import sys

sys.path.append("../dino-v2-detect/")
sys.path.append("../TRELLIS/")

import os

from trellis_detect.Module.detector import Detector


def demo():
    home = os.environ['HOME']

    dino_model_file_path = (
        home + "/chLi/Model/DINOv2/dinov2_vitl14_reg4_pretrain.pth"
    )
    model_folder_path = home + "/chLi/Model/TRELLIS/TRELLIS-image-large"
    image_file_path = home + "/tmp/1.jpg"
    save_glb_file_path = "./output/test_gen.glb"
    save_ply_file_path = "./output/test_gen.ply"
    render = False

    detector = Detector(dino_model_file_path, model_folder_path)
    detector.detectImageFile(
        image_file_path,
        save_glb_file_path,
        save_ply_file_path,
        render,
    )
    return True
