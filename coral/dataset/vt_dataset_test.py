import torch.utils.data as data
from util.mask_utils import *
from util.preprocess.humanparsing.run_parsing import Parsing
from util.preprocess.openpose.run_openpose import OpenPose
from util.densepose_predictor import DensePosePredictor
import os
import torch

from typing import Tuple, Optional
import torchvision.transforms as transforms
    
class VitonHDTestDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        size: Tuple[int, int] = (1024, 768),
        data_list: Optional[str] = None
    ):
        self.dataroot = dataroot_path
        self.height = size[0]
        self.width = size[1]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.toTensor = transforms.ToTensor()
        self.phase = "test"

        print("Initializing Parsing Model...")
        self.parsing = Parsing(
            atr_path="ckpt/humanparsing/parsing_atr.onnx",
            lip_path="ckpt/humanparsing/parsing_lip.onnx",
        )
        self.openpose = OpenPose(
            body_model_path="ckpt/openpose/ckpts/body_pose_model.pth"
        )
        self.densepose_predictor = DensePosePredictor(
            config_path="ckpt/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="ckpt/densepose/model_final_162be9.pkl",
        )
        im_names = []
        c_names = []

        filename = data_list

        with open(filename, "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
        
        self.HAIR_CLASS = 2
        self.im_names = im_names
        self.c_names = c_names

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self,index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        result = {}
        result["c_name"] = c_name
        result["im_name"] = im_name

        cloth_pil = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name)).resize((self.width,self.height))
        cloth = self.transform(cloth_pil)

        im_pil = Image.open(
            os.path.join(self.dataroot, self.phase, "image", im_name)
        ).resize((self.width,self.height))
        result["model_img"] = np.array(im_pil)
        image = self.transform(im_pil)
        im_pil_2 = im_pil.resize((384, 512))

        # extract parsing & keypoints & densepose_predict
        parse_img, _ = self.parsing(im_pil.resize((384, 512)))
        keypoint = self.openpose(im_pil.resize((384, 512)))
        
        parse_img2_path = os.path.join(self.dataroot, self.phase,"image-parse-v3", im_name.replace('.jpg','.png'))
        if os.path.exists(parse_img2_path):
            parse_img2_pil = Image.open(parse_img2_path)
            parse_img2 = np.array(parse_img2_pil)
            hair_mask = (parse_img2 == self.HAIR_CLASS)
        else:
            parse_img2, _ = self.parsing(im_pil.resize((self.width, self.height)))
            parse_img2 = np.array(parse_img2)
        
        assert hair_mask.shape[:2] == (self.height, self.width), \
            f"hair_mask shape mismatch: got {hair_mask.shape[:2]}, expected {(self.height, self.width)} (parse={parse_img2_pil.size}, target={(self.width, self.height)})"

        densepose_path = os.path.join(self.dataroot, self.phase,"image-densepose", im_name.replace('.png','.jpg'))
        densepose_predict = self.densepose_predictor.predict(np.array(im_pil_2))
        if not os.path.exists(densepose_path):
            pose_img = densepose_predict['densepose_image'].resize((self.width,self.height))
        else:
            pose_img = Image.open(densepose_path).resize((self.width,self.height))

        pose_img = np.array(pose_img)
        parse_img2_pil_ = parse_img2_pil.convert("P")
        pal = parse_img2_pil_.getpalette()

        hair_color = pal[self.HAIR_CLASS*3 : self.HAIR_CLASS*3+3]
        pose_img[hair_mask] = hair_color
        pose_img = Image.fromarray(pose_img, mode="RGB")
        pose_img = self.transform(pose_img)

        result["pose_img"] = pose_img
        # mask
        benchmark_mask_path = os.path.join(self.dataroot, self.phase,"agnostic-mask", im_name.replace('.jpg','.png'))
        if os.path.exists(benchmark_mask_path):
            benchmark_mask = Image.open(benchmark_mask_path)
        else:
            benchmark_mask = None
            
        mask, _ = get_agnostic_mask_vt(src_img=im_pil.resize((384, 512)), 
                                            densepose_predict=densepose_predict, 
                                            parse_img=parse_img, 
                                            keypoint=keypoint, 
                                            densepose_predictor=self.densepose_predictor,
                                            parse_img2=parse_img2_pil.resize((384, 512), Image.NEAREST),
                                            category='upper_body',
                                            pair_phase=(c_name.split(".")[0] == im_name.split(".")[0]),
                                            benchmark_mask=benchmark_mask)
        mask = mask.resize((self.width,self.height), Image.NEAREST)
        mask = self.toTensor(mask)
        mask = mask[:1]
        result["agnostic_mask"] = mask

        mask = 1-mask
        masked_img = image * mask

        tgt_img = torch.cat([cloth, image], dim=2)
        result["tgt_image"] = tgt_img

        garment_mask = torch.zeros_like(1-mask)
        densepose_mask = torch.zeros_like(1-mask)
        binary_diptych = torch.cat([garment_mask, 1-mask, densepose_mask], dim=2)
        result["binary_diptych"] = binary_diptych

        return result

        




