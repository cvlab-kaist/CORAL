import torch.utils.data as data
from util.mask_utils import *
from util.preprocess.humanparsing.run_parsing import Parsing
from util.preprocess.openpose.run_openpose import OpenPose
from util.densepose_predictor import DensePosePredictor
import os
import torch
import re
import json
from typing import Tuple, Optional
import torchvision.transforms as transforms
import torch.nn.functional as F

class DressCodeTestDataset(data.Dataset):
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
        categories = []

        filename = data_list #os.path.join(dataroot_path, data_list)

        with open(filename, "r") as f:
            for line in f.readlines():
                bc = re.sub(r'\t',' ',line).split(" ")    
                im_name, c_name, category = re.sub(r'\t',' ',line).split(" ") #line.strip().split(" ")
        
                im_names.append(im_name)
                c_names.append(c_name)
                categories.append(category.strip('\n'))

        category_dict = category_dict = {"0" : "upper_body", "1" : "lower_body", "2" : "dresses" }
        categories = [category_dict[c] for c in categories]

        self.HAIR_CLASS = 2
        self.BAG_CLASS = 16
        self.im_names = im_names
        self.categories = categories
        self.c_names = c_names

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self,index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        category = self.categories[index]

        result = {}
        result["c_name"] = c_name
        result["im_name"] = im_name

        cloth_pil = Image.open(os.path.join(self.dataroot, category, "images", c_name)).resize((self.width,self.height))
        cloth = self.transform(cloth_pil)

        im_pil = Image.open(
            os.path.join(self.dataroot, category, "images", im_name)
        ).resize((self.width,self.height))
        result["model_img"] = np.array(im_pil)
        image = self.transform(im_pil)
        im_pil_2 = im_pil.resize((self.width, self.height))

        # extract parsing & keypoints & densepose_predict
        parse_img, _ = self.parsing(im_pil.resize((self.width, self.height)))
        keypoint_path = os.path.join(self.dataroot, category, "keypoints", im_name.replace('_0.jpg','_2.json'))
        
        if not os.path.exists(keypoint_path):
            keypoint_path = os.path.join(self.dataroot, category, "keypoints", im_name.replace('_0.png','_2.json'))
        if os.path.exists(keypoint_path):
            with open(keypoint_path, 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['keypoints']
                pose_data = np.array(pose_data)
                keypoint = pose_data.reshape((-1, 4))
        else:
            keypoint = self.openpose(im_pil.resize((self.width, self.height)))
            keypoint = keypoint["pose_keypoints_2d"]
            keypoint = np.array(keypoint)
            keypoint = keypoint.reshape((-1, 2))
        
        parse_img2_path = os.path.join(self.dataroot, category, "label_maps", im_name.replace('_0.jpg','_4.png'))
        if os.path.exists(parse_img2_path):
            parse_img2_pil = Image.open(parse_img2_path)
            parse_img2 = np.array(parse_img2_pil)
            hair_mask = (parse_img2 == self.HAIR_CLASS)
            bag_mask = (parse_img2 == self.BAG_CLASS)
            assert hair_mask.shape[:2] == (self.height, self.width), \
            f"hair_mask shape mismatch: got {hair_mask.shape[:2]}, expected {(self.height, self.width)} (parse={parse_img2_pil.size}, target={(self.width, self.height)})"

        densepose_path = os.path.join(self.dataroot, category, "image-densepose", im_name.replace('_0.png','_0.jpg'))
        densepose_predict = self.densepose_predictor.predict(np.array(im_pil_2))
        if not os.path.exists(densepose_path):
            print("Densepose path not found, using predicted densepose.")
            pose_img = densepose_predict['densepose_image'].resize((self.width,self.height))
        else:
            pose_img = Image.open(densepose_path).resize((self.width,self.height))

        pose_img = np.array(pose_img)
        parse_img2_pil_ = parse_img2_pil.convert("P")
        pal = parse_img2_pil_.getpalette()
        hair_color = [254, 0, 0]  # Default hair color (red)
        bag_color = pal[self.BAG_CLASS*3 : self.BAG_CLASS*3+3]
        pose_img[hair_mask] = hair_color
        pose_img[bag_mask] = bag_color
        pose_img = Image.fromarray(pose_img, mode="RGB")
        
        pose_img = self.transform(pose_img)

        result["pose_img"] = pose_img
        # mask
        mask, _ = get_agnostic_mask_dc(src_img=im_pil.resize((self.width, self.height)), 
                                            densepose_predict=densepose_predict, 
                                            parse_img=parse_img, 
                                            keypoint=keypoint, 
                                            densepose_predictor=self.densepose_predictor,
                                            parse_img2=parse_img2_pil.resize((self.width, self.height), Image.NEAREST),
                                            category=category,
                                            size=(self.width, self.height),
                                            pair_phase=(c_name.split("_")[0] == im_name.split("_")[0]))

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
        result["category"] = category

        return result