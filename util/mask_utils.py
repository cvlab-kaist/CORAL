## Modified from : yisol/IDM-VTON/inference_dc.py 9https://github.com/yisol/IDM-VTON/blob/0d5f3ec2d737487a9bb24e4100936ad254780383/inference_dc.py#L231)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from numpy.linalg import lstsq
import cv2
from typing import Tuple, Union
import torch

# Our parsemap color index
PARSE_TORSO_COLORS = [
    (255, 85, 0),  # 5: upper-clothes
    (0, 0, 85),    # 6: dress
    (0, 119, 221), # 7: coat
]

PARSE_ARMS_COLORS = [
    (51, 170, 221),     # 14: left arm
    (0, 255, 255)      # 15: right arm
]

PARSE_LOWER_COLORS = [
    (85, 85, 0),     # 8: socks
    (0, 85, 85),     # 9: pants
    (0, 128, 0),     # 12: skirt
    (85, 255, 170),  # 16: left leg
    (170, 255, 85),  # 17: right leg
    (255, 255, 0),   # 18: left shoe
    (255, 170, 0)    # 19: right shoe
]

label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
    "neck": 18,
}

def fill_leg_gap(dst, category, width, height):
    if category not in ["lower_body", "dresses"]:
        return dst

    dst = (dst > 0).astype(np.uint8) * 255

    y0 = int(height * 0.55)
    roi = dst[y0:, :].copy()

    kw = max(9, (width // 10) | 1)    
    kh = max(9, (height // 40) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)

    cx = width // 2
    band_w = max(20, width // 4)
    x1 = max(0, cx - band_w)
    x2 = min(width, cx + band_w)

    roi[:, x1:x2] = closed[:, x1:x2]
    dst[y0:, :] = roi
    return dst

## Modified from : Leffa/leffa_utils/utils.py : https://github.com/franciszzj/Leffa/blob/05a259104b6927607776c7edb3e86b75406f20aa/leffa_utils/utils.py#L76
def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width=1,
                 mode='constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst


def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)

    return refine_mask

def extend_arm_mask(wrist, elbow, scale):
    wrist = elbow + scale * (wrist - elbow)
    return wrist

def directional_dilate_np(mask_hw_u8, px):
    if isinstance(px, int):
        t = b = l = r = max(0, int(px))
    else:
        t, b, l, r = [max(0, int(v)) for v in px]

    H, W = mask_hw_u8.shape
    m = (mask_hw_u8 > 0).astype(np.uint8)

    if t > 0 or b > 0:
        m_pad = np.pad(m, ((t, b), (0, 0)), mode="constant", constant_values=0)
        k = np.ones((t + b + 1, 1), np.uint8)
        m_pad = cv2.dilate(m_pad, k, anchor=(0, 0), iterations=1)
        m = m_pad[:H, :]

    # horizontal (l,r)
    if l > 0 or r > 0:
        m_pad = np.pad(m, ((0, 0), (l, r)), mode="constant", constant_values=0)
        k = np.ones((1, l + r + 1), np.uint8)
        m_pad = cv2.dilate(m_pad, k, anchor=(0, 0), iterations=1)
        m = m_pad[:, :W]

    return m



def get_agnostic_mask_dc(src_img, densepose_predict, parse_img, keypoint, category, size=(384,512), parse_img2=None, densepose_predictor=None, pair_phase=False):
    width, height = size
    if category == 'lower_body':
        im_parse = parse_img2.resize((width,height),Image.NEAREST)
    else:
        im_parse = parse_img.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)
    parse_shape = (parse_array > 0).astype(np.float32)

    pose_data = keypoint
    src_img_array = np.array(src_img)
    iuv_predict = densepose_predictor.predict_iuv(src_img_array)

    parse_head = (parse_array == 1).astype(np.float32) + \
        (parse_array == 2).astype(np.float32) + \
        (parse_array == 3).astype(np.float32) + \
        (parse_array == 11).astype(np.float32) + \
        (parse_array == 18).astype(np.float32)
    
    parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                        (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["scarf"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)
    
    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)
    arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

    if category == 'dresses':
        label_cat = 7
        if not pair_phase:
            parse_list = [4, 5, 6, 7, 12, 13]
        else:
            parse_list = [7,12,13]
        parse_mask = np.zeros_like(parser_mask_changeable)
        for parse_index in parse_list:
            parse_mask += (parse_array == parse_index).astype(np.float32) 
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'upper_body':
        label_cat = 4
        parse_mask = (parse_array == 4).astype(np.float32)

        parser_mask_fixed += (parse_array == label_map["skirt"]).astype(np.float32) + \
                            (parse_array == label_map["pants"]).astype(np.float32)

        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    elif category == 'lower_body':
        label_cat = 6
        parse_mask = (parse_array == 6).astype(np.float32) + \
                    (parse_array == 12).astype(np.float32) + \
                    (parse_array == 13).astype(np.float32) + \
                    (parse_array == 5).astype(np.float32)

        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                            (parse_array == 14).astype(np.float32) + \
                            (parse_array == 15).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    

    parse_head = torch.from_numpy(parse_head)  # [0,1]
    parse_mask = torch.from_numpy(parse_mask)  # [0,1]
    parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
    parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

    # dilation
    parse_without_cloth = np.logical_and(parse_shape, np.logical_not(parse_mask))
    parse_mask = parse_mask.cpu().numpy()

    width = size[0]
    height = size[1]

    im_arms = Image.new('L', (width, height))
    arms_draw = ImageDraw.Draw(im_arms)
    if category == 'dresses' or category == 'upper_body':
        shoulder_right = tuple(np.multiply(pose_data[2, :2], height / 512.0))
        shoulder_left = tuple(np.multiply(pose_data[5, :2], height / 512.0))
        elbow_right = tuple(np.multiply(pose_data[3, :2], height / 512.0))
        elbow_left = tuple(np.multiply(pose_data[6, :2], height / 512.0))
        wrist_right = tuple(np.multiply(pose_data[4, :2], height / 512.0))
        wrist_left = tuple(np.multiply(pose_data[7, :2], height / 512.0))
        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                arms_draw.line([wrist_left, elbow_left, shoulder_left, shoulder_right], 'white', 30, 'curve')
            else:
                arms_draw.line([wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right], 'white', 30,
                            'curve')
        elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                arms_draw.line([shoulder_left, shoulder_right, elbow_right, wrist_right], 'white', 30, 'curve')
            else:
                arms_draw.line([elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right], 'white', 30,
                            'curve')
        else:
            arms_draw.line([wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right], 'white',
                        30, 'curve')
    
        if not pair_phase:
            if densepose_predict is not None:
                if densepose_predict[0] is not None:
                    I_map_bbox = densepose_predict[0][0].labels
                    bbox_position = [int(x) for x in densepose_predict[1][0].cpu().numpy().tolist()]
                    x1, y1, w, h = bbox_position
                    I_map = np.zeros((height, width))
                    I_map[y1:y1+h, x1:x1+w] = I_map_bbox.cpu().numpy()

                    if iuv_predict is not None:
                        iuv_predict = iuv_predict[:, :, 0].astype(np.int32)  # body part index
                        cmap = plt.cm.get_cmap("turbo", 25)  # 0~24 color
                        colored_I = cmap(iuv_predict / 24.0)[..., :3]  # RGBA → RGB
                        colored_I = (colored_I * 255).astype(np.uint8)
                    else:
                        colored_I = None

                    densepose_hands_I = [3, 4]
                    densepose_hands_I_mask = np.isin(I_map, densepose_hands_I).astype(np.uint8)

                    hand = (densepose_hands_I_mask > 0).astype(np.uint8)
                    hand = (hand & (arms > 0).astype(np.uint8)).astype(np.uint8)

                    densepose_arm_I   = [15, 16, 17, 18, 19, 20, 21, 22] 
                    densepose_arm_I_mask   = np.isin(I_map, densepose_arm_I).astype(np.uint8)
                    im_arms_base = (densepose_arm_I_mask > 0).astype(np.uint8)
                else:
                    hand = (arms > 0).astype(np.uint8)
                    im_arms_base = np.zeros_like(arms).astype(np.uint8)
                    colored_I = None
            else:
                colored_I = None
                hand = (arms > 0).astype(np.uint8)
                im_arms_base = np.zeros_like(arms).astype(np.uint8)
            
            hand_buffer = cv2.dilate(hand, np.ones((5, 5), np.uint8), iterations=5).astype(np.uint8)
            im_arms_base = np.logical_or(im_arms_base, (arms > 0)).astype(np.uint8)
            im_arms = np.logical_and(im_arms_base, np.logical_not(hand_buffer)).astype(np.uint8)
        else:
            colored_I = None
            #hand = (arms > 0).astype(np.uint8)
            #im_arms_base = np.zeros_like(arms).astype(np.uint8)

        
        if height > 512:
            if pair_phase:
                im_arms = cv2.dilate(np.float32(im_arms), np.ones((10,10), np.uint16), iterations=5)
            else:
                im_arms = cv2.dilate(np.float32(im_arms), np.ones((10,10), np.uint16), iterations=7)
        elif height > 256:
            if pair_phase:
                im_arms = cv2.dilate(np.float32(im_arms), np.ones((7,7), np.uint16), iterations=5)
            else:
                im_arms = cv2.dilate(np.float32(im_arms), np.ones((10,10), np.uint16), iterations=7)

        im_arms = (im_arms > 0).astype(np.uint8)
        hands = np.logical_and(np.logical_not(im_arms), arms)
        parse_mask += im_arms
        parser_mask_fixed += hands
        #parser_mask_fixed = np.logical_or(parser_mask_fixed.cpu().numpy(), hands)
    else:
        colored_I = None

    parse_head_2 = torch.clone(parse_head)
    if category == 'dresses' or category == 'upper_body':
        points = []
        points.append(np.multiply(pose_data[2, :2], height / 512.0))
        points.append(np.multiply(pose_data[5, :2], height / 512.0))
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords, rcond=None)[0]
        for i in range(parse_array.shape[1]):
            y = i * m + c
            parse_head_2[int(y - 20 * (height / 512.0)):, i] = 0
    
    parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2.detach().cpu().numpy(), dtype=np.uint16))
    parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head.detach().cpu().numpy(), dtype=np.uint16),
                                                        np.logical_not(np.array(parse_head_2.detach().cpu().numpy(), dtype=np.uint16))))

    if height > 512:
        parse_mask = cv2.dilate(parse_mask, np.ones((20, 20), np.uint16), iterations=5)
    elif height > 256:
        parse_mask = cv2.dilate(parse_mask, np.ones((10, 10), np.uint16), iterations=5)
    else:
        parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
    
    parse_mask = np.logical_and(parser_mask_changeable.detach().cpu().numpy(), np.logical_not(parse_mask))
    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    agnostic_mask = 1 - parse_mask_total
    img = np.where(agnostic_mask, 255, 0)

    if not pair_phase:
        dst = hole_fill(img.astype(np.uint8))

        if not pair_phase and category == "lower_body":
            dst = fill_leg_gap(dst, category, width, height)

        must_fix_label = [1, 2, 3, 9, 10, 16]
        must_fix_mask = np.zeros_like(parse_array).astype(np.uint8)
        for label in must_fix_label:
            must_fix_mask = np.logical_or(must_fix_mask, (parse_array == label).astype(np.uint8))
            must_fix_mask_u8 = (must_fix_mask.astype(np.uint8) * 255)
            must_fix_mask_u8 = cv2.erode(must_fix_mask_u8, np.ones((3, 3), np.uint8), iterations=1)
            must_fix_mask = (must_fix_mask_u8 > 0).astype(np.uint8)

            dst[must_fix_mask] = 0
            if category in ["dresses", "upper_body"]:
                dst[hands] = 0
        
        agnostic_mask =  (dst > 0).astype(np.uint8)
    else:
        agnostic_mask = (img>0).astype(np.uint8)

    if not pair_phase:
        if category == "lower_body":
            agnostic_mask = directional_dilate_np(agnostic_mask.astype(np.uint8), (5, 5, 13, 13))
        elif category == "upper_body":
            agnostic_mask = directional_dilate_np(agnostic_mask.astype(np.uint8), (15, 0, 22, 22))
        else:
            agnostic_mask = directional_dilate_np(agnostic_mask.astype(np.uint8), (15, 13, 0, 0))
    
    agnostic_mask = Image.fromarray(agnostic_mask.astype(np.uint8)*255)

    return agnostic_mask, colored_I

    
        

    
def get_agnostic_mask_vt(src_img, densepose_predict, parse_img, keypoint, size=(384,512), parse_img2=None, densepose_predictor=None, pair_phase=False, category='upper_body',benchmark_mask=None):
    width, height = size
    arm_width = 60
    im_parse = parse_img.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)
    
    parse_head = (parse_array == 1).astype(np.float32) + (parse_array == 3).astype(np.float32) + (parse_array == 11).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)
    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    arms_left = (parse_array == 14).astype(np.float32)
    arms_right = (parse_array == 15).astype(np.float32)

    #### 
    parser_mask_fixed_0 = parser_mask_fixed.copy()  
    parser_mask_changeable_0 = parser_mask_changeable.copy()

    parse_mask_0 = (parse_array == 4).astype(np.float32) + \
        (parse_array == 7).astype(np.float32)
    parser_mask_fixed_lower_cloth_0 = (parse_array == label_map["skirt"]).astype(np.float32) + \
                                    (parse_array == label_map["pants"]).astype(
                                        np.float32)
    parser_mask_fixed_0 += parser_mask_fixed_lower_cloth_0
    parser_mask_changeable_0 += np.logical_and(
        parse_array, np.logical_not(parser_mask_fixed_0))
    #### 

    if category == 'upper_body':
        # Densepose
        densepose_left_arm_I  = [15, 17, 19, 21]
        densepose_right_arm_I = [16, 18, 20, 22]
        densepose_torso_I     = []
        densepose_hands_I     = [3, 4]
        densepose_target_I    = densepose_left_arm_I + densepose_right_arm_I + densepose_torso_I

        color_thr = 10

        def color_mask_from_values(parse_img, colors, thr=10):
            parse_img = parse_img.convert("RGB") 
            arr = np.array(parse_img)
            mask = np.zeros(arr.shape[:2], dtype=np.uint8)
            for color in colors:
                diff = np.abs(arr - np.array(color)[None, None, :])  
                dist = np.sum(diff, axis=-1)
                mask |= (dist < thr).astype(np.uint8)
            return mask
        
        #if data_type == "vt":
        parse_mask = color_mask_from_values(parse_img2, PARSE_TORSO_COLORS, color_thr).astype(np.float32)

        src_img_array = np.array(src_img)
        iuv_predict = densepose_predictor.predict_iuv(src_img_array)
        densepose_predict = densepose_predictor.predict(src_img_array)

        if densepose_predict is not None:
            # Get I map from densepose
            if densepose_predict[0] is not None:
                I_map_bbox = densepose_predict[0][0].labels
                bbox_position = [int(x) for x in densepose_predict[1][0].cpu().numpy().tolist()]
                x1, y1, w, h = bbox_position
                I_map = np.zeros((height, width))
                I_map[y1:y1+h, x1:x1+w] = I_map_bbox.cpu().numpy()

                # Visualize I channel
                if iuv_predict is not None:
                    iuv_predict = iuv_predict[:, :, 0].astype(np.int32)  # body part index
                    cmap = plt.cm.get_cmap("turbo", 25)  # 0~24 color
                    colored_I = cmap(iuv_predict / 24.0)[..., :3]  # RGBA → RGB
                    colored_I = (colored_I * 255).astype(np.uint8)
                else:
                    colored_I = None

                # Densepose target region mask
                densepose_target_I_mask    = np.isin(I_map, densepose_target_I).astype(np.uint8)
            else:
                colored_I = None
                densepose_target_I_mask = np.zeros((height, width), dtype=np.uint8)
        else:
            colored_I = None
            densepose_target_I_mask = np.zeros((height, width), dtype=np.uint8)

        parser_mask_fixed_lower_cloth_ours = color_mask_from_values(parse_img2, PARSE_LOWER_COLORS, color_thr).astype(np.float32)
        parser_mask_left_arms              = color_mask_from_values(parse_img2, [PARSE_ARMS_COLORS[0]], color_thr).astype(np.float32)
        parser_mask_right_arms             = color_mask_from_values(parse_img2, [PARSE_ARMS_COLORS[1]], color_thr).astype(np.float32)
        parser_mask_arms                   = np.logical_or(parser_mask_left_arms, parser_mask_right_arms)

        parser_mask_fixed_lower_cloth_parser = (parse_array == label_map["skirt"]).astype(np.float32) + \
                                        (parse_array == label_map["pants"]).astype(np.float32) + \
                                        (parse_array == label_map["left_leg"]).astype(np.float32) + \
                                        (parse_array == label_map["right_leg"]).astype(np.float32) + \
                                        (parse_array == label_map["belt"]).astype(np.float32)
    

        if pair_phase:
            parser_mask_fixed_lower_cloth = np.logical_or(parser_mask_fixed_lower_cloth_ours, parser_mask_fixed_lower_cloth_parser)
        else:
            parser_mask_fixed_lower_cloth = np.logical_and(parser_mask_fixed_lower_cloth_ours, parser_mask_fixed_lower_cloth_parser)
        
        pants_num = np.all(np.array(parse_img2.convert("RGB")) == np.array([0, 128, 0]), axis=-1)
        skirt_num = np.all(np.array(parse_img2.convert("RGB")) == np.array([0, 85, 85]), axis=-1)
        dress_num = np.all(np.array(parse_img2.convert("RGB")) == np.array([0, 0, 85]), axis=-1)

        ## in case benchmark parser does not detect any lower cloth 
        if np.sum(pants_num) == 0 and np.sum(skirt_num) == 0 and np.sum(dress_num) == 0:
            print("[INFO] Not detecting lower at ours")
        
            pants_num_parser = np.all((parse_array == label_map["pants"]).astype(np.float32), axis=-1)
            skirt_num_parser = np.all((parse_array == label_map["skirt"]).astype(np.float32), axis=-1)
            dress_num_parser = np.all((parse_array == label_map["dress"]).astype(np.float32), axis=-1)
            leg_num_parser = np.all((parse_array == label_map["left_leg"]).astype(np.float32) + (parse_array == label_map["right_leg"]).astype(np.float32), axis=-1)

            if np.sum(pants_num_parser) == 0 or np.sum(skirt_num_parser) == 0 or np.sum(dress_num_parser) == 0 or np.sum(leg_num_parser) == 0:
                print("[INFO] Not detecting lower at parser")
                return benchmark_mask, colored_I
            else:
                mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32)
                mask = cv2.dilate(mask, np.ones((3, 3), np.uint16), iterations=5) 
                final_mask = Image.fromarray(mask.astype(np.uint8) * 255)
                
                return final_mask, colored_I

        parser_mask_fixed += parser_mask_fixed_lower_cloth
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        pose_data = keypoint["pose_keypoints_2d"]
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 2))

        im_arms_left = Image.new('L', (width, height))
        im_arms_right = Image.new('L', (width, height))
        arms_draw_left = ImageDraw.Draw(im_arms_left)
        arms_draw_right = ImageDraw.Draw(im_arms_right)

        shoulder_right      = np.multiply(tuple(pose_data[2][:2]), height / 512.0)
        shoulder_left       = np.multiply(tuple(pose_data[5][:2]), height / 512.0)
        elbow_right         = np.multiply(tuple(pose_data[3][:2]), height / 512.0)
        elbow_left          = np.multiply(tuple(pose_data[6][:2]), height / 512.0)
        wrist_right         = np.multiply(tuple(pose_data[4][:2]), height / 512.0)
        wrist_left          = np.multiply(tuple(pose_data[7][:2]), height / 512.0)
        ARM_LINE_WIDTH      = int(arm_width / 512 * height)
        size_left           = [shoulder_left[0] - ARM_LINE_WIDTH // 2, shoulder_left[1] - ARM_LINE_WIDTH //2, shoulder_left[0] + ARM_LINE_WIDTH // 2, shoulder_left[1] + ARM_LINE_WIDTH // 2]
        size_right          = [shoulder_right[0] - ARM_LINE_WIDTH // 2, shoulder_right[1] - ARM_LINE_WIDTH // 2, shoulder_right[0] + ARM_LINE_WIDTH // 2, shoulder_right[1] + ARM_LINE_WIDTH // 2]

        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            im_arms_right = arms_right
        else:
            wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
            arms_draw_right.line(np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_right.arc(size_right, 0, 360,'white', ARM_LINE_WIDTH // 2)

        if wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            im_arms_left = arms_left
        else:
            wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
            arms_draw_left.line(np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_left.arc(size_left, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        hands_left= np.logical_and(np.logical_not(im_arms_left), arms_left)
        hands_right = np.logical_and(np.logical_not(im_arms_right), arms_right)
        hands = np.logical_or(hands_left, hands_right).astype(np.uint8)
        hands_ours = np.isin(I_map, densepose_hands_I).astype(np.uint8)
        
        # Obtain both
        hands = np.logical_and(hands_ours, hands)
        parser_mask_fixed += hands
        parser_mask_fixed_0 += hands_left + hands_right

        parser_mask_fixed = cv2.erode(parser_mask_fixed, np.ones((5, 5), np.uint16), iterations=1)
        parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)

        parser_mask_fixed_0 = cv2.erode(parser_mask_fixed_0, np.ones((5, 5), np.uint16), iterations=1)
        parser_mask_fixed_0 = np.logical_or(parser_mask_fixed_0, parse_head)
        
        parse_mask = cv2.dilate(parse_mask, np.ones((3, 3), np.uint16), iterations=5)
        parse_mask_0 = cv2.dilate(parse_mask_0, np.ones((10, 10), np.uint16), iterations=5)

        # Predict neck
        neck_mask = (parse_array == 18).astype(np.float32)
        neck_mask = cv2.dilate(neck_mask, np.ones((5, 5), np.uint16), iterations=1)
        neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
        parse_mask = np.logical_or(parse_mask, neck_mask)
        parse_mask_0 = np.logical_or(parse_mask_0, neck_mask)
        
        # Predict arms
        if not pair_phase:
            arm_mask = cv2.dilate(np.logical_or(densepose_target_I_mask, parser_mask_arms).astype('float32'), np.ones((5, 5), np.uint16), iterations=10)
            arm_mask_0 = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), np.ones((5, 5), np.uint16), iterations=4)
        else:
            arm_mask_0 = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), np.ones((5, 5), np.uint16), iterations=4)
            arm_mask = cv2.dilate(np.logical_or(densepose_target_I_mask, parser_mask_arms).astype('float32'), np.ones((5, 5), np.uint16), iterations=7)

        parse_mask = np.logical_or(parse_mask, arm_mask)
        parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))

        parse_mask_0 = np.logical_or(parse_mask_0, arm_mask_0)
        parse_mask_0 = np.logical_and(parser_mask_changeable_0, np.logical_not(parse_mask_0))

        parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
        inpaint_mask = 1 - parse_mask_total
        img = np.where(inpaint_mask, 255, 0)
        dst = hole_fill(img.astype(np.uint8))
        dst = refine_mask(dst)
        dst[hands] = 0

        parse_mask_total_0 = np.logical_or(parse_mask_0, parser_mask_fixed_0)
        inpaint_mask_0 = 1 - parse_mask_total_0
        img_0 = np.where(inpaint_mask_0, 255, 0)
        dst_0 = hole_fill(img_0.astype(np.uint8))
        dst_0 = refine_mask(dst_0)
        inpaint_mask_0 = dst_0 / 255 * 1
        mask_0 = Image.fromarray(inpaint_mask_0.astype(np.uint8) * 255)

        agnostic_mask_0 =  (dst_0 > 0).astype(np.uint8)
        agnostic_mask = (dst > 0).astype(np.uint8)

        if pair_phase:
            combined_mask = np.logical_and(agnostic_mask_0, agnostic_mask).astype(np.uint8)
            agnostic_mask = Image.fromarray(combined_mask.astype(np.uint8) * 255)
        else:
            agnostic_mask = Image.fromarray(agnostic_mask.astype(np.uint8) * 255)

        return agnostic_mask, colored_I

    else:
        raise ValueError("category must be \'upper_body\' in vton-hd!")

            

            




