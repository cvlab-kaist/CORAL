from coral.pipeline import CORALPipeline
from coral.dataset.vt_dataset_test import VitonHDTestDataset
from coral.dataset.dc_dataset_test import DressCodeTestDataset
import argparse
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers import FluxTransformer2DModel
import torch
from tqdm import tqdm
import os
from accelerate.utils import set_seed
os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
os.environ.pop("PYTORCH_SHOW_CPP_STACKTRACES", None)
os.environ.pop("TORCH_SHOW_CPP_STACKTRACES", None)

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"):
    
    text_encoder_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, revision=revision)
    model_class = text_encoder_config.architectures[0]
    
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    
    else:
        raise ValueError(f"{model_class} is not supported.")

def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant)
    text_encoder_two = class_two.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant)
    return text_encoder_one, text_encoder_two

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--coral_model_path",
        type=str,
        required=True,
        help="Path to pretrained coral model.",
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="bf16",
        required=True,
        help="weight dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument(
        "--width", type=int, default=768, help="The width for generated image"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="The height for generated image"
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--data_list", type=str, help="The data list file for inference")
    parser.add_argument(
        "--dataroot",
        type=str,
        help=(
            'The dataset dir'
        ),
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        help=(
            'The dataset type'
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The output directory where the model predictions will be written",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

def setup_pipeline(args, model_path, weight_dtype, device):
    print("[INFO] : Load Tokenizers")
    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision,)
    tokenizer_two = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path,subfolder="tokenizer_2", revision=args.revision,)

    print("[INFO] : Load Text encoders")
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2")
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)

    print("[INFO] : Load Image encoders")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant,)
    vae_scale_factor = (2 ** (len(vae.config.block_out_channels) - 1) if vae is not None else 8)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_resize=True, do_convert_rgb=True, do_normalize=True)
    mask_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_resize=True, do_convert_grayscale=True, do_normalize=False, do_binarize=True,)

    print("[INFO] : Load Transformer")
    transformer = FluxTransformer2DModel.from_pretrained(model_path, subfolder="transformer", revision=args.revision, variant=args.variant, torch_dtype=torch.bfloat16,)

    if args.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif args.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    
    transformer.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder_one.to(device, dtype=weight_dtype)
    text_encoder_two.to(device, dtype=weight_dtype)

    pipeline = CORALPipeline.from_pretrained(args.pretrained_model_name_or_path, 
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        torch_dtype=weight_dtype,
    ).to(device)

    return pipeline, weight_dtype

def main(args):
    pipeline, weight_dtype = setup_pipeline(
        args,
        model_path=args.coral_model_path,
        weight_dtype=args.weight_dtype,
        device=args.device,
    )
    if args.seed is not None:
        set_seed(args.seed)

    if args.dataset_type == "vt":
        test_dataset = VitonHDTestDataset(
            dataroot_path=args.dataroot,
            size=(args.height, args.width),
            data_list=args.data_list,
        )
    else:
        test_dataset = DressCodeTestDataset(
            dataroot_path=args.dataroot,
            size=(args.height, args.width),
            data_list=args.data_list,
        )
    
    total_len = len(test_dataset)
    print(f"[INFO] : Load datasets with {args.dataset_type} which has {total_len} data")

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, batch in enumerate(tqdm(dataloader)):

        # Prompt brought from : https://github.com/nftblackmagic/catvton-flux
        if args.dataset_type == "vt":
            prompt = ["" 
                f"The pair of images highlights a clothing and its styling on a model, high resolution, 4K, 8K; "
                f"[IMAGE1] Detailed product shot of a clothing"
                f"[IMAGE2] The same cloth is worn by a model in a lifestyle setting."
            ]
        elif args.dataset_type == "dc":
            category = batch['category'][0]
            if category == "dresses":
                prompt = ["" 
                    f"The pair of images highlights a clothing and its styling on a model, high resolution, 4K, 8K; "
                    f"[IMAGE1] Detailed product shot of a dresses clothing"
                    f"[IMAGE2] The same dresses cloth is worn by a model in a lifestyle setting."
                ]
            elif category == "lower_body":
                prompt = ["" 
                    f"The pair of images highlights a clothing and its styling on a model, high resolution, 4K, 8K; "
                    f"[IMAGE1] Detailed product shot of a lower_body clothing"
                    f"[IMAGE2] The same lower_body cloth is worn by a model in a lifestyle setting."
                ]
            elif category == "upper_body":
                prompt = ["" 
                    f"The pair of images highlights a clothing and its styling on a model, high resolution, 4K, 8K; "
                    f"[IMAGE1] Detailed product shot of a upper_body clothing"
                    f"[IMAGE2] The same upper_body cloth is worn by a model in a lifestyle setting."
                ]


        tgt_image = batch['tgt_image']
        binary_diptych    = batch['binary_diptych']
        pose_image       = batch['pose_img']
        model_name       = batch['im_name'][0]
        model_img = batch['model_img']
        agnostic_mask  = batch['agnostic_mask']

        output_file_name = model_name.split(".")[0]
        save_path = os.path.join(save_dir, f"{output_file_name}.png")

        if os.path.exists(save_path):
            print(f"[INFO] : {save_path} already exists. Skip.")
            continue

        with torch.inference_mode():
            img = pipeline(
                prompt=prompt,
                height=args.height,
                width=args.width * 3,
                image=tgt_image,
                mask_image=binary_diptych,
                num_inference_steps=28,
                generator=torch.Generator(device=args.device).manual_seed(args.seed) if args.seed else None,
                guidance_scale=30,
                pose_image=pose_image,
                agnostic_mask=agnostic_mask,
                model_img=model_img
            ).images

        result_img = img[0]
        result_img.save(save_path)
        print(f"[INFO] : Saved at {save_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)