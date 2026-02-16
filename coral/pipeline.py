from diffusers import FluxTransformer2DModel, FluxFillPipeline
from dataclasses import dataclass
from diffusers.utils import is_torch_xla_available, BaseOutput
from diffusers.pipelines.flux.pipeline_flux_fill import FluxPipelineOutput, calculate_shift, retrieve_timesteps, retrieve_latents
import PIL.Image
import torch
from typing import Any, Callable, Dict, List, Optional, Union
import inspect
import numpy as np
from PIL import Image, ImageFilter

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
 
class CORALPipeline(FluxFillPipeline):    
    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        pose_image,
        batch_size,
        num_channels_latents,
        num_images_per_prompt,
        height,
        width,
        dtype,
        device,
        generator
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        if masked_image.shape[1] == num_channels_latents:
            masked_image_latents = masked_image
        else:
            masked_image_latents = retrieve_latents(self.vae.encode(masked_image), generator=generator)

        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        if pose_image.shape[1] == num_channels_latents:
            pose_image_latents = pose_image
        else:
            pose_image_latents = retrieve_latents(self.vae.encode(pose_image), generator=generator)

        pose_image_latents = (pose_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        pose_image_latents = pose_image_latents.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)
        if pose_image_latents.shape[0] < batch_size:
            if not batch_size % pose_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {pose_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            pose_image_latents = pose_image_latents.repeat(batch_size // pose_image_latents.shape[0], 1, 1, 1)

        masked_image_latents = self._pack_latents(
            masked_image_latents,
            batch_size,
            num_channels_latents,
            height,
            width,
        )

        pose_image_latents = self._pack_latents(
            pose_image_latents,
            batch_size,
            num_channels_latents,
            height,
            (width)//3,
        )

        mask = mask[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
        mask = mask.view(
            batch_size, height, self.vae_scale_factor, width, self.vae_scale_factor
        )  # batch_size, height, 8, width, 8
        mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
        mask = mask.reshape(
            batch_size, self.vae_scale_factor * self.vae_scale_factor, height, width
        )  # batch_size, 8*8, height, width
        
        mask = self._pack_latents(
            mask,
            batch_size,
            self.vae_scale_factor * self.vae_scale_factor,
            height,
            width,
        )
        mask = mask.to(device=device, dtype=dtype)

        return mask, masked_image_latents, pose_image_latents

    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            image: Optional[torch.FloatTensor] = None,
            pose_image: Optional[torch.FloatTensor]=None,
            mask_image: Optional[torch.FloatTensor] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            strength: float = 1.0,
            num_inference_steps: int = 50,
            sigmas: Optional[List[float]] = None,
            guidance_scale: float = 30.0,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 512,
            agnostic_mask: Optional[torch.FloatTensor] = None,
            model_img: Optional[torch.FloatTensor] = None,
    ):
        full_height = height or self.default_sample_size * self.vae_scale_factor
        full_width = width or self.default_sample_size * self.vae_scale_factor
        compression_ratio = self.vae_scale_factor * 2

        if pose_image is None:
            raise ValueError("CORALPipeline requires `pose_image` (cannot be None).")
        if full_width % 3 != 0:
            raise ValueError("full_width must be divisible by 3")
        if full_height % compression_ratio != 0:
            raise ValueError("full_height must be divisible by {}".format(compression_ratio))
        
        tile_width = full_width // 3

        if tile_width % compression_ratio != 0:
            raise ValueError("tile_width must be divisible by {}".format(compression_ratio))
    
        latent_h = full_height // compression_ratio
        latent_w = tile_width // compression_ratio

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            strength,
            full_height,
            full_width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            image=image,
            mask_image=mask_image
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        init_image = self.image_processor.preprocess(image, height=full_height, width=tile_width * 2)
        init_image = init_image.to(dtype=torch.float32)

        pose_image = self.image_processor.preprocess(pose_image, height=full_height, width=tile_width)
        pose_image = pose_image.to(dtype=torch.float32)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Prepare prompt embeddings
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = (int(full_height) // self.vae_scale_factor // 2) * (int(full_width) // self.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents, latent_image_ids = self.prepare_latents(
            init_image,
            latent_timestep,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            full_height,
            tile_width*2,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        pose_image_ids = latent_image_ids.clone()
        pose_image_ids = pose_image_ids.reshape(latent_h, latent_w * 2, -1)
        pose_image_ids = pose_image_ids[:, latent_w :, :]

        latent_image_ids = latent_image_ids.reshape(latent_h, latent_w * 2 , -1)
        concat_image_ids = torch.cat([latent_image_ids, pose_image_ids], dim=1)
        concat_image_ids = concat_image_ids.reshape(concat_image_ids.shape[0] * concat_image_ids.shape[1], -1)

        # 6. Prepare mask and masked image latents
        mask_image = self.mask_processor.preprocess(mask_image, height=full_height, width=full_width)
        masked_image = init_image * (1 - mask_image[:,:,:,:(tile_width * 2)])
        masked_image = masked_image.to(device=device, dtype=prompt_embeds.dtype)

        pose_image = pose_image.to(device=device, dtype=prompt_embeds.dtype)
        
        masked_image = torch.cat([masked_image, torch.zeros_like(pose_image)],dim=-1)
        height_, width_ = masked_image.shape[-2:]
        mask, masked_image_latents, pose_image_latents = self.prepare_mask_latents(
            mask_image,
            masked_image,
            pose_image,
            batch_size,
            num_channels_latents,
            num_images_per_prompt,
            height_,
            width_,
            prompt_embeds.dtype,
            device,
            generator,
        )
        masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        latents = latents.reshape(latents.shape[0], latent_h, latent_w * 2, -1)
        pose_image_latents = pose_image_latents.reshape(pose_image_latents.shape[0], latent_h, latent_w , -1)
        latents = torch.cat([latents, pose_image_latents], dim=2)
        latents = latents.reshape(latents.shape[0], latents.shape[1] * latents.shape[2], -1)

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                noise_pred = self.transformer(
                    hidden_states=torch.cat((latents, masked_image_latents), dim=2),
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=concat_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 8. Post-process the image
        if output_type == "latent":
            image = latents

        else:
            latents = latents.reshape(latents.shape[0], latent_h, latent_w * 3, -1)
            latents = latents[:,:,:latent_w * 2,:]
            latents = latents.reshape(latents.shape[0], latents.shape[1] * latents.shape[2], -1)
            latents = self._unpack_latents(latents, full_height, tile_width * 2, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        
        # Offload all models
        self.maybe_free_model_hooks()

        w, h = image[0].size[0], image[0].size[1]
        right_person = image[0].crop((w // 2, 0, w, h))

        kernel_size = h // 100
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        agnostic_mask = agnostic_mask[0].detach().cpu().numpy()
        agnostic_mask = Image.fromarray((agnostic_mask[0] * 255).astype(np.uint8), mode="L")
        agnostic_mask = agnostic_mask.filter(ImageFilter.GaussianBlur(kernel_size))
        model_np = model_img.cpu().numpy().astype(np.float32)
        out_np = np.array(right_person).astype(np.float32)
        agnostic_mask_np = np.array(agnostic_mask).astype(np.float32) / 255.0

        if agnostic_mask_np.ndim == 2:
            agnostic_mask_np = agnostic_mask_np[:, :, None]
        if agnostic_mask_np.shape[2] != 1 and agnostic_mask_np.shape[2] != 3:
            agnostic_mask_np = agnostic_mask_np[:, :, :1]
        
        final_result = model_np * (1.0 - agnostic_mask_np) + out_np * agnostic_mask_np
        final_result = np.clip(final_result, 0, 255).astype(np.uint8)
        image = [PIL.Image.fromarray(final_result.squeeze(0))]

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)