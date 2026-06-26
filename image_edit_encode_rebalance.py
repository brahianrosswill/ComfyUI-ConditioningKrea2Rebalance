import math

import torch

try:
    import comfy.utils
    import node_helpers
    _COMFY_AVAILABLE = True
except ImportError:
    _COMFY_AVAILABLE = False


def _unit_norm_dim(t, eps=1e-8):
    dtype = t.dtype
    t = t.float()
    norm = torch.sqrt(t.pow(2).sum(dim=-1, keepdim=True) + eps)
    return (t / norm).to(dtype)


def _split_bands(t, n_bands=12):
    flat = t.shape[-1]
    if n_bands > 1 and flat % n_bands == 0:
        d = flat // n_bands
        return t.view(*t.shape[:-1], n_bands, d), d
    return None, None


def _merge_bands(t):
    n_bands = t.shape[-2]
    d = t.shape[-1]
    return t.reshape(*t.shape[:-2], n_bands * d)


def _extract_cond_tensor(item):
    if isinstance(item, (list, tuple)) and len(item) == 2 \
            and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
        return item[0]
    if isinstance(item, torch.Tensor):
        return item
    return None


def _match_batch(ref_dir, target_batch):
    if ref_dir.shape[0] == 1 and target_batch != 1:
        return ref_dir.expand(target_batch, *ref_dir.shape[1:])
    if ref_dir.shape[0] != target_batch:
        ref_dir = ref_dir.mean(dim=0, keepdim=True).expand(target_batch, *ref_dir.shape[1:])
    return ref_dir


def _parse_floats(s):
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        vals = [float(x) for x in s.replace(";", ",").split(",") if x.strip() != ""]
    except ValueError:
        return None
    if len(vals) < 2:
        return None
    return vals

# ignored
SYS_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def compile_edit(clip, vae, prompt, image, strength=1.0):
    if not _COMFY_AVAILABLE:
        raise RuntimeError("Krea 2 Edit requires ComfyUI (comfy.utils, node_helpers).")

    images_vl = None
    combined_latents = None
    if image is not None:
        samples = image.movedim(-1, 1)  # NHWC -> NCHW

        PIXEL_B = 1_843_200
        src_pixels = samples.shape[3] * samples.shape[2]
        if src_pixels > PIXEL_B:
            scale_by = math.sqrt(PIXEL_B / src_pixels)
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)
            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
        else:
            s = samples
        images_vl = [s.movedim(1, -1)]  # back to NHWC for clip.tokenize

        latent = vae.encode(samples.movedim(1, -1)[:, :, :, :3])
        combined_latents = [latent * strength]

        image_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
        full_prompt = image_prompt + prompt
    else:
        full_prompt = prompt

    tokens = clip.tokenize(
        full_prompt,
        images=images_vl,
        llama_template=SYS_TEMPLATE,
    )
    conditioning = clip.encode_from_tokens_scheduled(tokens)

    if combined_latents is not None:
        conditioning = node_helpers.conditioning_set_values(
            conditioning,
            {"reference_latents": combined_latents},
            append=True,
        )

    return conditioning


def _scale_cond_tensor(t, scale, weights=None):
    if weights is None:
        return t * scale

    flat = t.shape[-1]
    n_layers = len(weights)
    if n_layers > 1 and flat % n_layers == 0:
        layer_dim = flat // n_layers
        orig_dtype = t.dtype
        t = t.float()
        t = t.view(*t.shape[:-1], n_layers, layer_dim)
        gains = torch.tensor(weights, dtype=t.dtype, device=t.device)
        t = t * gains.view(*([1] * (t.dim() - 2)), n_layers, 1)
        t = t.view(*t.shape[:-2], flat)
        return t.to(orig_dtype) * scale
    return t * scale


def scale_conditioning(structure, scale, weights=None):
    if isinstance(structure, list):
        out = []
        for item in structure:
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_t, extras = item
                new_cond = _scale_cond_tensor(cond_t, scale, weights)
                out.append([new_cond, dict(extras)])
            else:
                out.append(scale_conditioning(item, scale, weights))
        return out
    if isinstance(structure, torch.Tensor):
        return _scale_cond_tensor(structure, scale, weights)
    if isinstance(structure, dict):
        return {k: scale_conditioning(v, scale, weights)
                for k, v in structure.items()}
    return structure


def refocus(conditioning, scale, weights):
    plw = _parse_floats(weights) if weights else None
    return scale_conditioning(conditioning, scale, weights=plw)


def _project_dissim_per_band(cond_bands, ref_bands, d, n_bands, strength, per_band_strengths, sign):
    b = cond_bands.shape[0]
    cond_mean = cond_bands.float().mean(dim=1)
    ref_mean = ref_bands.float().mean(dim=1)
    ref_mean = _match_batch(ref_mean, b)
    direction = _unit_norm_dim(cond_mean - ref_mean)

    if per_band_strengths is None:
        gains = [strength] * n_bands
    else:
        gains = list(per_band_strengths)
        if len(gains) < n_bands:
            gains = gains + [strength] * (n_bands - len(gains))
        elif len(gains) > n_bands:
            gains = gains[:n_bands]

    gains_t = torch.tensor(gains, dtype=cond_bands.float().dtype, device=cond_bands.device)
    gains_t = gains_t.view(1, 1, n_bands, 1)

    cond_f = cond_bands.float()
    dir_exp = direction.unsqueeze(1)
    proj = (cond_f * dir_exp).sum(dim=-1, keepdim=True)
    out = cond_f + sign * gains_t * proj * dir_exp
    return _merge_bands(out.to(cond_bands.dtype))


def _project_dissim_whole(cond_t, ref_t, strength, sign):
    b = cond_t.shape[0]
    cond_mean = cond_t.float().mean(dim=1, keepdim=True)
    ref_mean = ref_t.float().mean(dim=1, keepdim=True)
    ref_mean = _match_batch(ref_mean, b)
    direction = _unit_norm_dim(cond_mean - ref_mean)
    proj = (cond_t.float() * direction).sum(dim=-1, keepdim=True)
    out = cond_t.float() + sign * strength * proj * direction
    return out.to(cond_t.dtype)


def _apply_dissim(cond_t, ref_t, strength, per_band_strengths, n_bands=12):
    cond_bands, d = _split_bands(cond_t, n_bands)
    ref_bands, d2 = _split_bands(ref_t, n_bands)
    if cond_bands is not None and ref_bands is not None and d == d2:
        return _project_dissim_per_band(cond_bands, ref_bands, d, n_bands, strength, per_band_strengths, sign=+1)
    return _project_dissim_whole(cond_t, ref_t, strength, sign=+1)


def dissim_guidance_conditioning(structure, ref_structure, strength, per_band_strengths=None):
    if isinstance(structure, list):
        out = []
        ref_iter = iter(ref_structure) if isinstance(ref_structure, list) else None
        for item in structure:
            ref_item = next(ref_iter, None) if ref_iter is not None else None
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_t, extras = item
                ref_t = _extract_cond_tensor(ref_item) if ref_item is not None else None
                new_cond = _apply_dissim(cond_t, ref_t, strength, per_band_strengths) \
                    if ref_t is not None else cond_t
                out.append([new_cond, dict(extras)])
            else:
                out.append(dissim_guidance_conditioning(item, ref_item, strength, per_band_strengths))
        return out
    if isinstance(structure, torch.Tensor):
        ref_t = _extract_cond_tensor(ref_structure) if ref_structure is not None else None
        if ref_t is not None:
            return _apply_dissim(structure, ref_t, strength, per_band_strengths)
        return structure
    return structure


def guidance(conditioning, reference, strength):
    return dissim_guidance_conditioning(conditioning, reference, strength, per_band_strengths=None)


class Krea2EditRebalance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP",),
            "vae": ("VAE",),
        },
        "optional": {
            "image": ("IMAGE",),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning"

    @staticmethod
    def _process_cond(cond):
        cond_ref = refocus(
            cond, 4.00, "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
        )
        cond_main = refocus(
            cond, 4.00, "0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,9.0,1.0,1.0,1.0",
        )
        return guidance(cond_main, cond_ref, 0.500)

    def main(self, text, clip, vae, image=None):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Krea 2 Edit requires ComfyUI (comfy.utils, node_helpers).")

        prompt = "(Subject:2) {}".format(text)

        cond_text = compile_edit(clip, vae, prompt, None, 1.0)
        cond_text = self._process_cond(cond_text)
        cond_text = node_helpers.conditioning_set_values(
            cond_text, {"start_percent": 0.000, "end_percent": 0.175},
        )

        cond_image = compile_edit(clip, vae, prompt, image, 1.0)
        cond_image = self._process_cond(cond_image)
        cond_image = node_helpers.conditioning_set_values(
            cond_image, {"start_percent": 0.175, "end_percent": 1.000},
        )

        final = cond_text + cond_image

        return (final,)


NODE_CLASS_MAPPINGS = {
    "Krea2EditRebalance": Krea2EditRebalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2EditRebalance": "Krea 2 Image Edit Rebalance",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
