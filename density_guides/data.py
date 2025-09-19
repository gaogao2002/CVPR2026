import os
from typing import Optional, Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from diffusers.image_processor import VaeImageProcessor

class FSC147OneCountTensor(Dataset):
    """
    固定只加载一个数量 (count)。
    返回:
      {
        'pixel_values': FloatTensor [3,H,W]  由 diffusers 的 VaeImageProcessor 预处理，范围约 [-1,1]
        'density':      FloatTensor [1,H,W]  单通道密度图
        'Prompt':       str                  "A photo of {class_name}  {count} objects"
      }
    不返回路径、不返回类别字符串。
    """
    def __init__(
        self,
        root: str,                       # 数据集根目录，例如 "./data/FSC147"
        count: int,                      # 此时需要训练的数量token
        Height: int = 1024,               # 数据集默认高度
        Width: int = 1024,                # 数据集默认宽度
        strict_match: bool = True,       # 仅接受与 image 同名的 density
        keep_density_sum: bool = True,   # resize 后是否保持密度积分不变（建议 True）
        valid_exts: Tuple[str,...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
    ):
        self.root = root
        self.count = int(count)
        self.strict_match = strict_match
        self.keep_density_sum = keep_density_sum
        self.valid_exts = tuple(e.lower() for e in valid_exts)
        
        self.Height, self.Width = Height, Width
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.samples: List[Tuple[str, str, int]] = []  # (img_path, den_path, class_id)

        if not self.samples:
            raise RuntimeError(f"No samples for count={self.count} under {self.root}")
        self.Height, self.Width = Height, Width

    def __len__(self):
        return len(self.samples)

    def _preprocess_image(self, img_pil: Image.Image) -> torch.FloatTensor:
        # 用 diffusers 的 VaeImageProcessor 做 resize/normalize -> [-1,1]
        # 这里假设你初始化 image_processor 时已设置了 do_resize/size/vae_scale_factor 等
        px = self.image_processor.preprocess(img_pil)  # 返回 [C,H,W] 或 [B,C,H,W]
        if px.dim() == 4:
            px = px[0]
        return px

    def _preprocess_density(self, den_pil: Image.Image, orig_size: Tuple[int,int]) -> torch.FloatTensor:
        # orig_size 必须是 (H0, W0)
        H0, W0 = orig_size
        H, W = self.height, self.width  # 统一只用这两个名字

        # 灰度化
        if den_pil.mode != "L":
            den_pil = den_pil.convert("L")

        # 原像素和（float32），用于“保持积分”
        arr0 = np.asarray(den_pil, dtype=np.float32)     # [H0, W0]，范围可能是 0..255 或其它
        sum0 = float(arr0.sum())

        if (H0, W0) != (H, W):
            # 注意 PIL.resize 的入参是 (W, H)
            den_resized = den_pil.resize((W, H), resample=Image.BILINEAR)
            arr = np.asarray(den_resized, dtype=np.float32)
            if self.keep_density_sum and sum0 > 0:
                sum1 = float(arr.sum())
                if sum1 > 0:
                    arr *= (sum0 / sum1)   # 用“像素和比”保持积分不变
        else:
            arr = arr0  # 尺寸相同，直接用原图

        # 如果你**不**想保持积分，而是希望标准化到 [0,1] 灰度，可在这里处理：
        if not getattr(self, "keep_density_sum", True) and getattr(self, "density_as_uint8", False):
            arr = arr / 255.0

        den_t = torch.from_numpy(arr).unsqueeze(0).contiguous()   # [1,H,W] float32
        return den_t

    def __getitem__(self, idx: int):
        img_path, den_path, class_name = self.samples[idx]

        # 读图
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        H0, W0 = img.size[1], img.size[0]  # PIL 的 size 是 (W, H)，注意顺序

        # 用 diffusers 的 image processor 处理原图
        pixel_values = self._preprocess_image(img)  # [3,H,W], float32, ~[-1,1]

        # 处理 density（按原图尺寸做保持积分缩放）
        den = Image.open(den_path)
        density = self._preprocess_density(den, orig_size=(H0, W0))  # [1,H,W]
        prompt = f"A photo of {self.count} {class_name}"

        return {
            "pixel_values": pixel_values,              # [3,H,W] float32
            "density": density,                        # [1,H,W] float32
            "Prompt": prompt,                          # str, "A photo of {class_name}  {count} objects"
        }
