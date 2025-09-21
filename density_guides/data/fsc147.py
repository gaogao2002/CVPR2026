import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import cv2
import torch

class FSC147OneCount(Dataset):
    """
    固定只加载一个数量 (count)。
    返回:
      {
        'pixel_values': FloatTensor [3,H,W]  由 diffusers 的 VaeImageProcessor 预处理，范围约 [-1,1]
        'density':      FloatTensor [1,H,W]  单通道密度图
        'Prompt':       str                  "A photo of {class_name} {count} objects"
      }
    不返回路径、不返回类别字符串。
    """
    def __init__(
        self,
        root: str,                       # 数据集根目录，例如 "./data/FSC147"
        count: int,                      # 此时需要训练的数量token
        keep_density_acc: bool = False,   # resize 后是否使用高斯核卷积（建议 True）
        placeholder_token : str = None,
    ):
        self.root = root
        self.count = int(count)

        self.keep_density_acc = keep_density_acc
        self.placeholder_token = placeholder_token
        
        self.image_tranform = transforms.Compose([
            transforms.Resize((384, 384)),           # 调整大小
            transforms.ToTensor(),                   # 转换为 tensor
            transforms.Normalize(                    # 标准化
                mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5]
        )
        ]) 
        self.samples = self.__get_data()
    
    def __get_data(self,):
        def sort_by_id(files):
            return sorted(
                files,
                key=lambda x: int(x.split("_")[-1].split(".")[0])  # 取最后一段数字
            )
        image_dir = os.path.join(self.root, f"{self.count}", "image")
        density_dir = os.path.join(self.root, f"{self.count}", "density")

        image_files = sort_by_id(os.listdir(image_dir))
        density_files = sort_by_id(os.listdir(density_dir))

        assert len(image_files) == len(density_files), "数量不一致！"

        image_paths = [os.path.join(image_dir, f) for f in image_files]
        density_paths = [os.path.join(density_dir, f) for f in density_files]
        class_names = [f.split("_")[1] for f in image_files]

        samples = list(zip(image_paths, density_paths, class_names))
        return samples

    
            
    def __len__(self):
        return len(self.samples)
    
    def _preprocess_density_map(self, density_array, new_h=384, new_w=384):
        """
        density: numpy array, shape [H, W] (float), sum(density) ≈ count
        new_h, new_w: target size
        """
        density_matrix = np.reshape(density_array,(-1,384)) 
        old_h, old_w = density_matrix.shape
        
        # resize
        resized = cv2.resize(density_matrix, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 关键：保持总和一致（按面积比例缩放）
        scale_factor = (old_h * old_w) / (new_h * new_w)
        resized = resized * scale_factor
        density_tensor = torch.from_numpy(resized)
        density_tensor = density_tensor.unsqueeze(0)
        return density_tensor
    
    def _preprocess_density_map_acc(self,density, new_h=384, new_w=384):
        return 0
        

    def __getitem__(self, idx: int):
        img_path, den_path, class_name = self.samples[idx]

        # 读图
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        pixel_value = self.image_tranform(img)
    

        # 处理 density（按原图尺寸做保持积分缩放）
        den = np.load(den_path)
        
        if self.keep_density_acc:
            density = self._preprocess_density_map_acc(den)  # [1,H,W]
        else:
            density = self._preprocess_density_map(den)  # [1,H,W]
            
        prompt = f"A photo of {self.placeholder_token} {self.count} {class_name}"
        prompt_init = f"A photo of {self.count} {class_name}"

        return {
            "pixel_values": pixel_value.unsqueeze(0),              # [B,3,H,W] float32
            "density": density.unsqueeze(0),                        # [B,1,H,W] float32
            "Prompt": prompt,                          # str, "A photo of {count} {class_name}"
            "Prompt_init" : prompt_init,
        }


def collate_fn(examples):
    images = torch.cat([example["pixel_values"] for example in examples],dim=0)
    density_maps = torch.cat([example["density"] for example in examples],dim=0)
    prompts = [example["Prompt"] for example in examples]
    prompts_init = [example["Prompt_init"] for example in examples]

    return {
        "pixel_values": images,              # [B,3,H,W] float32
        "density_maps": density_maps,                        # [B,1,H,W] float32
        "prompts": prompts,                          # list of str, "A photo of {count} {class_name}"
        "prompts_init" : prompts_init
    }