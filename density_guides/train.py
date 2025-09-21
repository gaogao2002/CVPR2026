from pathlib import Path
import itertools
from accelerate import Accelerator
from transformers import YolosForObjectDetection, YolosImageProcessor

from data import FSC147OneCount,collate_fn
from diffusers.pipelines import AutoPipelineForText2Image
import time
import numpy as np
import cv2
import torchvision.transforms.functional as TF
import os
import utils
import shutil
import torch
torch.autograd.set_detect_anomaly(True)
from tqdm import *
from collections import namedtuple
import argparse
from math import ceil
import torch.nn.functional as F

def main(args):

    # device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Counting model-clip count 用来输出密度图
    counting_model = utils.prepare_counting_model(device) # clip-count

    
    # Define dataloades
    fsc_dataset = FSC147OneCount(args.data_root_path,args.count,args.keep_density_acc,args.placeholder_token)
    train_dataloader = torch.utils.data.DataLoader(
        fsc_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 准备模型
    
    pipeline = AutoPipelineForText2Image.from_pretrained(
        args.stable_diffusion_pipeline_path,
        torch_dtype=torch.float32
    ).to(device)

    unet, vae, text_encoder, text_encoder_2, scheduler, tokenizer, tokenizer_2 = (
        pipeline.unet,
        pipeline.vae,
        pipeline.text_encoder,
        pipeline.text_encoder_2,
        pipeline.scheduler,
        pipeline.tokenizer,
        pipeline.tokenizer_2,
    )

    '''
    往 tokenizer 里加一个新的“占位符 token”，并把它的词向量初始化成一个已有 token 的词向量。
    这在 Textual Inversion / DreamBooth 等任务里很常见。
    '''
    
    # 在两个 tokenizer 里都加占位符，并记录各自的 placeholder id
    
    placeholder_id_1 = utils.add_placeholder_for(tokenizer,   text_encoder,   args.placeholder_token, args.initializer_token)
    placeholder_id_2 = utils.add_placeholder_for(tokenizer_2, text_encoder_2, args.placeholder_token, args.initializer_token)
    

    ## Freeze vae and unet
    utils.freeze_params(vae.parameters())
    utils.freeze_params(unet.parameters())

    ## Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
        text_encoder_2.text_model.encoder.parameters(),
        text_encoder_2.text_model.final_layer_norm.parameters(),
        text_encoder_2.text_model.embeddings.position_embedding.parameters(),
        text_encoder_2.text_projection.parameters()
    )
    
    utils.freeze_params(params_to_freeze)

    optimizer = torch.optim.AdamW(
        itertools.chain(
            text_encoder.text_model.embeddings.token_embedding.parameters(),
            text_encoder_2.text_model.embeddings.token_embedding.parameters(),
        ),
        lr=args.lr, betas=(args.betas1, args.betas2),
        weight_decay=args.weight_decay, eps=args.eps,
    )
    
    criterion = torch.nn.MSELoss()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    if args.gradient_checkpointing:
        text_encoder_2.gradient_checkpointing_enable()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    text_encoder, text_encoder_2, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, text_encoder_2, optimizer, train_dataloader
    )


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    

    # Move vae and unet to device
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    counting_model = counting_model.to(accelerator.device,dtype=weight_dtype)
    text_encoder   = text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2 = text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # Keep vae in eval mode as we don't train it
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    unet.train()
    
    
    num_samples = len(fsc_dataset)
    steps_per_epoch = ceil(num_samples / args.batch_size)
    total_steps = steps_per_epoch * args.num_train_epochs

    global_step = 0
    
    pbar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)

    # ---------- 早停参数 ----------
    patience = getattr(args, "early_stopping_patience", 5)   # 连续多少个epoch不提升就停
    min_delta = getattr(args, "early_stopping_min_delta", 0.0)  # 至少提升这么多才算提升
    best_metric = float("inf")
    bad_epochs = 0
    
    # Define token output dir
    token_dir_path = f"token/{args.count}"
    token_path = f"{args.output_path}/{token_dir_path}/{args.hyperparam}"
    if not os.path.exists(token_path):
        os.makedirs(token_path)
 
    for epoch in range(args.num_train_epochs):
        
        generator = torch.Generator(device=device)
        epoch_loss = 0
        n_batches = 0
        sched  = scheduler            # 或者用 pipeline.scheduler
        scale  = float(getattr(vae.config, "scaling_factor", 0.13025))  # SDXL 默认 0.13025
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder,text_encoder_2):

                # 1) 目标图像（[-1,1]）→ VAE 编码
                x = batch["pixel_values"].to(device, dtype=weight_dtype)  
                H, W = x.shape[-2], x.shape[-1]# [B,3,H,W] in [-1,1]
                posterior = vae.encode(x)                                        # 保留计算图，别 no_grad / detach
                z = posterior.latent_dist.sample() * scale                       # [B,4,h,w]

                # 2) 采样时间步 & 加噪
                B = z.size(0)
                t = torch.randint(0, sched.config.num_train_timesteps, (B,), device=device).long()
                noise = torch.randn_like(z)
                z_t = sched.add_noise(z, noise, t)

                # 3) SDXL 文本条件（encode_prompt + time_ids）
                pipeline.text_encoder   = text_encoder
                pipeline.text_encoder_2 = text_encoder_2    
                enc_out = pipeline.encode_prompt(
                    batch["prompts"],
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=None,
                )
                # 兼容不同 diffusers 版本的返回
                if isinstance(enc_out, (list, tuple)) and len(enc_out) == 4:
                    prompt_embeds, _, pooled_embeds, _ = enc_out
                elif isinstance(enc_out, (list, tuple)) and len(enc_out) == 2:
                    prompt_embeds, pooled_embeds = enc_out
                else:
                    raise RuntimeError(f"Unexpected encode_prompt outputs for SDXL: {type(enc_out)}")
                
                add_time_ids = torch.tensor([H, W, 0, 0, H, W], dtype=prompt_embeds.dtype, device=device).unsqueeze(0).repeat(B, 1)
               
                added = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}

                # 4) UNet 预测 ε 或 v
                pred = unet(z_t, t, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added).sample

                # 5) 噪声损失（noise_loss）
                ptype = getattr(sched.config, "prediction_type", "epsilon")
                if ptype in ("epsilon", "eps"):
                    target = noise
                    eps_hat = pred
                elif ptype in ("v", "v_prediction"):
                    target = sched.get_velocity(z, noise, t)
                    eps_hat = sched.get_noise(z, t, pred)  # v̂ -> ε̂（用于反推 x0）
                else:
                    raise ValueError(f"Unsupported prediction_type: {ptype}")

                noise_loss = criterion(pred.float(),target.float())   # ← 噪声回归损失


                # 6) 反推 x0 并解码为像素图（供后续 img/density/num loss 使用）
                alpha_bar = sched.alphas_cumprod.to(device).gather(0, t).view(B, 1, 1, 1)
                x0 = (z_t - (1.0 - alpha_bar).sqrt() * eps_hat) / alpha_bar.sqrt()
                images = (vae.decode(x0).sample)/scale   # [-1,1]，可直接替代原来的 images
                
                # 计算密度图损失
                # 这里不能用加了token的
                images_count_input = utils.transform_img_tensor(images)
                
                B, C, H, W = images_count_input.shape
                assert (H, W) == (224, 224), f"Expected (224,224), got {(H,W)}"
                
                estimate_density = counting_model(images_count_input, batch['prompts_init']) # depth img
                estimate_num = torch.sum(estimate_density, dim=(-2, -1)) / 60.0
                
                density_loss = criterion(estimate_density.to(accelerator.device),batch['density_maps'].to(accelerator.device))
            

                loss = noise_loss + args.lambda_1 * density_loss 
                
                epoch_loss += loss.detach().item()
                n_batches += 1

                # log
                txt = f"On epoch {epoch} \n"
                with torch.no_grad():
                    txt += f"{batch['prompts']} \n"
                    txt += f"Global step: {[global_step]} \n"
                    txt += f"Estimate_num: {estimate_num.flatten().tolist()}\n"
                    txt += f"Loss: {loss.detach().item()} \n"
                    txt += f"Img loss: {noise_loss.detach().item()} \n"
                    txt += f"Density loss: {density_loss.detach().item()} \n"
                    txt += f"-------------------------------------- \n"
                    # txt += f"Num loss: {num_loss.detach().item()}"
                    with open("run_log.txt", "a") as f:
                        print(txt, file=f)
                    

                accelerator.backward(loss)
            
                
                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                def mask_grad(text_encoder, tokenizer,placeholder_token_id):
                    if accelerator.num_processes > 1:
                        grads = (
                            text_encoder.module.get_input_embeddings().weight.grad
                        )
                    else:
                        grads = text_encoder.get_input_embeddings().weight.grad

                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = (
                        torch.arange(len(tokenizer)) != placeholder_token_id
                    )
                    grads.data[index_grads_to_zero, :] = grads.data[
                        index_grads_to_zero, :
                    ].fill_(0)
                    
                mask_grad(text_encoder,   tokenizer,   placeholder_id_1)
                mask_grad(text_encoder_2, tokenizer_2, placeholder_id_2)
                
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(
                        text_encoder.get_input_embeddings().parameters(),
                        text_encoder_2.get_input_embeddings().parameters(),
                    ),
                    float(args.max_grad_norm),
                )
                
            
                # Checks if the accelerator has performed an optimization step behind the scenes\n",
                if  global_step % args.save_steps == 0 and global_step!= 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        token_embeds_1 = text_encoder.get_input_embeddings().weight.data
                        token_embeds_2 = text_encoder_2.get_input_embeddings().weight.data
                        
                        torch.save(token_embeds_1[placeholder_id_1].detach().cpu().clone(), 
                                   f"{token_path}/{global_step}_token_embeds_te1.pt")
                        torch.save(token_embeds_2[placeholder_id_2].detach().cpu().clone(), 
                                   f"{token_path}/{global_step}_token_embeds_te2.pt")
                        print("----------------save----------------------")
                

                global_step += 1
                pbar.update(1)
                pbar.set_postfix({"loss":loss.detach().item(),"global_step": global_step})

                optimizer.step()
                optimizer.zero_grad()
                
        epoch_loss_avg = epoch_loss / max(n_batches, 1)
        metric = epoch_loss_avg  # 没有验证集，就用训练集平均 loss

        improved = metric < (best_metric - min_delta)
        if improved:
            best_metric = metric
            bad_epochs = 0

            # 保存 best（
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                token_embeds_1 = text_encoder.get_input_embeddings().weight
                token_embeds_2 = text_encoder_2.get_input_embeddings().weight

                torch.save(token_embeds_1[placeholder_id_1].detach().cpu().clone(),
                        os.path.join(token_path, "best_token_embeds_te1.pt"))
                torch.save(token_embeds_2[placeholder_id_2].detach().cpu().clone(),
                        os.path.join(token_path, "best_token_embeds_te2.pt"))
        else:
            bad_epochs += 1
            if accelerator.is_main_process:
                print(f"[EarlyStopping] epoch={epoch}, metric={metric:.6f} "
                    f"no improvement vs best={best_metric:.6f} (min_delta={min_delta}). "
                    f"bad_epochs={bad_epochs}/{patience}")

            if bad_epochs >= patience:
                if accelerator.is_main_process:
                    print(f"[EarlyStopping] No improvement for {patience} epochs. Stopping.")
                break

    pbar.close()
            
        
        
   


if __name__ == "__main__":
  
    
    parser = argparse.ArgumentParser()

    # ---------- basic / device ----------
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["fp16", "fp32", "no"]) 
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="开启梯度检查点；默认已开启")

    # ---------- task ----------
    parser.add_argument("--count", type=int, default=8)


    # ---------- schedule ----------
    parser.add_argument("--num_train_epochs", type=int, default=50)

    # ---------- dataset / batching ----------
    parser.add_argument("--data_root_path", type=str, default="/root/autodl-tmp/datasets/FSC147-good")
    parser.add_argument("--keep_density_acc", type=int, default=0)
    
    

    # gradient_checkpointing：你原来默认是 True，那就这么写
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # ---------- checkpoints / io ----------
    parser.add_argument("--output_path", type=str, default="/root/autodl-tmp/results/density_control")
    parser.add_argument("--stable_diffusion_pipeline_path", type=str, default="/root/autodl-tmp/stable-diffusion/sdxl-turbo")

    # ---------- optim ----------
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--betas1", type=float, default=0.9)
    parser.add_argument("--betas2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)  # 保持原来是字符串
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--lambda_1", type=float, default=1.0)
    parser.add_argument("--lambda_2", type=float, default=0.0)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    
    # ---------- save ---------------
    parser.add_argument("--save_steps", type=int, default=1000)

    # ---------- diffusion ----------
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)

    # ---------- textual inversion ----------
    parser.add_argument("--placeholder_token", type=str, default="newcls")
    parser.add_argument("--initializer_token", type=str, default="some")
    
    # param_meter 
    parser.add_argument("--hyperparam", type=str, default="param-0")


    args = parser.parse_args()

    main(args)

    

   
