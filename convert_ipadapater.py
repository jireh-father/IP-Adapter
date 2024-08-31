import torch
import argparse
import os

def main(args):
    sd = torch.load(args.ckpt_path, map_location="cpu")
    image_proj_sd = {}
    ip_sd = {}
    for k in sd:
        if k.startswith("unet"):
            pass
        elif k.startswith("image_proj_model"):
            image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
        elif k.startswith("adapter_modules"):
            ip_sd[k.replace("adapter_modules.", "")] = sd[k]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="checkpoint-50000/pytorch_model.bin")
    #outpupt path
    parser.add_argument("--output_path", type=str, default="./ip_adapter.bin")
    main(parser.parse_args())