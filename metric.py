import os
import cv2
import numpy as np
from glob import glob
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import argparse
import sys


class PSNR():
    def __init__(self):
        self.loss_item = torch.nn.MSELoss()

    def forward(self, x, y, data_range=1.):
        with torch.no_grad():
            mse = self.loss_item(x.clamp(0, 1.), y.clamp(0., 1.))
        return 10*torch.log10(data_range**2/(mse+1e-14))


def main(res_root,gt_root):
    # log_file = open("results_log.txt", "w")
    # import sys
    # sys.stdout = log_file

    # 모든 _res.jpg 파일 가져오기 (예측한 frames)
    all_files = sorted(glob(os.path.join(res_root, "*.jpg")))

    res_files = [f for f in all_files if "_res" in os.path.basename(f)]
    # org_files = [f for f in all_files if "_res" not in os.path.basename(f)]

    psnr_results = 0.0
    with open("psnr_results.txt", "w") as log_file:
        for res_file in res_files:
            base = os.path.basename(res_file)
            parts = base.split("_")

            # 폴더 이름 (예: aquarium_08)
            folder_name = "_".join(parts[:-3])
            # 
            # 프레임 번호
            frame_num = int(parts[-3])
            plus_num = int(parts[-2]) +1
            next_frame_num = frame_num + plus_num
            
            # GT 비교 대상 경로 (images 폴더 내 PNG)
            gt_folder = os.path.join(gt_root, folder_name, "images")
            gt_path = os.path.join(gt_folder, f"{next_frame_num:06d}.png")

            if not os.path.exists(gt_path):
                print(f"skip: {gt_path} not found")
                continue
            
            totensor = ToTensor()

            # (res, gt) 이미지 로드
            res_img = totensor(Image.open(res_file))
            gt_img =  totensor(Image.open(gt_path))
            print(res_file, "&", gt_path)

            h, w = gt_img.shape[1:]
            hn, wn = (h//32-1)*32, (w//32-1)*32
            hleft = (h-hn)//2
            wleft = (w-wn)//2
            gt_img =  gt_img[:, hleft:hleft+hn, wleft:wleft+wn]

            if res_img is None or gt_img is None:
                print(f"skip: failed to read {res_file} or {gt_path}")
                continue

            res_img = res_img.float() 
            gt_img =gt_img.float() 

            # metric 계산
            metric_psnr = PSNR()
            psnr_val = metric_psnr.forward(res_img, gt_img)
            print("psnr", psnr_val)

            psnr_results += psnr_val
            res_name = res_file.split("/")[-1]
            gt_name = gt_path.split("/")[-1]
            log_file.write(f"res: {res_name}, gt: {gt_name}, psnr: {psnr_val}\n")


    all_avg = psnr_results / len(res_files)
    print("psnr :", all_avg)
    


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--res_root', type=str, default="/home/sieun/KETI/validation/Eventcamera_VFI/x5_interpolation/gopro_bsegrb/Expv8_large_x5adamwithLPIPS/Validation_Visual_Examples/images/48")
    parser.add_argument('--gt_root', type=str, default="/home/sieun/KETI/TimeLens-custom/dataset/bs_ergb/1_TEST")
    
    args = parser.parse_args()

    main(args.res_root, args.gt_root)
    # log_file.close()

