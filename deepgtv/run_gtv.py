import sys
import argparse
import cv2
import torch
from train_gtv import *
from torchvision import transforms

cuda = True if torch.cuda.is_available() else False
if cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

def run_gtv(input_path, model_path, cuda=True):
    # Load the GTV model
    gtv = GTV(width=512, cuda=cuda)
    gtv.load_state_dict(torch.load(model_path, map_location='cuda' if cuda else 'cpu'))
    gtv.eval()

    # Read the input image
    input_image = cv2.imread(input_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    output_path = "result/kq.png"
    # Denoise the image
    with torch.no_grad():
        if cuda:
            input_image = input_image.cuda()
        denoised_image = gtv(input_image)

    # Save the denoised image
    denoised_image = denoised_image.cpu().squeeze().numpy()
    denoised_image = (denoised_image * 255).astype('uint8')
    cv2.imwrite(output_path, denoised_image)

def main():
    parser = argparse.ArgumentParser(description="Run GTV for image denoising")
    parser.add_argument(
        "-w",
        "--width",
        help="Resize image to a square image with given width",
        type=int,
    )
    parser.add_argument("--stride", default=18, type=int)
    parser.add_argument("input", help="Path to the input image")
    # parser.add_argument("output", help="Path to save the denoised output image")
    parser.add_argument("-m","--model", help="Path to the trained GTV model")
    # parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")

    args = parser.parse_args()

    # Initialize the options and logger (optional)
    opt = OPT(
        batch_size=32,
        channels=1,
        lr=1e-4,
        momentum=0.9,
        u_max=1000,
        u_min=0.0001,
        cuda=True if torch.cuda.is_available() else False
    )
    supporting_matrix(opt)

    # Denoise the image and save the result
    run_gtv(args.input, args.output, args.model, cuda=args.cuda)
    print(f"Denoised image saved to {args.output}")

if __name__ == "__main__":
    main()


#python run_gtv.py input_image.jpg output_denoised_image.jpg model/GTV_model.pth --cuda
