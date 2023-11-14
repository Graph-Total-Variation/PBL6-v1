import sys
import pickle
import torch
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import argparse
from train_gtv import *
import logging


cuda = True if torch.cuda.is_available() else False
if cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

resroot = "result"

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-w",
        "--width",
        help="Resize image to a square image with given width",
        type=int,
    )
    parser.add_argument("-m", "--model")
    parser.add_argument("--stride", default=18, type=int)
    parser.add_argument(
        "--multi", default=30, type=int, help="# of patches evaluation in parallel"
    )
    parser.add_argument("--opt", default="opt")
    parser.add_argument("--image_path_train")
    parser.add_argument("--image_path_test")
    parser.add_argument("--image_path")
    parser.add_argument("--layers", default=1, type=int)
    parser.add_argument("--row", default=100, type=int)
    args = parser.parse_args()
    return args
    
def denoise(
    inp,
    gtv,
    argref,
    normalize=False,
    stride=36,
    width=324,
    prefix="_",
    verbose=0,
    opt=None,
    approx=False,
    args=None,
    logger=None,
):
    try:
        from skimage.metrics import structural_similarity as compare_ssim
    except Exception:
        from skimage.measure import compare_ssim

    sample = cv2.imread(inp)
    if width is None:
        width = sample.shape[0]
    else:
        sample = cv2.resize(sample, (width, width))
    sample = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    sample = np.expand_dims(sample, axis=2)
    sample = sample.transpose((2, 0, 1))
    shape = sample.shape

    # if normalize:
    #     sample = _norm(sample, newmin=0, newmax=1)
    sample = torch.from_numpy(sample)

    cuda = True if torch.cuda.is_available() else False

    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    if argref:
        ref = cv2.imread(argref)
        if ref.shape[0] != width or ref.shape[1] != width:
            ref = cv2.resize(ref, (width, width))
        ref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
        ref_p = resroot + "/ref_" + argref.split("/")[-1]
        plt.imsave(ref_p, ref,cmap='gray')
        ref = np.expand_dims(ref, axis=2)
        logger.info(ref_p)
        tref = ref.copy()
        ref = ref.transpose((2, 0, 1))
        ref = torch.from_numpy(ref)
        # if normalize:
        #     ref = _norm(ref, newmin=0, newmax=1)

    tstart = time.time()
    T1 = sample
    if argref:
        T1r = ref

    T1 = torch.nn.functional.pad(T1, (0, stride, 0, stride), mode="constant", value=0)
    shapex = T1.shape
    T2 = (
        torch.from_numpy(T1.detach().numpy().transpose(1, 2, 0))
        .unfold(0, opt.width, stride)
        .unfold(1, opt.width, stride)
    ).type(dtype)
    T2 = T2.contiguous()
    if argref:
        T1r = torch.nn.functional.pad(
            T1r, (0, stride, 0, stride), mode="constant", value=0
        )
    MAX_PATCH = args.multi
    oT2s0 = T2.shape[0]
    T2 = T2.view(-1, opt.channels, opt.width, opt.width)
    dummy = torch.zeros(T2.shape).type(dtype)
    logger.info("{0}".format(T2.shape))
    with torch.no_grad():
        for ii, i in enumerate(range(0, T2.shape[0], MAX_PATCH)):
            P = gtv.predict(
                T2[i : (i + MAX_PATCH), :, :, :].float().contiguous().type(dtype),
            )
            dummy[i : (i + MAX_PATCH)] = P
    dummy = dummy.view(oT2s0, -1, opt.channels, opt.width, opt.width)
    dummy = dummy.cpu()
    if verbose:
        logger.info("Prediction time: {0}".format(time.time() - tstart))
    else:
        logger.info("Prediction time: {0}".format(time.time() - tstart))

    dummy = (
        patch_merge(dummy, stride=stride, shape=shapex, shapeorg=shape).detach().numpy()
    )

    ds = np.array(dummy).copy()
    d = np.minimum(np.maximum(ds, 0), 255)
    logger.info("RANGE: {0} - {1}".format(d.min(), d.max()))
    d = d.transpose(1, 2, 0) / 255
    d = d[:,:,0]
    if 0:
        opath = args.output
    else:
        filename = inp.split("/")[-1]
        opath = resroot + "/{0}_{1}".format(prefix, filename)
        opath = opath[:-3] + "png"
    d = np.minimum(np.maximum(d, 0), 1)
    plt.imsave(opath, d, cmap='gray')
    if argref:
        mse = ((d - (tref / 255.0)) ** 2).mean() * 255
        logger.info("MSE: {:.5f}".format(mse))
        logger.info("Saved {0}".format(opath))
        # return (0, score, 0, psnr2, mse, d)  # psnr, ssim, denoised image
    return d


def patch_merge(P, stride=36, shape=None, shapeorg=None):
    S1, S2 = P.shape[0], P.shape[1]
    m = P.shape[-1]

    R = torch.zeros(shape)
    Rc = torch.zeros(shape)

    ri, rj = 0, 0
    c = 1

    for i in range(S1):
        for j in range(S2):

            R[:, ri : (ri + m), rj : (rj + m)] += P[i, j, :, :, :].cpu()
            Rc[:, ri : (ri + m), rj : (rj + m)] += 1
            rj += stride
            c += 1
        ri += stride
        rj = 0

    return (R / Rc)[:, : shapeorg[-1], : shapeorg[-1]]

def main_eva(
    seed,
    model_name,
    trainset,
    testset,
    imgw=None,
    verbose=0,
    image_path_train=None,
    image_path_test=None,
    image_path = None,
    noise_type="gauss",
    opt=None,
    args=None,
    logger=None,
):
    cuda = True if torch.cuda.is_available() else False
    torch.autograd.set_detect_anomaly(True)
    opt.logger.info("CUDA: {0}".format(cuda))
    if cuda:
        dtype = torch.cuda.FloatTensor
        opt.logger.info(torch.cuda.get_device_name(0))
    else:
        dtype = torch.FloatTensor
    gtv = GTV(width=6, cuda=cuda, opt=opt)  # just initialize to load the trained model, no need to change
    PATH = model_name
    device = torch.device("cuda") if cuda else torch.device("cpu")
    gtv.load_state_dict(torch.load(PATH, map_location=device))
    gtv.cuda()
    width = gtv.opt.width
    opt.width = width
    opt = gtv.opt
    if image_path_train:
        trainset = [i.split(".")[0] for i in os.listdir(os.path.join(args.image_path_train,"ref"))]
    else: 
        trainset = None
    if image_path_test:
        testset = [i.split(".")[0] for i in os.listdir(os.path.join(args.image_path_test,"ref"))]
    else:
        testset = None
    _, _ = main_eva(
        seed="gauss",
        model_name=args.model,
        trainset=trainset,
        testset=testset,
        imgw=args.width,
        verbose=1,
        image_path_train=args.image_path_train,
        image_path_test=args.image_path_test,
        image_path = args.image_path,
        noise_type="gauss",
        opt=opt,
        args=args,
        logger=logger,
    )

# python test_gtv.py -w 512 -m model/GTV_19.pkl --stride 9 --multi 200 --image_path_test dataset/Test --image_path_train dataset/dataset_structure
