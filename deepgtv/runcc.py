import sys
import pickle
import torch
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import argparse
from deepgtv.train_gtv import *
import logging


cuda = True if torch.cuda.is_available() else False
if cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

resroot = "uploads"

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
        "--multi", default=200, type=int, help="# of patches evaluation in parallel"
    )
    parser.add_argument("--opt", default="opt")
    parser.add_argument("--image_path_train")
    parser.add_argument("--image_path_test")
    parser.add_argument("--image_path")
    parser.add_argument("--layers", default=1, type=int)
    parser.add_argument("--row", default=100, type=int)
    args = parser.parse_args()
    return args

def load_gtv_model(model_path):
    # Tạo đối tượng OPT để cấu hình
    opt = OPT(
        batch_size=32,
        channels=1,
        lr=1e-4,
        momentum=0.9,
        u_max=1000,
        u_min=0.0001,
        cuda=True if torch.cuda.is_available() else False
    )

    # Thiết lập logger
    import logging
    logger = logging.getLogger("root")
    logger.addHandler(logging.StreamHandler(sys.stdout))
    opt.logger = logger
    opt.legacy = True

    # Chuẩn bị ma trận hỗ trợ
    supporting_matrix(opt)

    # Tạo đối tượng GTV
    gtv = GTV(width=16, cuda=opt.cuda, opt=opt)
    device = torch.device("cuda") if cuda else torch.device("cpu")
    # Load trọng số đã được đào tạo
    gtv.load_state_dict(torch.load(model_path, map_location=device))
    # gtv.cuda()
    width = gtv.opt.width
    opt.width = width
    opt = gtv.opt

    return gtv
 
opt = OPT(
    batch_size=32,
    channels=1,
    lr=1e-4,
    momentum=0.9,
    u_max=1000,
    u_min=0.0001,
    cuda=True if torch.cuda.is_available() else False
    )
import logging
logger = logging.getLogger("root")
logger.addHandler(logging.StreamHandler(sys.stdout))
opt.logger = logger
opt.legacy = True
# Chuẩn bị ma trận hỗ trợ
supporting_matrix(opt)
logger.info("GTV evaluation")


def denoise_image(
    inp,
    gtv,
    stride=8, #ảnh hưởng đến tốc độ denoise
    width=None,
    prefix="img2",
    verbose=0,
    opt=opt,
    approx=False,
    args=None,
    logger=logger
):

    sample = cv2.imread(inp)
    if width is None:
        width = sample.shape[0]
    else:
        sample = cv2.resize(sample, (width, width))
        
    sample = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    sample = np.expand_dims(sample, axis=2)
    sample = sample.transpose((2, 0, 1))
    shape = sample.shape

    sample = torch.from_numpy(sample)

    cuda = True if torch.cuda.is_available() else False

    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    tstart = time.time()
    T1 = sample

    T1 = torch.nn.functional.pad(T1, (0, stride, 0, stride), mode="constant", value=0)
    shapex = T1.shape
    T2 = (
        torch.from_numpy(T1.detach().numpy().transpose(1, 2, 0))
        .unfold(0, opt.width, stride)
        .unfold(1, opt.width, stride)
    ).type(dtype)
    T2 = T2.contiguous()

    # MAX_PATCH = args.multi
    MAX_PATCH = 500
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
    # if 0:
    #     opath = args.output
    # else:
    #     filename = inp.split("/")[-1]
    #     opath = resroot + "/{0}_{1}".format(prefix, filename)
    #     opath = opath[:-3] + "png"
    d = np.minimum(np.maximum(d, 0), 1)
    # plt.imsave(opath, d, cmap='gray')

    return d



def denoise_image2(
    sample,
    gtv,
    stride=8, #ảnh hưởng đến tốc độ denoise
    width=None,
    prefix="img2",
    verbose=0,
    opt=opt,
    approx=False,
    args=None,
    logger=logger
):
    
    # sample = cv2.imread(inp)
    if width is None:
        width = sample.shape[0]
    else:
        sample = cv2.resize(sample, (width, width))

    sample = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    sample = np.expand_dims(sample, axis=2)
    sample = sample.transpose((2, 0, 1))
    shape = sample.shape

    sample = torch.from_numpy(sample)

    cuda = True if torch.cuda.is_available() else False

    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
   

    tstart = time.time()
    T1 = sample

    T1 = torch.nn.functional.pad(T1, (0, stride, 0, stride), mode="constant", value=0)
    shapex = T1.shape
    T2 = (
        torch.from_numpy(T1.detach().numpy().transpose(1, 2, 0))
        .unfold(0, opt.width, stride)
        .unfold(1, opt.width, stride)
    ).type(dtype)
    T2 = T2.contiguous()
    
    # MAX_PATCH = args.multi
    MAX_PATCH = 500
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
    # if 0:
    #     opath = args.output
    # else:
    #     filename = inp.split("/")[-1]
    #     opath = resroot + "/{0}_{1}".format(prefix, filename)
    #     opath = opath[:-3] + "png"
    # opath = "PBL6-v1/deepgtv/result/img2.png"
    d = np.minimum(np.maximum(d, 0), 1)
    # plt.imsave(opath, d, cmap='gray')
    return d


def patch_merge(P, stride=8, shape=None, shapeorg=None):
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

