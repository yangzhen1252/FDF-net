import cv2
import torch
from modelnew1 import IRModel
from torchvision import transforms
import os
import numpy as np
import sklearn.preprocessing as sp
from time import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = IRModel()

weight = 'epoch_426_122.214.pt'
if os.path.exists(weight):
    net.load_state_dict(torch.load(weight))
img_path = 'data4/images/00009.png'
mask_path = 'data4/labels/00009.png'
mask_path1 = 'data4/labels/00009.png'

transforms_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4372, 0.4372, 0.4373],
                         std=[0.2479, 0.2475, 0.2485])
])
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
if __name__ == '__main__':
    img_tensor_list = []
    origin = cv2.imread(img_path)
    origin1 = cv2.imread(img_path)


    cv2.imshow('origin', origin)
    tr = transforms.Compose([transforms.ToTensor()])
    img = transforms_test(origin)
    img_tensor_list.append(img)
    img1 = transforms_test(origin1)
    img_tensor_list.append(img1)
    img_tensor_list = torch.stack(img_tensor_list, 0)
    T=cv2.imread(mask_path, 1)
    T = cv2.resize(T, (256,256))
    mask = tr(T)
    mask1 = tr(cv2.imread(mask_path1, 0))
    mask=mask.to(device)
    # mask1 = mask1.to(device)
    # pred=mask1
    T1 = cv2.imread(mask_path, 0)
    T1 = cv2.resize(T1, (256,256))
    T1 = normalization(T1)
    net.eval()

    with torch.no_grad():
        begin_time = time()
        pred = net(img_tensor_list[0:1].cuda())
        end_time = time()
        time = end_time - begin_time
        print('一共运行时间:', time)
    heatmap = pred .squeeze().cpu()

    single_map = heatmap
    hm = single_map.detach().numpy()

    hm = normalization(hm)

    #
    bin = sp.Binarizer(threshold=0.35)
    hm = bin.transform(hm)
    hmm=hm

    hm = np.uint8(255 * hm)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET )
    hm = cv2.resize(hm, (256,256))
    origin = cv2.resize(origin, (256,256))
    hm3 = cv2.applyColorMap(T, cv2.COLORMAP_JET)
    hm3 = cv2.resize(T, (256,256))
    #
    # a=abs(hm-T)
    # a= cv2.applyColorMap(a, cv2.COLORMAP_JET)
    # origin=T+hm

    cv2.imwrite("outputIR/%d.png" % 1, hm)
    cv2.imwrite("outputIR/%d.png" % 2, hm3)
    cv2.imwrite("outputIR/%d.png" % 3, origin)


    # pred[pred >= 0.8] = 1
    # pred[pred < 0.8] = 0
    pred1=hmm
    TP = ((pred1 == 1) & (T1 == 1)).sum()
    TN = ((pred1 == 0) & (T1== 0)).sum()
    FN = ((pred1 == 0) & (T1 == 1)).sum()
    FP = ((pred1 == 1) & (T1 == 0)).sum()
    P=TP/(TP+FP)
    pa = (TP + TN) / (TP + TN + FP + FN)
    iou = TP / (TP + FP + FN)
    R=TP/(TP+FN)
    F1=(2*P*R)/(P+R)
    print('pa: ', pa)
    print('R: ', R)
    print('P: ', P)
    print('F1: ', F1)

    print('iou', iou)

    # cv2.imshow('origin_out', np.hstack([img, pred]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
