import torch
import torchvision
from torchvision import transforms
from torch import nn as nn
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import DataLoader

resnet18 = torchvision.models.resnet18(pretrained=False)
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 20)
resnet18.load_state_dict(torch.load('./models/resnet18'))


# режем каптчу на буквы
def add_border(img, val, mode='color'):
    img = np.array(img)
    if mode == 'gray':
        col = np.array([[(200)] * val] * img.shape[0], np.uint8)
        row = np.array([[(200)] * (img.shape[1] + 2 * val)] * val, np.uint8)
    else:
        img = img[:, :, :3]
        col = np.array([[(200, 200, 200)] * val] * img.shape[0], np.uint8)
        row = np.array([[(200, 200, 200)] * (img.shape[1] + 2 * val)] * val, np.uint8)
    img = np.hstack((col, img, col))
    img = np.vstack((row, img, row))
    return img


def find_bb(contours, min_w, min_h, max_w):
    bboxes = []
    for c in contours:
        bb = cv2.boundingRect(c)
        # x, y, w, h = bb
        # if w != max_w and (w > min_w or h > min_h):
        bboxes.append(bb)
    bboxes = sorted(bboxes, key=lambda r: r[2] * r[3])[-5:-1]  # сортируем по площади и берем 4
    bboxes = sorted(bboxes, key=lambda x: x[0])  # сортируем по координате
    return bboxes


class BboxExceprion(Exception):
    pass


def open_filter_image(name):
    img = np.array(Image.open(name))
    imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    me = np.mean(imgray)
    st = np.std(imgray)
    th = me - 1.5 * st
    imgray = add_border(imgray, 10, 'gray')
    img = add_border(img, 10)
    return img, imgray, th


def crop_image(name, min_w, min_h, erode):
    img, imgray, th = open_filter_image(name)
    ret, thresh = cv2.threshold(imgray, th, 255, 0)
    img_erode = cv2.erode(thresh, np.ones(erode, np.uint8), iterations=1)
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = find_bb(contours, min_w, min_h, max_w=imgray.shape[1])
    if len(bboxes) == 4:
        images = []
        for i, bbox in enumerate(bboxes):
            x, y, w, h = bbox
            img_cur = Image.fromarray(img[y:y + h, x:x + w])
            img_cur = img_cur.resize((28, 28), resample=Image.NEAREST)
            images.append(img_cur)
        return images
    else:
        BboxExceprion.bboxes = bboxes
        raise BboxExceprion


split_label = lambda name: name.split('_')[1][:-4]


def crop(names, min_w=7, min_h=12, erode=(15, 3)):
    for name in names:
        try:
            labels = split_label(name)
            imgs = crop_image(name, min_w, min_h, erode)
            yield name, imgs, labels

        # сортируем по количеству bboxes в словарь
        except BboxExceprion as ex:
            print(f'Ошибка загрузки: {len(BboxExceprion.bboxes)} bboxes')


def predict_one_sample(model, inputs):
    """Предсказание, для одной картинки"""
    with torch.no_grad():
        model.eval()
        logit = model(inputs)
        probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
    return probs

labeles = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c',
       'd', 'e', 'f', 'g', 'h', 'j', 'k'])

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet normalize
                         std=[0.229, 0.224, 0.225])
    ])

def predict(saved_filename):
    load_gen = crop([saved_filename])
    name, imgs, label_true = next(load_gen)
    imgs_transformed = []
    for img in imgs:
        imgs_transformed.append(transform(img))

    val_loader = DataLoader(imgs_transformed, batch_size=4, shuffle=False)
    for inputs in val_loader:
        probs_im = predict_one_sample(resnet18, inputs)

    y_pred = np.argmax(probs_im,-1)
    y_pred = labeles[y_pred]
    probs = [pr[np.argsort(pr)[-1]] for pr in probs_im]
    return y_pred, probs