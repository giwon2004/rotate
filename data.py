import os
from argparse import Namespace
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
from torchvision import transforms
import cv2

from bts.pytorch.bts import BtsModel

obj_num = 100
min_angle = 5
angle_num = 72
original_path = 'coil-100/coil-100'
depth_path = 'coil-100/depth'

class Coil100(Dataset):
    def __init__(self, max_diff, path):
        super(Coil100, self).__init__()
        self.max_diff = max_diff
        self.path = path

    def __len__(self):
        return obj_num*angle_num*(self.max_diff//min_angle)

    def __getitem__(self, idx):
        obj = idx//(angle_num*(self.max_diff//min_angle))+1
        angle = idx%(angle_num*(self.max_diff//min_angle))
        start_angle = (angle//(self.max_diff//min_angle))*min_angle
        angle_diff = (angle%(self.max_diff//min_angle)+1)*min_angle
        end_angle = (start_angle+angle_diff)%360
        start_name = os.path.join(self.path, filename(obj, start_angle))
        end_name = os.path.join(self.path, filename(obj, end_angle))
        start_image = Image.open(start_name)
        end_image = Image.open(end_name)
        sample = {'start':start_image, 'end':end_image, 'angle':angle_diff}
        return sample

def filename(obj, angle):
    return f'obj{obj}__{angle}.png'

def run_bts():
    if not os.path.isdir(depth_path):
        os.mkdir(depth_path)
    args = Namespace(bts_size=512, do_kb_crop=False, encoder='densenet161_bts', input_height=224,
                     input_width=224,
                     max_depth=10.0, mode='test', model_name='bts_nyu_v2_pytorch_densenet161', save_lpg=False,
                     checkpoint_path='./bts/pytorch/models/bts_nyu_v2_pytorch_densenet161/model',
                     dataset='coil-100')
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for obj in range(1, obj_num+1):
        print(f'Object {obj}')
        for angle in range(0, 360, min_angle):
            print(angle, end=' ')
            file = os.path.join(original_path, filename(obj, angle))
            image = Image.open(file)
            image = preprocess(image)
            image = image.unsqueeze(0)
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, 518.8579)
            depth_est = interpolate(depth_est, (128,128))
            depth_est = depth_est[0][0].unsqueeze(2)*100
            depth_est = depth_est.detach().numpy()
            cv2.imwrite(os.path.join(depth_path, filename(obj, angle)), depth_est)
        print('')

# temporary function for changing image size to (128,128)
def fix_image_size():
    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    for obj in range(1, obj_num+1):
        print(f'Object {obj}')
        for angle in range(0, 360, min_angle):
            file = os.path.join(depth_path, filename(obj, angle))
            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            image = preprocess(image)
            image = image.unsqueeze(0)*255
            image = interpolate(image, (128,128))
            image = image[0][0].unsqueeze(2)
            image = image.detach().numpy()
            cv2.imwrite(os.path.join(depth_path, filename(obj, angle)), image)


fix_image_size()
