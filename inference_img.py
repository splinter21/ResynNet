import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='ResynNet for 2 reference image')
parser.add_argument('--ref', dest='ref', nargs=2, required=True)
parser.add_argument('--origin', dest='origin', required=True)
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')

args = parser.parse_args()

from model.model import Model
model = Model()
model.load_model(args.modelDir, -1)
model.eval()
model.device()

ref = []
for i in range(2):
    ref.append(cv2.imread(args.ref[i], cv2.IMREAD_UNCHANGED))
    ref[i] = (torch.tensor(ref[i].transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
origin = cv2.imread(args.origin, cv2.IMREAD_UNCHANGED)
origin = (torch.tensor(origin.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

n, c, h, w = ref[0].shape
ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
for i in range(2):
    ref[i] = F.pad(ref[i], padding)
origin = F.pad(origin, padding)
img = model.inference(torch.cat(ref, 1), origin)
img = img.permute(0, 2, 3, 1).cpu().numpy()
cv2.imwrite('output.png', (img[0][:h, :w] * 255).astype('uint8'))
