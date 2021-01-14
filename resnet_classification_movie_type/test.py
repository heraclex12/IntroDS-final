import os

from model.resnet import resnet101

from utils import getConfig

# pytorch
import torch
import torchvision.transforms as transforms
from PIL import Image


def test():
    ##
    config = getConfig()
    device = torch.device("cuda:" + str(config.device))
    # define transform image
    transform_test = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        # transforms.CenterCrop(config.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    net = resnet101(pretrained=True, use_bap=False)
    in_features = net.fc_new.in_features
    new_linear = torch.nn.Linear(
        in_features=in_features, out_features=25)
    net.fc_new = new_linear

    # load checkpoint
    use_gpu = torch.cuda.is_available() and config.use_gpu
    if use_gpu:
        net = net.to(device)
    gpu_ids = [int(r) for r in config.gpu_ids.split(',')]
    if use_gpu and config.multi_gpu:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)
    checkpoint_path = os.path.join(config.checkpoint_path, 'model_best.pth.tar')
    load_state_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
    new_state_dict = {}
    for key, value in load_state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    net.load_state_dict(new_state_dict)
    img_dir = config.image
    image = Image.open(img_dir).convert('RGB')
    image = transform_test(image)
    preds, _, _ = net(image.unsqueeze(0).to(device))
    print(torch.sigmoid(preds))

if __name__ == '__main__':
    test()
