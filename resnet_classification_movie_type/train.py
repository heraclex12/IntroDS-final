import os

from model.resnet import resnet101
from dataset.custom_dataset import CustomDataset

from utils import getConfig
from utils import get_lr, save_checkpoint, set_seed
from utils import Engine

# pytorch
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
GLOBAL_SEED = 1231

def _init_fn(worker_id):
    set_seed(GLOBAL_SEED+worker_id)

def train():
    # input params
    config = getConfig()
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(config.device)
    best_prec1 = 0.
    rate = 0.875
    device = torch.device("cuda:" + str(config.device))
    # define train_dataset and loader
    transform_train = transforms.Compose([
        transforms.Resize((int(config.input_size//rate), int(config.input_size//rate))),
        transforms.RandomCrop((config.input_size,config.input_size)),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=32./255.,saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dataset = CustomDataset('data/movie_train.csv', 'data/movie/images', transform=transform_train)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=True,
        worker_init_fn=_init_fn)

    transform_test = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.CenterCrop(config.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_dataset = CustomDataset('data/movie_val.csv', 'data/movie/images', transform=transform_test)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size*2, shuffle=False, num_workers=config.workers, pin_memory=True,
        worker_init_fn=_init_fn)
    # logging dataset info
    print('Train:[{train_num}], Val:[{val_num}]'.format(
        train_num=len(train_dataset),
        val_num=len(val_dataset)))
    print('Batch Size:[{0}], Total:::Train Batches:[{1}],Val Batches:[{2}]'.format(
        config.batch_size, len(train_loader), len(val_loader)
    ))

    net = resnet101(pretrained=True, use_bap=False)
    in_features = net.fc_new.in_features
    new_linear = torch.nn.Linear(
        in_features=in_features, out_features=train_dataset.num_classes)
    net.fc_new = new_linear

    # gpu config
    use_gpu = torch.cuda.is_available() and config.use_gpu
    if use_gpu:
        net = net.to(device)

    gpu_ids = [int(r) for r in config.gpu_ids.split(',')]
    if use_gpu and config.multi_gpu:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)

    # define optimizer
    assert config.optim in ['sgd', 'adam'], 'optim name not found!'
    if config.optim == 'sgd':
        optimizer = torch.optim.SGD(
            net.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optim == 'adam':
        optimizer = torch.optim.Adam(
            net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # define learning scheduler
    assert config.scheduler in ['plateau',
                                'step', 'cosine_annealing'], 'scheduler not supported!!!'
    if config.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.1)
    elif config.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.9)

    # define loss
    criterion = torch.nn.BCEWithLogitsLoss()

    # train val parameters dict
    state = {'model': net, 'train_loader': train_loader,
             'val_loader': val_loader, 'criterion': criterion,
             'config': config, 'optimizer': optimizer, 'device': device, 'step': 0, 'lr': config.lr}
    ## train and val
    engine = Engine()
    print(config)
    best_prec = 0
    for e in range(config.epochs):
        if config.scheduler == 'step':
            scheduler.step()
        lr_val = get_lr(optimizer)
        print("Start epoch %d ==========,lr=%f" % (e, lr_val))
        train_prec, train_loss = engine.train(state, e)
        prec, val_loss = engine.validate(state)
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        print('Epoch: {}, Train-Loss: {:.4f}, Train-accuracy: {:.4f},'\
                     'Test-accuracy: {:.4f}'.format(e + 1, train_loss, train_prec, prec))
        print('Best accuracy: {:.4f}'.format(best_prec))
        save_checkpoint({
            'epoch': e + 1,
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, config.output)
        if config.scheduler == 'plateau':
            scheduler.step(val_loss)

def validate():
    ##
    engine = Engine()
    config = getConfig()
    device = torch.device("cuda:" + str(config.device))
    # define dataset
    transform_test = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
       # transforms.CenterCrop(config.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_dataset = CustomDataset('data/movie_val.csv', 'data/movie/images', transform=transform_test)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers, pin_memory=True)

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

    # define loss
    criterion = torch.nn.BCEWithLogitsLoss()
    if use_gpu:
        criterion = criterion.cuda()
    state = {'model': net, 'val_loader': val_loader, 'criterion': criterion, 'config': config,
             'device': device, 'step': 0, 'lr': config.lr}
    prec1, fprec, val_loss = engine.validate(state)
    print(prec1)


if __name__ == '__main__':
    config = getConfig()
    engine = Engine()

    train()
