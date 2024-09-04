import torch
from torch import nn
import tlib
import torchvision
import torchvision.transforms as transforms
from imagenet_loader import imagenet_dataset

# config
# arch = 'resnet18'
arch = "swin_v2_t"
dataset = 'imagenet100'
logger = tlib.Logger(output='stdout')
pretrained = False
drop_out_p = 0.35
weight_decay_rt = 0.0001
# Select Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Prepare Dataset
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 期望的输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

if dataset == 'imagenet100':
    train_set = imagenet_dataset(
        root_dir='./data/imagenet100/train',
        label_dir='./data/imagenet100/Labels_100.json',
        transform=transform
    )
    test_set = imagenet_dataset(
        root_dir='./data/imagenet100/val',
        label_dir='./data/imagenet100/Labels_100.json',
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# Prepare Model
def prepare_model(model_name, num_class):
    base_model = getattr(torchvision.models, model_name)(pretrained=pretrained)
    if 'resnet' in model_name:
        feature_dim = getattr(base_model, 'fc').in_features
        setattr(base_model, "fc", nn.Linear(feature_dim, num_class))
    elif 'swin' in model_name:
        base_model.head = nn.Linear(base_model.head.in_features, num_class)
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Dropout):
                # print(f"Before: {module}")
                module.p = drop_out_p  # 修改 dropout 概率
    else:
        raise ValueError(f"Model {model_name} not supported")
    return base_model

model = prepare_model(arch, 100).to(device)
logger.trace(model)
# Train Model
loss = torch.nn.CrossEntropyLoss(reduction='none').to(device)
num_epochs = 1000
updater = torch.optim.SGD(model.parameters(), lr=0.004, weight_decay=weight_decay_rt)
checkpoint_manager = tlib.CheckpointManager(
    arch=arch,
    dataset=dataset,
    logger=logger
)
model = checkpoint_manager.load(model, f'{arch}_{dataset}.pth', f'{arch}_{dataset}_best.pth')
tlib.train_ch3_plus(
    model,
    train_loader,
    test_loader,
    loss,
    num_epochs,
    updater,
    device,
    use_animator=False,
    tensorboard_path='./tensorboard',
    logger=logger,
    checkpoint_manager=checkpoint_manager,
    test_freq=1,
    train_batch_log= True,
    test_batch_log= True,
    batch_tensorboard= True
)
