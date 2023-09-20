import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# PytorchのDataLoaderを返す関数
def make_dataloader(config):
    config_data = config['data']

    # 画像のサイズを変換してテンソルに変換
    transform = transforms.Compose([
        transforms.Resize((config_data['size'], config_data['size'])),
        transforms.ToTensor()
    ])

    # 画像データを読み込み
    dataset = datasets.ImageFolder(root=config_data['path'], transform=transform)

    # train, valid, testに分ける
    train_size = int(len(dataset) * config_data['train_size'])
    valid_size = int(len(dataset) * config_data['valid_size'])
    test_size  = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = \
        random_split(dataset, [train_size, valid_size, test_size])

    # PyTorchでデータを読み込むために定義
    train_loader = DataLoader(train_dataset, batch_size=config_data['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config_data['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=config_data['batch_size'], shuffle=True)

    return train_loader, valid_loader, test_loader