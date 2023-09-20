import torch

def run(config, model, dataloader, criterion, optim, state='train'):
    running_loss = 0.0
    if state == 'train':
        model.train()                                   # 訓練モード
        for x, _ in dataloader:
            x = x.to(config['device'])                     # データをGPUへ
            optim.zero_grad()                           # パラメータ初期化
            pred, _ = model(x)                          # モデル出力
            loss = criterion(pred, x)                   # 評価関数に入れてlossを計算
            loss.backward()                             # 逆伝播
            optim.step()                                # パラメータ更新
            running_loss += loss.item() * x.size(0)
    
    elif state == 'eval':
        model.eval()                                    # 評価モードへ
        with torch.no_grad():                           # パラメータ更新しない
            for x, _ in dataloader:
                x = x.to(config['device'])
                pred, _ = model(x)
                loss = criterion(pred, x)
                running_loss += loss.item() * x.size(0)
    
    else:
        raise Exception("Choose from ['train', 'eval]")
        
    return running_loss / len(dataloader)
