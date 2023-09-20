import os
import torch
import datetime as dt
import matplotlib.pyplot as plt

class EarlyStopping():
    def __init__(self, config):
        self.config = config
        # 最良のモデルを保存する先
        self.path = os.path.join(os.getcwd(), "models")
        # lossの記録を保持する変数
        self.best_loss = float('inf')
        # 最小lossを更新できなかった連続回数
        self.patience = 0
    
    def check(self, loss, model):
        # 最小のlossを更新できた場合
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience = 0
          
            return False

        # lossを上回った場合
        else:
            self.patience += 1
            # patienceを上回った場合
            if self.patience >= self.config['model']['patience']:
                print("========== Early Stopping ===========")
                print(f"Best valid loss: {self.best_loss:.4f}")
                if not os.path.exists(self.path):
                    os.makedirs("models")
                
                file_nm = os.path.join(self.path, f"{dt.datetime.now().date()}_Autoencoder")
                torch.save(model.state_dict(), file_nm)

                return True
            
            return False

# lossを保存
def save_loss(train_loss, valid_loss):
    path = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(path):
        os.makedirs("log/losses")
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.title("losses")
    plt.ylabel('loss')
    plt.xlabel("epoch")
    plt.legend()

    file_nm = os.path.join(path, 'loss.png')
    plt.savefig(file_nm)
    plt.close()
    
    
# 出力結果を保存
def save_img(config, model, dataloader):
    path = os.path.join(os.getcwd(), 'log/images')
    if not os.path.exists(path):
        os.makedirs("log/images")

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= config['compare_num']:
                break
            x = x.to(config['device'])
            pred, z = model(x)
            fig, axes = plt.subplots(1, 3)
            axes[0].imshow(x[0].permute(1, 2, 0).cpu().numpy())
            axes[1].imshow(pred[0].permute(1, 2, 0).cpu().numpy())
            axes[2].imshow(z[0].permute(1, 2, 0).cpu().numpy())
            plt.savefig(os.path.join(path, f"image_{i+1}.png"))