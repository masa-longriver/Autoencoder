import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    # モデル構造を定義
    def __init__(self, config):
        super(Autoencoder, self).__init__()
        config_model = config['model']
        self.input_channel   = config_model['input_channel']
        self.hidden_channel1 = config_model['hidden_channel1']
        self.hidden_channel2 = config_model['hidden_channel2']
        self.stride1         = config_model['stride1']
        self.stride2         = config_model['stride2']
        self.padding1        = config_model['padding1']
        self.padding2        = config_model['padding2']
        self.kernel_size1    = config_model['kernel_size1']
        self.kernel_size2    = config_model['kernel_size2']
        
        self.Encoder = nn.Sequential(
            # 畳み込み1: チャネル数 3 -> 16
            nn.Conv2d(
                self.input_channel, self.hidden_channel1, 
                self.kernel_size1, stride = self.stride1, padding = self.padding1,
                ),
            # 活性化関数1
            nn.LeakyReLU(),
            # 畳み込み2: チャネル数 16 -> 16
            nn.Conv2d(
                self.hidden_channel1, self.hidden_channel2, 
                self.kernel_size1, stride = self.stride1, padding = self.padding1,
                ),
            # 活性化関数2
            nn.LeakyReLU(),
            # 畳み込み3: チャネル数 16 -> 3
            nn.Conv2d(
                self.hidden_channel2, config_model['latent_channel'], 
                self.kernel_size2, stride = self.stride2, padding = self.padding2)
        )
        self.Decoder = nn.Sequential(
            # Encoderと対称的な処理
            nn.ConvTranspose2d(
                config_model['latent_channel'], self.hidden_channel2, 
                self.kernel_size2, stride = self.stride2, padding = self.padding2,
                output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                self.hidden_channel2, self.hidden_channel1, 
                self.kernel_size1, self.stride2, self.padding2,
                output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                self.hidden_channel1, self.input_channel, 
                self.kernel_size1, self.stride2, self.padding2),
            # 値を0から1の間に収める
            nn.Sigmoid()
        )
    
    # 実際に動く部分
    def forward(self, x):
        z = self.Encoder(x)
        x = self.Decoder(z)

        return x, z