import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, residual_in_fp32=False,
                 lstm_hidden_size=128):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.relu = nn.ReLU()

        # 定义3个1D卷积层，分别使用不同的kernel_size
        self.conv_layer_1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                      padding=0)
        self.conv_layer_3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                      padding=0)
        self.conv_layer_5 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                      padding=0)

        self.conv_layer_lstm = nn.Sequential(nn.Conv1d(512, 128, kernel_size=1, bias=False),
                                             nn.Conv1d(128, 512, kernel_size=1, bias=False))

        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=2, batch_first=True)

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=256, batch_first=True, bidirectional=True)

        self.gru = nn.GRU(input_size=out_channels, hidden_size=256, batch_first=True, bidirectional=True)
        # 定义层归一化
        self.norm = nn.LayerNorm(out_channels)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=4,  # 使用1个头简化实现
            batch_first=True
        )

        # 线性层：将双向LSTM的输出映射到embed_dim
        # self.fc1 = nn.Linear(in_channels, in_channels)
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.BatchNorm1d(in_channels * 4),  # 批归一化
            nn.GELU(),  # 更平滑的激活函数
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(in_channels * 4, in_channels)
        )
        # self.fc2 = nn.Linear(2*in_channels, in_channels)
        self.mlp2 = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels * 4),
            nn.BatchNorm1d(in_channels * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_channels * 4, in_channels)
        )
        # self.fc3 = nn.Linear(in_channels, in_channels)
        self.mlp3 = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.BatchNorm1d(in_channels * 4),  # 批归一化
            nn.GELU(),  # 更平滑的激活函数
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(in_channels * 4, in_channels)
        )
        # 替换 conv_layer_lstm 为 Transformer 层
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=512,  # 输入/输出维度（需匹配双向GRU的输出维度）
            nhead=4,  # 注意力头数（根据需求调整）
            dim_feedforward=256,  # 前馈网络隐藏层维度
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

    def filtered_fft(self, x, keep_ratio=0.5):
        fft = torch.fft.fft(x, dim=2)

        # 2. 创建高频掩膜（与低频相反）
        n_features = x.shape[2]
        mask = torch.ones(n_features)  # 初始全1（保留所有）
        mask[:int(n_features * (1 - keep_ratio))] = 0  # 前(1-keep_ratio)置0（抑制低频）

        # 滤波后逆变换
        filtered = fft * mask.to(x.device)
        return torch.fft.ifft(filtered, dim=2).real  # 仍为[64,28,256]

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None):
        r"""Pass the input through the convolution block.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = ConvBlock(LN(residual))
        """
        hidden_states_residual = hidden_states
        # 1. 如果有残差连接，进行加和操作
        if residual is not None:
            residual = hidden_states + residual
        else:
            residual = hidden_states
        k_v = hidden_states
        # 3. 通过卷积层进行操作
        # hidden_states1 = self.conv_layer_1(hidden_states) + hidden_states
        batchsize = hidden_states.shape[0]
        cluster_number = hidden_states.shape[1]

        hidden_states = hidden_states.permute(0, 2, 1).contiguous()
        # 3. 通过卷积层进行操作
        hidden_states1 = self.conv_layer_1(hidden_states) + hidden_states
        hidden_states2 = self.conv_layer_3(hidden_states) + hidden_states
        hidden_states3 = self.conv_layer_5(hidden_states) + hidden_states

        hidden_states1 = hidden_states1.permute(0, 2, 1).contiguous()
        hidden_states2 = hidden_states2.permute(0, 2, 1).contiguous()
        hidden_states3 = hidden_states3.permute(0, 2, 1).contiguous()

        hidden_states1 = torch.fft.fft(hidden_states1, dim=2).abs().permute(0, 2, 1) + hidden_states
        # hidden_states2 = self.conv_layer_3(hidden_states) + hidden_states
        fft2 = torch.fft.fft(hidden_states2, dim=2)

        hidden_states2 = fft2.real + fft2.imag
        hidden_states2 = hidden_states2.permute(0, 2, 1) + hidden_states

        # hidden_states3 = self.conv_layer_5(hidden_states) + hidden_states
        hidden_states3 = self.filtered_fft(hidden_states3, keep_ratio=0.3).permute(0, 2, 1) + hidden_states

        hidden_states1 = hidden_states1.permute(0, 2, 1).contiguous()
        hidden_states2 = hidden_states2.permute(0, 2, 1).contiguous()
        hidden_states3 = hidden_states3.permute(0, 2, 1).contiguous()

        atten_states1, _ = self.attention(hidden_states1, hidden_states1, hidden_states1)
        atten_states1 = atten_states1 + hidden_states1
        atten_states2, _ = self.attention(hidden_states2, hidden_states2, hidden_states2)
        atten_states3, _ = self.attention(hidden_states3, hidden_states3, hidden_states3)
        atten_states3 = atten_states3 + hidden_states3

        cross_states1, _ = self.attention(atten_states2, atten_states1, atten_states1)
        cross_states2, _ = self.attention(atten_states2, atten_states3, atten_states3)
        fused_feature_map = cross_states1 * cross_states2
        fused_feature = fused_feature_map * atten_states2
        encodings_point_cloud = hidden_states_residual + fused_feature

        attn_output, attn_weights = self.cross_attention(
            query=k_v,
            key=encodings_point_cloud,
            value=encodings_point_cloud
        )
        attn_output = attn_output + encodings_point_cloud
        # 双向LSTM处理
        lstm_output, _ = self.gru(attn_output)

        # === 替换部分：用 Transformer 替代卷积操作 ===
        lstm_output_res = lstm_output
        lstm_output = self.transformer_layer(lstm_output)  # Transformer 处理
        lstm_output = lstm_output + lstm_output_res  # 残差连接

        indices = list(range(0, 512, 2))  # 这将生成 [0, 2, 4, ..., 510]

        narrowed_tensor = lstm_output[:, :, indices]

        return narrowed_tensor, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        # 如果需要，可能用于推理阶段的缓存分配
        return None