import torch
from torch import nn
import torch.nn.functional as F

# 门控机制类
class GatedFusion(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        # 定义一个线性层，用于生成门控值 g
        self.gate_layer = nn.Linear(emb_size * 2, emb_size)

    def forward(self, x_emb, context_vector):
        # 拼接输入特征和上下文向量
        combined = torch.cat([x_emb, context_vector], dim=-1)  # [batch_size, seq_len, emb_size * 2]
        
        # 通过线性层生成门控值
        gate = torch.sigmoid(self.gate_layer(combined))  # [batch_size, seq_len, emb_size]
        
        # 融合 x_emb 和 context_vector
        fused_output = gate * x_emb + (1 - gate) * context_vector  # [batch_size, seq_len, emb_size]
        
        return fused_output


# 改进后的 DKTSemRDropAttn 模型
class DKTSemRDropAttn(nn.Module):
    def __init__(self, emb_size: int, kc_emb_matrix: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        self.kc_emb_matrix = kc_emb_matrix  # 所有知识点的嵌入矩阵
        text_emb_size = kc_emb_matrix.shape[1]  # 知识点嵌入的维度
        
        # 输入编码器
        self.input_encoder = nn.Sequential(
            nn.Linear(text_emb_size * 3, emb_size),  # 输入是拼接的 teacher_embs, student_embs, kc_embs
            nn.Tanh()
        )
        
        # 正确性编码器（0或1的embedding）
        self.correctness_encoder = nn.Embedding(2, emb_size)
        
        # LSTM 层
        self.lstm_layer = nn.LSTM(emb_size, emb_size, batch_first=True)
        
        # Dropout 层
        self.dropout_layer = nn.Dropout(dropout)
        
        # 输出层
        self.out_layer = nn.Linear(emb_size, text_emb_size)
        
        # 添加一个线性变换，用于调整 context_vector 的维度
        self.context_projector = nn.Linear(kc_emb_matrix.shape[1], emb_size)
        
        # 添加门控机制
        self.gated_fusion = GatedFusion(emb_size)
        
        # 可学习的 Query 和 Key 变换矩阵
        self.query_transform = nn.Linear(kc_emb_matrix.shape[1], emb_size)
        self.key_transform = nn.Linear(kc_emb_matrix.shape[1], emb_size)

    def forward(self, batch, use_rdrop=False):
        """
        前向传播
        :param batch: 包含以下键的字典
            - teacher_embs: 教师嵌入 [batch_size, seq_len, emb_dim]
            - student_embs: 学生嵌入 [batch_size, seq_len, emb_dim]
            - kc_embs: 知识点嵌入 [batch_size, seq_len, emb_dim]
            - labels: 正确性标签 [batch_size, seq_len]
        :param use_rdrop: 是否使用 R-drop 正则化
        :return: y1, y2 两次前向传播的预测值
        """
        # 获取输入特征向量
        text_emb = self.input_encoder(
            torch.concat([batch["teacher_embs"], batch["student_embs"], batch["kc_embs"]], dim=2)
        )  # [batch_size, seq_len, emb_size]
        
        # 正确性嵌入
        correctness_emb = self.correctness_encoder(torch.clip(batch["labels"], min=0))  # [batch_size, seq_len, emb_size]
        
        # 输入特征与正确性特征相加
        xemb = text_emb + correctness_emb  # [batch_size, seq_len, emb_size]

        # ========================= 改进的注意力机制 =========================
        # 对知识点嵌入矩阵进行 Query 和 Key 的线性变换
        query = self.query_transform(batch["kc_embs"])  # [batch_size, seq_len, emb_size]
        key = self.key_transform(self.kc_emb_matrix)  # [num_kcs, emb_size]

        # 归一化 Query 和 Key
        query = F.normalize(query, dim=-1)  # [batch_size, seq_len, emb_size]
        key = F.normalize(key, dim=-1)  # [num_kcs, emb_size]

        # 计算注意力分数 (scaled dot-product attention)
        attention_scores = torch.bmm(query, key.T.unsqueeze(0).repeat(batch["kc_embs"].shape[0], 1, 1))  # [batch_size, seq_len, num_kcs]
        attention_scores = attention_scores / (key.shape[-1] ** 0.5)  # 缩放因子

        # 计算注意力权重
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, seq_len, num_kcs]

        # 计算上下文向量
        context_vector = torch.bmm(attention_weights, self.kc_emb_matrix.unsqueeze(0).repeat(batch["kc_embs"].shape[0], 1, 1))  # [batch_size, seq_len, kc_emb_dim]

        # 调整上下文向量的维度
        context_vector = self.context_projector(context_vector)  # [batch_size, seq_len, emb_size]

        # ========================= 门控机制融合 =========================
        xemb = self.gated_fusion(xemb, context_vector)  # 使用门控机制融合输入特征和上下文向量

        # ========================= LSTM 层 =========================
        h, _ = self.lstm_layer(xemb)  # [batch_size, seq_len, emb_size]

        # ========================= Dropout 和输出 =========================
        # 第一次前向传播
        h1 = self.dropout_layer(h)
        h_text_space1 = self.out_layer(h1)  # [batch_size, seq_len, kc_emb_dim]
        y1 = torch.bmm(h_text_space1, self.kc_emb_matrix.T.unsqueeze(0).expand(batch["labels"].shape[0], -1, -1))  # [batch_size, seq_len, num_kcs]
        y1 = torch.sigmoid(y1)

        # 第二次前向传播（用于 R-Drop）
        h2 = self.dropout_layer(h)
        h_text_space2 = self.out_layer(h2)  # [batch_size, seq_len, kc_emb_dim]
        y2 = torch.bmm(h_text_space2, self.kc_emb_matrix.T.unsqueeze(0).expand(batch["labels"].shape[0], -1, -1))  # [batch_size, seq_len, num_kcs]
        y2 = torch.sigmoid(y2)

        return y1, y2
