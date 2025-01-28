import torch
from torch import nn

ALT_ARCH = False

class DKTSemCL(nn.Module):
    def __init__(self, emb_size: int, kc_emb_matrix: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        self.kc_emb_matrix = kc_emb_matrix
        text_emb_size = kc_emb_matrix.shape[1]
        if ALT_ARCH:
            self.input_encoder = nn.Sequential(
                nn.Linear(text_emb_size, emb_size),
                nn.Tanh()
            )
        else:
            self.input_encoder = nn.Sequential(
                nn.Linear(text_emb_size * 3, emb_size),
                nn.Tanh()
            )
        self.correctness_encoder = nn.Embedding(2, emb_size)
        self.lstm_layer = nn.LSTM(emb_size, emb_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.out_layer = nn.Linear(emb_size, text_emb_size)
        
        # 对比学习温度参数
        self.temperature = 0.5
        
    def forward(self, batch, use_rdrop=False):
        # 第一次前向传播
        xemb1, y1 = self._forward_pass(batch)
            
        # 第二次前向传播(不同的dropout mask)
        xemb2, y2 = self._forward_pass(batch)
        
        return (xemb1, y1), (xemb2, y2)
        
    def _forward_pass(self, batch):
        # 获取输入向量
        if ALT_ARCH:
            xemb = self.input_encoder(batch["turn_embs"])
        else:
            text_emb = self.input_encoder(
                torch.concat([batch["teacher_embs"], batch["student_embs"], batch["kc_embs"]], dim=2)
            )
            correctness_emb = self.correctness_encoder(torch.clip(batch["labels"], min=0))
            xemb = text_emb + correctness_emb

        xemb = self.dropout_layer(xemb)  # 应用dropout
        h, _ = self.lstm_layer(xemb)     # 通过LSTM
        h = self.dropout_layer(h)        # 再次应用dropout
        h_text_space = self.out_layer(h)
        y = torch.bmm(h_text_space, self.kc_emb_matrix.T.unsqueeze(0).expand(batch["labels"].shape[0], -1, -1))
        y = torch.sigmoid(y)  # B x L x K(all)
        return xemb, y
        
    def contrastive_loss(self, xemb1, xemb2, neg_xemb):
        """
        计算特征表示空间的对比学习损失
        xemb1, xemb2: 同一输入的两次不同dropout结果得到的特征表示(正例对)
        neg_xemb: 随机抽取的其他样本的特征表示(负例)
        """
        # 对特征进行L2归一化
        xemb1 = nn.functional.normalize(xemb1, dim=-1)
        xemb2 = nn.functional.normalize(xemb2, dim=-1)
        neg_xemb = nn.functional.normalize(neg_xemb, dim=-1)
        
        # 正例对相似度
        pos_sim = torch.cosine_similarity(xemb1, xemb2, dim=-1)
        
        # 负例相似度 
        neg_sim1 = torch.cosine_similarity(xemb1, neg_xemb, dim=-1)
        neg_sim2 = torch.cosine_similarity(xemb2, neg_xemb, dim=-1)
        
        # InfoNCE loss
        pos_term = torch.exp(pos_sim / self.temperature)
        neg_term = torch.exp(neg_sim1 / self.temperature) + torch.exp(neg_sim2 / self.temperature)
        
        loss = -torch.log(pos_term / (pos_term + neg_term))
        return loss.mean()
