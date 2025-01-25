import torch
from torch import nn

ALT_ARCH = False

class DKTSemRDropAttn(nn.Module):
    def __init__(self, emb_size: int, kc_emb_matrix: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        self.kc_emb_matrix = kc_emb_matrix  # 所有知识点的嵌入矩阵
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
        
        # 添加一个线性变换，用于调整 context_vector 的维度
        self.context_projector = nn.Linear(kc_emb_matrix.shape[1], emb_size)

    def forward(self, batch, use_rdrop=False):
        # Get input vectors from transformed text embeddings and correctness embedding
        text_emb = self.input_encoder(
            torch.concat([batch["teacher_embs"], batch["student_embs"], batch["kc_embs"]], dim=2)
        )
        correctness_emb = self.correctness_encoder(torch.clip(batch["labels"], min=0))
        xemb = text_emb + correctness_emb

        # 注意力机制
        expanded_kc_emb_matrix = self.kc_emb_matrix.T.unsqueeze(0).repeat(batch["kc_embs"].shape[0], 1, 1)
        attention_scores = torch.bmm(batch["kc_embs"], expanded_kc_emb_matrix)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vector = torch.bmm(attention_weights, self.kc_emb_matrix.unsqueeze(0).repeat(batch["kc_embs"].shape[0], 1, 1))

        # 调整 context_vector 的维度
        context_vector = self.context_projector(context_vector)  # [batch_size, seq_len, emb_size]

        # 将上下文向量与 xemb 融合
        xemb = xemb + context_vector

        # Run embeddings through LSTM, compute bilinear with KC embedding matrix to get predictions
        h, _ = self.lstm_layer(xemb)

        # First forward pass
        h1 = self.dropout_layer(h)
        h_text_space1 = self.out_layer(h1)
        y1 = torch.bmm(h_text_space1, self.kc_emb_matrix.T.unsqueeze(0).expand(batch["labels"].shape[0], -1, -1))
        y1 = torch.sigmoid(y1)
        
        # Second forward pass with different dropout mask
        h2 = self.dropout_layer(h)
        h_text_space2 = self.out_layer(h2)
        y2 = torch.bmm(h_text_space2, self.kc_emb_matrix.T.unsqueeze(0).expand(batch["labels"].shape[0], -1, -1))
        y2 = torch.sigmoid(y2)
        
        return y1, y2

