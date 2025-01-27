import torch
from torch import nn

ALT_ARCH = False

class DKTSem(nn.Module):
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

    def forward(self, batch, use_rdrop=False):
        # Get input vectors from transformed text embeddings and correctness embedding
        if ALT_ARCH:
            xemb = self.input_encoder(batch["turn_embs"])
        else:
            text_emb = self.input_encoder(
                torch.concat([batch["teacher_embs"], batch["student_embs"], batch["kc_embs"]], dim=2)
            )
            correctness_emb = self.correctness_encoder(torch.clip(batch["labels"], min=0))
            xemb = text_emb + correctness_emb

        if use_rdrop:
            # First forward pass with dropout applied on xemb
            xemb1 = self.dropout_layer(xemb)
            h1, _ = self.lstm_layer(xemb1)
            h1 = self.dropout_layer(h1)
            h_text_space1 = self.out_layer(h1)
            y1 = torch.bmm(h_text_space1, self.kc_emb_matrix.T.unsqueeze(0).expand(batch["labels"].shape[0], -1, -1))
            y1 = torch.sigmoid(y1)

            # Second forward pass with different dropout mask on xemb
            xemb2 = self.dropout_layer(xemb)
            h2, _ = self.lstm_layer(xemb2)
            h2 = self.dropout_layer(h2)
            h_text_space2 = self.out_layer(h2)
            y2 = torch.bmm(h_text_space2, self.kc_emb_matrix.T.unsqueeze(0).expand(batch["labels"].shape[0], -1, -1))
            y2 = torch.sigmoid(y2)

            return y1, y2
        else:
            # Add missing LSTM and dropout operations for the `else` branch
            xemb = self.dropout_layer(xemb)  # Apply dropout
            h, _ = self.lstm_layer(xemb)    # Pass through LSTM
            h = self.dropout_layer(h)       # Apply dropout again
            h_text_space = self.out_layer(h)
            y = torch.bmm(h_text_space, self.kc_emb_matrix.T.unsqueeze(0).expand(batch["labels"].shape[0], -1, -1))
            y = torch.sigmoid(y)  # B x L x K(all)
            return y
