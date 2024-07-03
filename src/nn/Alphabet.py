from torch import nn

class AlphabetNetwork(nn.Module):
    def __init__(
        self,
        num_tokens,
        emb_size,
        hidden_dim,
        padding
    ):
        super(AlphabetNetwork, self).__init__()
        self.embedding_layer = nn.Embedding(num_tokens, emb_size, padding_idx=padding)
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_dim,
            num_layers=4,
            bidirectional=True,
        )

    def forward(self, x):
        embedding = self.embedding_layer(x)
        out, _ = self.lstm(embedding)

        return out[-1, :, :]