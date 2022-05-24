import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import AutoModel


class TokenEmbedding(nn.Module):
    """Map tokens to a word embedding, using a lookup table from a pretrained
    language model. By default, embeddings are frozen so no gradient updates
    are performed to this module.

    Args:
        base_model (str): Name of the base pre-trained language model
    """
    def __init__(self, base_model: str):
        super().__init__()
        base_model = AutoModel.from_pretrained(base_model)
        self.token_embeddings = base_model.embeddings.word_embeddings
        self.token_embeddings.weight.requires_grad = False
        self.embedding_dim = self.token_embeddings.embedding_dim

    def forward(self, input_ids: torch.Tensor):
        return self.token_embeddings(input_ids)


class TextEncoder(nn.Module):
    """Apply a pre-trained transformer encoder to a sequence of token
    embeddings.

    Args:
        base_model (str): Name of the base pre-trained language model

    Inputs:
        - input_embeds (torch.Tensor): shape (batch_size, length)
        - mask (torch.Tensor): shape (batch_size, length)

    Outputs:
        h (torch.Tensor): shape (batch_size, length, dim)
    """
    def __init__(self, base_model: str):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(base_model)
        self.out_features = self.encoder.config.hidden_size

    def forward(self, input_embeds: torch.Tensor, mask: torch.Tensor):
        h = self.encoder(input_ids=None, attention_mask=mask,
                         inputs_embeds=input_embeds).last_hidden_state
        return h


class SlotAttention(nn.Module):
    """A variant of Slot Attention, originally proposed in
    Locatello et al. 'Object-Centric Learning with Slot Attention' (2020).
    Performs a soft clustering of vectors at the input into a predefined
    number of slots.

    This variant performs clustering over (contextualized) word embeddings,
    and includes a 'default' slot at a fixed position with a learned
    representation.

    Args:
        in_features (int): Dimension of input in the last dimension
        num_slots (int): Number of slots
        num_iters (int): Number of iterations of the clustering algorithm

    Inputs:
        - input_embeds (torch.Tensor): shape (batch_size, length, dim)
        - mask (torch.Tensor): shape (batch_size, length)

    Outputs:
        attention (torch.Tensor): coefficients mapping inputs to slots,
        shape (batch, 1 + num_slots, length). In the '1 + num_slots'
        dimension, the first slot corresponds to the default slot.
        Coefficients are normalized across slots.
    """
    # Adapted from https://github.com/lucidrains/slot-attention
    def __init__(self, in_features: int, num_slots: int, num_iters: int):
        super().__init__()
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.eps = 1e-8
        self.scale = in_features ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, in_features))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, in_features))
        nn.init.xavier_uniform_(self.slots_logsigma)
        self.default_slot = nn.Parameter(torch.randn(1, 1, in_features))

        self.to_q = nn.Linear(in_features, in_features)
        self.to_k = nn.Linear(in_features, in_features)
        self.to_v = nn.Linear(in_features, in_features)

        self.gru = nn.GRUCell(in_features, in_features)

        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features)
        )

        self.norm_input = nn.LayerNorm(in_features)
        self.norm_slots = nn.LayerNorm(in_features)
        self.norm_pre_ff = nn.LayerNorm(in_features)

    def forward(self, input_embeds, mask):
        b, n, d, device = *input_embeds.shape, input_embeds.device

        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(b, self.num_slots, -1)

        random_slots = mu + sigma * torch.randn(mu.shape, device=device)
        slots = torch.cat((self.default_slot.expand(b, 1, -1), random_slots),
                          dim=1)

        input_embeds = self.norm_input(input_embeds)
        k, v = self.to_k(input_embeds), self.to_v(input_embeds)

        mask = mask.unsqueeze(1)
        attention = None
        for _ in range(self.num_iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attention = (dots.softmax(dim=1) + self.eps) * mask
            weights = attention / attention.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, weights)

            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return attention


class EntityExtractor(nn.Module):
    """A module to assign subsequences of vectors in an input sequence to a
    predefined number of slots. Input embeddings are passed to a
    Transformer-based pretrained encoder, followed by a SlotAttention layer.

    Args
        base_model (str): Name of the base pre-trained language model
        num_slots (int): Number of slots to which vectors are assigned
        num_iters (int): Number of iterations of the slot asssignment algorithm
        stretch (bool): if True, attention coefficients are stretched to [0, 1]

    Inputs:
        - input_embeds (torch.Tensor): shape (batch_size, length, dim)
        - mask (torch.Tensor): shape (batch_size, length)

    Outputs:
        - selections (torch.Tensor): embeddings selected from the input after
            applying the resulting attention for each slot.
            shape (batch, num_slots, length, dim)
        - non_default_attention (torch.Tensor): attention coefficients for all
            slots except the default. shape (batch, num_slots, length)
        - full_attention (torch.Tensor): coefficients for all slots.
            shape (batch, 1 + num_slots, length)
    """
    def __init__(self, base_model: str, num_slots: int, num_iters: int,
                 stretch: bool):
        super().__init__()
        self.encoder = TextEncoder(base_model)
        self.slot_attention = SlotAttention(self.encoder.out_features,
                                            num_slots, num_iters)
        self.hidden_size = self.encoder.out_features
        self.register_buffer('temperature', torch.tensor([0.5]))
        self.stretch = stretch

    def forward(self, input_embeds, mask):
        h = self.encoder(input_embeds, mask)
        full_attention = self.slot_attention(h, mask)

        # Stretch values to the closed interval [0, 1]
        if self.stretch:
            full_attention = torch.clamp(full_attention * 1.2 - 0.1,
                                         min=0.0, max=1.0)

        non_default_attention = full_attention[:, 1:]
        selections = non_default_attention.unsqueeze(-1) * input_embeds.unsqueeze(1)

        return selections, non_default_attention, full_attention


class ConvolutionalDiscriminator(nn.Module):
    """Given a sequence of embeddings, determines if they correspond to names
    of true entities.

    Args:
        in_features (int): Dimension of the input embeddings

    Inputs:
        - input_embeds (torch.Tensor): shape (batch_size, length, dim)

    Outputs:
        scores (torch.Tensor): shape (batch_size, 1)
    """
    def __init__(self, in_features: int):
        super().__init__()

        cnn = [nn.Conv1d(in_features, out_channels=128,
                         kernel_size=(3,), padding=(3,)),
               nn.ReLU(),
               nn.Conv1d(in_channels=128, out_channels=64,
                         kernel_size=(3,), padding=(3,)),
               nn.ReLU(),
               nn.Conv1d(in_channels=64, out_channels=64,
                         kernel_size=(3,), padding=(3,)),
               nn.ReLU(),
               nn.Conv1d(in_channels=64, out_channels=64,
                         kernel_size=(3,), padding=(3,))]
        self.cnn = nn.Sequential(*cnn)

        self.mlp = nn.Sequential(nn.Linear(in_features=64, out_features=32),
                                 nn.ReLU(),
                                 nn.Linear(in_features=32, out_features=1))

    def forward(self, input_embeds):
        # input_embeds: (batch, seq_length, dim)
        input_embeds = f.pad(input_embeds, (0, 0, 3, 3))
        x = self.cnn(input_embeds.transpose(1, 2))
        x, _ = torch.max(x, dim=-1)
        x = self.mlp(x)
        return x
