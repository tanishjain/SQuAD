"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, context_indices, question_indices):
        context_mask = torch.zeros_like(context_indices) != context_indices
        question_mask = torch.zeros_like(question_indices) != question_indices
        context_len, question_len = context_mask.sum(-1), question_mask.sum(-1)

        context_emb = self.emb(context_indices)         # (batch_size, context_len, hidden_size)
        question_emb = self.emb(question_indices)         # (batch_size, question_len, hidden_size)

        c_enc = self.enc(context_emb, context_len)    # (batch_size, context_len, 2 * hidden_size)
        q_enc = self.enc(question_emb, question_len)    # (batch_size, question_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       context_mask, question_mask)    # (batch_size, context_len, 8 * hidden_size)

        mod = self.mod(att, context_len)        # (batch_size, context_len, 2 * hidden_size)

        out = self.out(att, mod, context_mask)  # 2 tensors, each (batch_size, context_len)

        return out

# TODO: Add QANet Implementation
class QANet(nn.Module):
    """QANet model for SQuAD.

    Based on the paper:
    "QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension"
    by Adams Wei Yu, David Dohan, Minh-Thang Luong, Rui Zhao, Kai Chen, Mohammad Norouzi, Quoc V. Le

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
        context_len (int): Length of context.
        question_len (int): Length of question.

    """
    def __init__(self, word_vectors, hidden_size=128, drop_prob=0., context_len=400, question_len=50, word_embed=300, heads=4):
        super(QANet, self).__init__()

        self.context_emb = layers.InputLayer(word_vectors=word_vectors, drop_prob=0.1)
        self.question_emb = layers.InputLayer(word_vectors=word_vectors, drop_prob=0.1)

        self.embedding_args = (3, # number of convolutions
            5,      # size of kernel
            64,     # number of kernels
            4,      # number of heads
            1,      # number of encoder blocks
            0.0,    # dropout probability
            context_len, 
            word_embed,
            hidden_size)

        self.context_encoder = layers.EmbeddingEncoder(*(self.embedding_args))
        self.question_encoder = layers.EmbeddingEncoder(*(self.embedding_args))

        self.self_att = layers.ContextQueryLayer(hidden_size=hidden_size, drop_prob=drop_prob)
        self.convolution = layers.ConvBlock(word_embed=hidden_size*4, sent_len=context_len, hidden_size=hidden_size, kernel_size=5)
        self.mod_enc = layers.ModelEncoder(*(self.embedding_args))

        self.start_span = layers.OutputLayer(drop_prob=drop_prob, word_embed=hidden_size) 
        self.end_span = layers.OutputLayer(drop_prob=drop_prob, word_embed=hidden_size)  

    def forward(self, context_indices, question_indices):
        context_mask = torch.as_tensor( [1 if i != 0 else 0 for i in context_indices] )
        context_mask = torch.as_tensor( [1 if i != 0 else 0 for i in question_indices] )

        context_emb = self.context_emb(context_indices) 
        question_emb = self.question_emb(question_indices)

        context_encoder = self.context_encoder(context_emb, context_mask) 
        question_encoder  = self.question_encoder(question_emb, question_mask)
        self_att = self.self_att(context_encoder, question_encoder, context_mask, question_mask) 
        convolution = self.convolution(self_att) 

        a = self.mod_enc(convolution, context_mask) 
        b = self.mod_enc(a, context_mask) 
        start_span = self.start_span(mod_enc_1, mod_enc_2, context_mask)
        c = self.mod_enc(b, context_mask) 
        end_span = self.end_span(a, c, context_mask)
        out = start_span, end_span

        return out