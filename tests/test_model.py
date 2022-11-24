import unittest
import torch
from motion_inbetween.model import transformer


class TestTransformer(unittest.TestCase):
    def test_sinusoid_pos_encoding(self):

        dim = 128
        seq_len = 100
        pos = torch.arange(seq_len)
        pe_module = transformer.SinusoidPositionalEncoding(dim)
        pe = pe_module(pos)

        self.assertEqual(pe.shape, (1, seq_len, dim))

        if False:
            # visualize positional encoding
            import matplotlib.pyplot as plt
            plt.imshow(pe[0])
            plt.imshow(pe[0])
            plt.colorbar()
            plt.xlabel("position encoding dimension")
            plt.ylabel("sequence length")
            plt.title("Sinusoid Positional Encoding")
            plt.show()

    def test_positionwise_feed_forward(self):
        batch = 100
        dim = 128
        dim_inner = 64
        x = torch.rand(batch, dim)
        pffn = transformer.PositionwiseFeedForward(dim, dim_inner)
        y = pffn(x)
        self.assertEqual(y.shape, (batch, dim))

    def test_multi_headed_attention(self):
        batch = 100
        seq = 20
        n_head = 8
        d_model = 1024
        d_head = 128
        mem_len = 40

        hidden = torch.randn(batch, seq, d_model)
        memory = torch.randn(batch, mem_len, d_model)
        mask = torch.ones(seq, seq + mem_len).tril()

        multi_head_att = transformer.MultiHeadedAttention(
            n_head, d_model, d_head, pre_lnorm=True)

        multi_head_att.train()
        output = multi_head_att(hidden, memory, mask)
        self.assertEqual(output.shape, (batch, seq, d_model))

        multi_head_att.eval()
        output = multi_head_att(hidden, memory, mask)
        self.assertEqual(output.shape, (batch, seq, d_model))
