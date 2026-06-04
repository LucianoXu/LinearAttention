import numpy as np


def test_circle_slice():
    from la.ds import circle_slice

    arr = np.array(list(range(10)))

    assert np.all(circle_slice(arr, 2, 2) == np.array([2, 3]))
    assert np.all(circle_slice(arr, 0, 10) == arr)
    assert np.all(circle_slice(arr, 4, 8) == np.array([4,5,6,7,8,9,0,1]))


def test_rope():
    from la.transformer import RoPE, TransformerArgs
    import torch

    args = TransformerArgs()
    rope = RoPE(args)

    a = torch.rand(1, args.context_len, args.dim)
    a = a / torch.norm(a, dim=-1, keepdim=True)
    pos_emb = rope.forward(a)

    assert torch.allclose(
        torch.norm(pos_emb, dim=-1, keepdim=True),
        torch.ones_like(pos_emb)
        )