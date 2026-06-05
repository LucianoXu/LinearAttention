import numpy as np


def test_circle_slice():
    from .ds import circle_slice

    arr = np.array(list(range(10)))

    assert np.all(circle_slice(arr, 2, 2) == np.array([2, 3]))
    assert np.all(circle_slice(arr, 0, 10) == arr)
    assert np.all(circle_slice(arr, 4, 8) == np.array([4,5,6,7,8,9,0,1]))


def test_rope():
    from .model import RoPE, ModelArgs
    import torch

    args = ModelArgs()
    rope = RoPE(args)

    head_dim = args.dim // args.head
    a = torch.rand(1, args.context_len, head_dim)   # RoPE now operates per head (head_dim)
    a = a / torch.norm(a, dim=-1, keepdim=True)
    pos_emb = rope.forward(a)

    assert torch.allclose(
        torch.norm(pos_emb, dim=-1, keepdim=True),
        torch.ones_like(pos_emb)
        )
    
def test_recurrent_parallel_equivalence():
    from .model import Kath20LA, ModelArgs
    import torch

    model_args = ModelArgs()

    model = Kath20LA(model_args).eval()

    B = 4

    causal_mask = torch.tril(torch.ones(model_args.context_len, model_args.context_len, dtype=torch.int, device='cpu'))

    input = torch.randint(0, model_args.vocab_size, (B, model_args.context_len))

    with torch.no_grad():
        # parallel
        logits1 = model.forward(input, causal_mask)

        # recurrent: collect logits at every step -> (B, L, V)
        kv_state, phik_state = None, None
        rec = []
        for i in range(model_args.context_len):
            output, kv_state, phik_state = model.recurrent_forward(input[:, i].unsqueeze(-1), i, kv_state, phik_state)
            rec.append(output[:, -1])              # (B, V)
        rec = torch.stack(rec, dim=1)              # (B, L, V)

    # compare ALL positions; loose tolerance for float32 accumulation-order differences
    assert torch.allclose(logits1, rec, rtol=1e-4, atol=1e-4)
