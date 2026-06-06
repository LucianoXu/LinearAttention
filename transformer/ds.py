import numpy as np

def get_tiny_shakespeare(path: str):
    # `datasets` is only needed for this one-off download, and it is NOT part of
    # the MPCDF pytorch module. Import it lazily so that training (which only
    # reads the already-downloaded .txt) never depends on it.
    from datasets import load_dataset

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    ds = load_dataset("text", data_files=url, sample_by="document")

    with open(path, 'w') as p:
        p.write(ds["train"]['text'][0])



def txt_to_np(path_to_txt: str) -> np.ndarray:
    with open(path_to_txt, 'r') as p:
        txt = p.read()

    # np.frombuffer returns a read-only view; copy so downstream tensors are writable
    arr = np.frombuffer(txt.encode("utf-8"), dtype=np.uint8).copy()
    return arr


def circle_slice(arr: np.ndarray, idx: int, l: int) -> np.ndarray:
    '''
    idx: 0 ~ size - 1
    size >= l
    '''
    size = len(arr)
    if size >= idx + l:
        return arr[idx:idx+l]
    
    else:
        # circling scenario
        l_ = idx + l - size
        # np.concatenate works on every numpy version; np.concat is a numpy>=2.0
        # alias that does NOT exist in the container's numpy 1.24.
        return np.concatenate((arr[idx:], arr[:l_]))