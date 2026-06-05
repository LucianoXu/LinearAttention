from datasets import load_dataset
import numpy as np

def get_tiny_shakespeare(path: str):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    ds = load_dataset("text", data_files=url, sample_by="document")

    with open(path, 'w') as p:
        p.write(ds["train"]['text'][0])


def get_enwik8(path: str):
    """Download the enwik8 benchmark (first 100MB of English Wikipedia) and
    write the raw bytes to `path`. Like get_tiny_shakespeare it just produces
    the corpus file; the DataLoader splits it via train_ratio."""
    import io
    import zipfile
    import urllib.request

    url = "http://mattmahoney.net/dc/enwik8.zip"
    with urllib.request.urlopen(url) as resp:
        zipped = resp.read()
    with zipfile.ZipFile(io.BytesIO(zipped)) as z:
        data = z.read("enwik8")          # exactly 100,000,000 bytes, valid UTF-8

    with open(path, "wb") as p:
        p.write(data)



def txt_to_np(path_to_txt: str) -> np.ndarray:
    # byte-level corpus: read raw bytes directly (locale-independent, fast,
    # and exact for non-ASCII like enwik8). .copy() so tensors are writable.
    return np.fromfile(path_to_txt, dtype=np.uint8).copy()


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
        return np.concat((arr[idx:], arr[:l_]))