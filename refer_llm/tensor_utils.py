import json, io
import torch
from torch import Tensor

def tensor_to_bytes(t: Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()

def bytes_to_tensor(b: bytes) -> Tensor:
    return torch.load(io.BytesIO(b), weights_only=True)

def make_bytes_list(blist: list[bytes]) -> bytes:
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()

def bytes_list_to_list(b: bytes) -> list[bytes]:
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

