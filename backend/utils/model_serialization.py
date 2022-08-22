import io

import torch


def save(module: torch.jit.ScriptModule) -> bytes:
    with io.BytesIO() as buffer:
        torch.jit.save(module, buffer)
        return buffer.getvalue()


def load(module_data: bytes) -> torch.jit.ScriptModule:
    with io.BytesIO(module_data) as buffer:
        return torch.jit.load(buffer)
