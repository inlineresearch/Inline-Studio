"""Read a safetensors checkpoint tensor by tensor, without mapping the whole file.

``safetensors.safe_open`` maps the entire file at once, and Linux refuses a mapping larger than
physical RAM when there is no swap (the default heuristic overcommit mode). A 26GB Krea 2 checkpoint
is therefore unreadable on a 16GB machine - it fails with ``Cannot allocate memory`` before any GPU
work, no matter how small the model would be once quantized.

Reading each tensor's byte range instead keeps peak host RAM at one tensor and works for any file
size. Only used for the big single-file checkpoints; small files (LoRAs, VAEs) still go through
safetensors directly.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from ..errors import ComponentError

#: safetensors dtype names -> torch dtypes, resolved lazily so importing this module stays cheap.
_DTYPE_NAMES = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


class CheckpointReader:
    """Random access to one safetensors file, one tensor at a time."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        try:
            size = self._path.stat().st_size
            with self._path.open("rb") as handle:
                (header_len,) = struct.unpack("<Q", handle.read(8))
                # Garbage bytes decode to an enormous length; reject it rather than trying to
                # allocate it, which would raise MemoryError instead of a usable message.
                if not 0 < header_len <= size - 8:
                    raise ValueError("header length is not plausible for this file")
                header = json.loads(handle.read(header_len))
            if not isinstance(header, dict):
                raise ValueError("header is not a JSON object")
        except (OSError, ValueError, struct.error) as error:
            raise ComponentError(f"Could not read checkpoint {self._path.name}: {error}") from error
        self._start = 8 + header_len
        self._index: dict[str, Any] = {k: v for k, v in header.items() if k != "__metadata__"}
        self.metadata: dict[str, Any] = header.get("__metadata__") or {}

    def keys(self) -> list[str]:
        return list(self._index)

    def get_tensor(self, key: str, device: str | None = None) -> Any:
        """One tensor, read straight from its byte range into a fresh buffer."""
        import torch

        entry = self._index[key]
        dtype_name = _DTYPE_NAMES.get(entry["dtype"])
        if dtype_name is None:
            raise ComponentError(
                f"Checkpoint {self._path.name} uses the unsupported dtype {entry['dtype']!r} "
                f"for {key!r}."
            )
        start, end = entry["data_offsets"]
        buffer = bytearray(end - start)
        with self._path.open("rb") as handle:
            handle.seek(self._start + start)
            if handle.readinto(buffer) != len(buffer):
                raise ComponentError(f"Checkpoint {self._path.name} is truncated at {key!r}.")
        tensor = torch.frombuffer(buffer, dtype=getattr(torch, dtype_name))
        tensor = tensor.reshape(entry["shape"]) if entry["shape"] else tensor.reshape(())
        return tensor.to(device) if device else tensor

    def __enter__(self) -> CheckpointReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None
