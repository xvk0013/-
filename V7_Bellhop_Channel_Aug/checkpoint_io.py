"""Bounded retry for checkpoint replacement on Windows-mounted filesystems."""
import time
from pathlib import Path
import torch

def replace_checkpoint(temporary, target, *, attempts=12):
    temporary, target = Path(temporary), Path(target)
    for attempt in range(attempts):
        try:
            temporary.replace(target)
            return
        except PermissionError as exc:
            if attempt == attempts-1:
                raise PermissionError(
                    f"Checkpoint replacement stayed blocked: {temporary} -> {target}. "
                    "The complete temporary file is preserved; no checkpoint was deleted."
                ) from exc
            if attempt == 0:
                print(f"Checkpoint temporarily locked; retrying replacement: {target.name}", flush=True)
            time.sleep(min(0.1 * 2**attempt, 1.0))

def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    replace_checkpoint(temporary, path)

def recover_best(path, latest):
    """last is committed before best; repair an interrupted best promotion on resume."""
    path = Path(path)
    if latest["stale"] != 0:
        return False
    expected = (latest["epoch"], latest["global_step"], latest["best"])
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        actual = (saved["epoch"], saved["global_step"], saved["best"])
        del saved
        if actual == expected:
            return False
        if actual[0] > expected[0]:
            raise ValueError("Best checkpoint is newer than resume checkpoint; refusing to overwrite")
    print(f"Recovering interrupted best checkpoint at epoch {latest['epoch']}", flush=True)
    # Rebuild from the fully loaded committed last; never trust a possibly partial .tmp.
    save_checkpoint(path, latest)
    return True
