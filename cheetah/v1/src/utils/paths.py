import os

def normalizer_path_from_checkpoint(checkpoint_path: str) -> str:
    base, ext = os.path.splitext(checkpoint_path)
    return f"{base}_normalizer.pkl"