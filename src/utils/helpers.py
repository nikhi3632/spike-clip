import torch

def get_device(prefer: str = "auto", verbose: bool = True) -> str:
    """
    Select the best available compute device (cuda or cpu).

    Args:
        prefer (str): One of {"auto", "cuda", "gpu", "cpu"}.
        verbose (bool): Print which device was selected.
    Returns:
        str: Device string to use with torch.device()
    """
    device = "cpu"

    if prefer in ("cuda", "gpu") and torch.cuda.is_available():
        device = "cuda"
    elif prefer == "auto":
        if torch.cuda.is_available():
            device = "cuda"

    if verbose:
        print(f"Using device: {device.upper()}")
    return device

if __name__ == "__main__":
    print(get_device())
