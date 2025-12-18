import torch


def check_safe(data: torch.Tensor) -> bool:
    """
    Checks if the input data contains any NaN or infinite values.

    Args:
        data (torch.Tensor): Input data to be checked.
    """
    flag = True
    if torch.isnan(data).any():
        print("Input data contains NaN values.")
        print(torch.isnan(data).nonzero(as_tuple=True))
        print(torch.isnan(data).nonzero(as_tuple=False))
        flag = False
    if torch.isinf(data).any():
        print("Input data contains infinite values.")
        print(torch.isinf(data).nonzero(as_tuple=True))
        print(torch.isinf(data).nonzero(as_tuple=False))
        flag = False
    
    return flag