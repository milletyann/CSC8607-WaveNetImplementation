import numpy as np
import torch
import torch.nn.functional as F

def mu_law_companding_transformation(x, mu=255):
    return np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu))

# quantizer en 8bit
# tentative de définition des types pour les entrées et sorties
def quantize(wav: torch.Tensor, bit: int) -> torch.Tensor:
    wav = mu_law_companding_transformation(wav, 2**bit - 1)
    return ((wav + 1) * 2**(bit - 1)).to(torch.int)