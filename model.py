import numpy as np
import pandas as pd
# TORCH:
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Klassen

# Daten importieren
class TrainingsDataset(Dataset):
    def __init__(self,daten):
        
        self.data = np.load(f"data/{daten}")
        self.x = torch.from_numpy(self.data["x"]).float()
        self.y = torch.from_numpy(self.data["y"]).float()
    
        if len(self.x.shape) == 3:
            self.x=self.x.unsqueeze(1)

    def __len__(self): return len(self.x)
    def __getitem__(self, index): return self.x[index],self.y[index]

# Model 

# inspired by: Attosecond Streaking Phase Retrieval Via Deep Learning Methods (page 7, CNN)
class Model(nn.Module):
    def __init__(self,start_layer, *args, **kwargs):
        
        # Stand 20260107: start_layer=800*40*30=960000=9.6*1e5
    