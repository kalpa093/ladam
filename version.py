import sys
import torch
import transformers
import pandas as pd
import numpy as np


print(f"Python=={sys.version.split()[0]}")
print(f"torch=={torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"transformers=={transformers.__version__}")
print(f"pandas=={pd.__version__}")
print(f"numpy=={np.__version__}")
