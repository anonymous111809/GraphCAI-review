import torch
from esm.models.esmc import ESMC
from esm.tokenization.sequence import SequenceTokenizer
from Bio import SeqIO
import numpy as np
import os

device = torch.device("cuda:6" if torch.cuda.is_available() and torch.cuda.device_count() > 6 else "cpu")

# 1. Load ESMC-600M
model = ESMC.from_pretrained("esmc-600m", device=device)
# model.eval()

# 2. Initialize tokenizer
tokenizer = SequenceTokenizer()

fasta_path = "/path/to/Dataset/NN/fasta/allseq.fa"
output_directory = "/path/to/Dataset/NN/esmc_600M"
os.makedirs(output_directory, exist_ok=True)

with open(fasta_path) as fasta_file:
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        print(record.id, len(seq))

        # 3. Encode a single sequence
        tokens = tokenizer.encode(seq)
        tokens = tokens.to(device)

        with torch.no_grad():
            embeddings = model(tokens.unsqueeze(0))["embeddings"]  # [1, L, 5120]
        embeddings = embeddings.squeeze(0).cpu().numpy()

        np.save(os.path.join(output_directory, f"{record.id}.npy"), embeddings)
        print(f"Saved {record.id}.npy")
