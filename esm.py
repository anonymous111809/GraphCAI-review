import torch
import esm
import os
from Bio import SeqIO
import numpy as np

device = torch.device(f"cuda:6" if torch.cuda.is_available() and torch.cuda.device_count() > 6 else "cpu")

model_name = 'esm2_t36_3B_UR50D'  
model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
model = model.to(device)
model.eval()
fasta_path = '/home/xxx/Dataset/NN/fasta/allseq.fa'
# output_directory = 'esm_ebd_DECOY_train'
output_directory = '/home/xxx/Dataset/NN/esmc_6B'
os.makedirs(output_directory, exist_ok=True)

with open(fasta_path, 'r') as fasta_file:
    for record in SeqIO.parse(fasta_file, "fasta"):
        print(f"Sequence ID: {record.id}")
        print(f"Sequence: {record.seq}")
        sequence = str(record.seq)
        print(f"Length: {len(sequence)}")
        data = [(record.id, sequence)]

        batch_converter = alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)

        tokens_embs = results['representations'][33].cpu().numpy()

        np.save(os.path.join(output_directory, f'{record.id}.npy'), tokens_embs)
        print(f'Saved embeddings for {record.id}')
