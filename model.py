import pickle
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import h5py
from tqdm import tqdm

MAP_CUTOFF = 8  # Modify the contact-map cutoff here
INPUT_DIM = 64
HIDDEN_DIM = 512
NLAYER = 3
DROPOUT = 0.1
LEARNING_RATE = 5E-5
BATCH_SIZE = 1
NUM_CLASSES = 2
NUMBER_EPOCHS = 300
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_EC1_num(ec_onehot):
    ec1 = list(int(i) for i in ec_onehot)
    ec_index = ec1.index(1.0)
    if ec_index == 0:
        EC_index = 7
    elif ec_index == 1:
        EC_index = 3
    elif ec_index == 2:
        EC_index = 4
    elif ec_index == 3:
        EC_index = 5
    elif ec_index == 4:
        EC_index = 6
    elif ec_index == 5:
        EC_index = 1
    else:
        EC_index = 2
    return EC_index


def remove_nan(matrix, padding_value=0.):
    aa_has_nan = np.isnan(matrix).reshape([len(matrix), -1]).max(-1)
    matrix[aa_has_nan] = padding_value
    return matrix


def get_cluster_center(ec2id, data_path, epoch):
    cluster_center = {}
    if epoch != 0:
        f_read = open(data_path + 'updated_enzfeas/dict_enzfeas.pkl', "rb")
        dict_enzfeas = pickle.load(f_read)
        for ec in tqdm(list(ec2id.keys())):
            avg_pro_feas = []
            for pro_id in ec2id[ec]:
                if pro_id in dict_enzfeas.keys():
                    pro_feature = dict_enzfeas[pro_id][:]
                    avg_pro_feas.append(torch.squeeze(pro_feature))
            cluster_center[ec] = torch.stack(avg_pro_feas, dim=0).mean(dim=0)
            cluster_center[ec] = torch.unsqueeze(cluster_center[ec], 0)  # Add a batch dimension
    else:
        with h5py.File(data_path + 'Prot5/train_per_protein_embeddings.h5', "r") as f:
            for ec in tqdm(list(ec2id.keys())):
                avg_pro_feas = []
                for pro_id in ec2id[ec]:
                    pro_feature = f[pro_id][:]
                    avg_pro_feas.append(pro_feature)
                cluster_center[ec] = torch.from_numpy(np.mean(avg_pro_feas, axis=0))
                cluster_center[ec] = torch.unsqueeze(cluster_center[ec], 0)
    return cluster_center


def embedding(sequence_name, data_path):
    pssm_feature = np.load(data_path + "pssm/processed_pssm/" + sequence_name + '.npy')
    hmm_feature = np.load(data_path + "hhm/processed_hhm/" + sequence_name + '.npy')
    with h5py.File(data_path + 'Prot5/train_per_residue_embeddings.h5', "r") as f:
        evo_feature = f[sequence_name][:]
    return pssm_feature, hmm_feature, evo_feature.astype(np.float32)


def get_atom_features(sequence_name, data_path):
    atom_feature = np.load(data_path + 'Atom_feas/matched_atomfea/' + sequence_name + '.npy')
    atom_feature = remove_nan(atom_feature, padding_value=0.)

    seq_feature = np.load(data_path + 'seqfea/' + sequence_name + '.npy')
    return atom_feature, seq_feature


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** 0.5).flatten()
    for i in range(len(r_inv)):
        r_inv[i] = 0 if r_inv[i] == 0 else 1 / r_inv[i]
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def load_graph(sequence_name, data_path):
    # Use PDB structure
    dismap = np.load(data_path + "contact_map/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= MAP_CUTOFF))
    adjacency_matrix = mask.astype(int)
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return norm_matrix


def convert_sequence_to_ngram(sequence, word_dict_path, n=3):
    # Load the saved word_dict
    word_dict = np.load(word_dict_path, allow_pickle=True).item()

    tokens = []
    for i in range(len(sequence) - n + 1):
        ngram = sequence[i:i + n]
        if ngram in word_dict:
            tokens.append(word_dict[ngram])
        else:
            tokens.append(word_dict.get('<unk>', 0))  # Use 0 as the fallback ID for unknown tokens

    return tokens


class CPIPredictionSetting3(nn.Module):
    def __init__(self, n_word, dim, layer_cnn, device):
        super(CPIPredictionSetting3, self).__init__()
        self.embed_word = nn.Embedding(n_word, dim)  # n-gram embedding
        self.W_cnn = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=2 * 3 + 1,  # window=3
                stride=1,
                padding=3
            ) for _ in range(layer_cnn)
        ])
        self.W_attention = nn.Linear(dim, dim)
        self.simple_layer = nn.Linear(in_features=2 * dim, out_features=dim)
        self.output_layer = nn.Linear(dim, 1)

        self.device = device
        self.dim = dim
        self.layer_cnn = layer_cnn

    def matrix_cnn(self, x, layer):
        x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)  # [1, 1, L, D]
        for i in range(layer):
            x = torch.relu(self.W_cnn[i](x))  # CNN
        x = x.view(-1, self.dim)  # [L, D]
        h = torch.relu(self.W_attention(x))  # Attention projection
        return torch.unsqueeze(torch.mean(h, 0), 0)  # [1, D]

    def attention_p(self, x, xs, layer):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)  # [1, 1, L, D]
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))  # CNN
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)  # [L, D]

        h = torch.relu(self.W_attention(x))  # [1, D]
        hs = torch.relu(self.W_attention(xs))  # [L, D]
        weights_p = torch.tanh(F.linear(h, hs))  # [1, L]
        ys = torch.t(weights_p) * hs  # [L, D]
        return torch.unsqueeze(torch.mean(ys, 0), 0)  # [1, D]

    def forward(self, word_ids, pssm_matrix):
        # word_ids: [L], int token ids
        # pssm_matrix: [L, D], float matrix

        ngram_embed = self.embed_word(word_ids)  # [L, D]
        pssm_vector = self.matrix_cnn(pssm_matrix, self.layer_cnn)  # [1, D]
        fusion_vector = self.attention_p(pssm_vector, ngram_embed, self.layer_cnn)  # [1, D]

        combined = torch.cat([pssm_vector, fusion_vector], dim=1)  # [1, 2D]
        out = torch.relu(self.simple_layer(combined))  # [1, D]
        score = self.output_layer(out)  # [1, 1]
        return score


def load_esm(sequence_name, data_path):
    esm_feature = np.load(data_path + "esm/" + sequence_name + '.npy')

    return esm_feature


def load_interpro_data(protein_id,
                       save_dir='/path/to/GraphCAI/Dataset/training_data/interproscan/process_interproscan'):
    """
    Load the corresponding .pkl file by protein ID and return the data.

    Parameters:
    - protein_id: Protein ID used to determine the .pkl filename.
    - save_dir: Directory where .pkl files are stored.

    Returns:
    - inter_feature: Tuple containing indices and offsets, or None if the file does not exist.
    """
    # Build the .pkl file path
    pkl_file = os.path.join(save_dir, f'{protein_id}.pkl')

    # Check whether the file exists
    if not os.path.exists(pkl_file):
        print(f"File {pkl_file} does not exist.")
        return None

    # Load the .pkl file
    with open(pkl_file, 'rb') as fr:
        inter_feature = pickle.load(fr)

    return inter_feature


def load_interpro_data_test(protein_id,
                            save_dir='/path/to/GraphCAI/Dataset/PC/interproscan/process_interproscan'):
    """
    Load the corresponding .pkl file by protein ID and return the data.

    Parameters:
    - protein_id: Protein ID used to determine the .pkl filename.
    - save_dir: Directory where .pkl files are stored.

    Returns:
    - inter_feature: Tuple containing indices and offsets, or None if the file does not exist.
    """
    # Build the .pkl file path
    pkl_file = os.path.join(save_dir, f'{protein_id}.pkl')

    # Check whether the file exists
    if not os.path.exists(pkl_file):
        print(f"File {pkl_file} does not exist.")
        return None

    # Load the .pkl file
    with open(pkl_file, 'rb') as fr:
        inter_feature = pickle.load(fr)

    return inter_feature


class TransformerBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, head=1):
        super(TransformerBlock, self).__init__()
        self.head = head
        self.hidden_dim = hidden_dim
        self.trans_q_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.trans_k_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.trans_v_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])

        self.concat_trans = nn.Linear(hidden_dim * head, hidden_dim, bias=False)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, residue_h, inter_h):
        """
        Forward pass implementing multi-head attention and a feed-forward network.

        Parameters:
            residue_h (torch.Tensor): Residue features with shape (num_nodes, in_dim).
            inter_h (torch.Tensor): InterPro features with shape (num_nodes, in_dim).

        Returns:
            torch.Tensor: Processed features with shape (num_nodes, hidden_dim).
        """
        multi_output = []
        for i in range(self.head):
            q = self.trans_q_list[i](residue_h)  # Shape: (num_nodes, hidden_dim)
            k = self.trans_k_list[i](inter_h)  # Shape: (num_nodes, hidden_dim)
            v = self.trans_v_list[i](residue_h)  # Shape: (num_nodes, hidden_dim)

            # Compute attention scores using dot product and a scaling factor
            att = torch.sum(torch.mul(q, k) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32)), dim=1,
                            keepdim=True)  # Shape: (num_nodes, 1)

            # Apply global normalization to obtain attention weights
            alpha = F.softmax(att, dim=0)  # Shape: (num_nodes, 1)

            # Multiply values (V) by attention weights to obtain weighted features
            tp = v * alpha  # Shape: (num_nodes, hidden_dim)

            multi_output.append(tp)

        # Concatenate the outputs of all heads along dimension 1
        multi_output = torch.cat(multi_output, dim=1)  # Shape: (num_nodes, hidden_dim * head)

        # Apply a linear transformation to the concatenated features
        multi_output = self.concat_trans(multi_output)  # Shape: (num_nodes, hidden_dim)

        # Residual connection and layer normalization
        multi_output = self.layernorm(multi_output + residue_h)  # Shape: (num_nodes, hidden_dim)

        # Feed-forward network
        multi_output = self.layernorm(self.ff(multi_output) + multi_output)  # Shape: (num_nodes, hidden_dim)

        return multi_output


class inter_model(nn.Module):
    def __init__(self, input_size, hidden_size):  # Number of protein-related entries, 1280
        super(inter_model, self).__init__()

        self.embedding_layer = nn.EmbeddingBag(input_size, hidden_size, mode='sum', include_last_offset=True)

        self.linearLayer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU()
        )

    def forward(self, inter_feature):
        # print("Initial inter_feature[0] (input):", inter_feature[0])
        # print("Initial inter_feature[1] (offsets):", inter_feature[1])
        input_tensor = inter_feature[0].view(-1)  # Convert the tensor to 1D
        offsets_tensor = inter_feature[1].squeeze()  # Convert the tensor to 1D
        # Recombine them into a tuple
        inter_feature = (input_tensor, offsets_tensor)
        # print("inter_feature[0] (input):", inter_feature[0])
        # print("inter_feature[1] (offsets):", inter_feature[1])
        inter_feature = F.relu(self.embedding_layer(*inter_feature))
        inter_feature = self.linearLayer(inter_feature)
        return inter_feature  # (batch_size, hidden_size)


class ProDataset(Dataset):
    def __init__(self, dataframe, data_path):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.EC1 = dataframe['EC1'].values
        self.data_path = data_path

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        EC1 = np.array(self.EC1[index])
        data_path = self.data_path

        pssm_feature, hmm_feature, evo_feature = embedding(sequence_name, data_path)  # evo_feature is read from the H5 file generated by Prot5
        atom_features, seq_feature = get_atom_features(sequence_name,
                                                       data_path)  # atom_features are read from Atom/matched NPY files; seq_feature is read from seqfea NPY files

        esm_feature = load_esm(sequence_name, data_path)
        esm_feature = esm_feature.squeeze(0)
        esm_feature = esm_feature[:-2, :]  # Remove the last two rows

        inter_feature = load_interpro_data(sequence_name)

        node_features = np.concatenate([pssm_feature, hmm_feature, atom_features, seq_feature, esm_feature], axis=1)
        # node_features = np.concatenate([pssm_feature, hmm_feature, atom_features, seq_feature], axis=1)

        graph = load_graph(sequence_name, data_path)
        # print(graph.shape,node_features.shape,evo_feature.shape)

        return sequence_name, sequence, label, EC1.astype(int), node_features, graph, evo_feature, inter_feature

    def __len__(self):
        return len(self.labels)


# class ProDataset(Dataset):
#     def __init__(self, dataframe, data_path):
#         self.names = dataframe['ID'].values
#         self.sequences = dataframe['sequence'].values
#         self.labels = dataframe['label'].values
#         self.EC1 = dataframe['EC1'].values
#         self.data_path = data_path
#         self.error_log = 'error_list.txt'
#
#     def __getitem__(self, index):
#         sequence_name = self.names[index]
#         sequence = self.sequences[index]
#         label = np.array(self.labels[index])
#         EC1 = np.array(self.EC1[index])
#         data_path = self.data_path
#
#         pssm_feature, hmm_feature, evo_feature = embedding(sequence_name, data_path)
#         atom_features, seq_feature = get_atom_features(sequence_name, data_path)
#
#         try:
#             node_features = np.concatenate([pssm_feature, hmm_feature, atom_features, seq_feature], axis=1)
#         except ValueError as e:
#             print(f"[ERROR] {sequence_name} concatenate failed: {e}")
#             print(f"[DEBUG] shapes -> pssm: {pssm_feature.shape}, hmm: {hmm_feature.shape}, atom: {atom_features.shape}, seq: {seq_feature.shape}")
#             self.log_error(sequence_name)
#             # Return placeholder data
#             node_features, graph, evo_feature = self.generate_placeholder()
#             return sequence_name, sequence, label, EC1.astype(int), node_features, graph, evo_feature
#
#         graph = load_graph(sequence_name, data_path)
#
#         # Check shapes
#         if not (
#             graph.shape[0] == graph.shape[1] == node_features.shape[0] == evo_feature.shape[0]
#             and node_features.shape[1] == 64
#             and evo_feature.shape[1] == 1024
#         ):
#             print(f"[ERROR] {sequence_name} shape mismatch: graph: {graph.shape}, node_features: {node_features.shape}, evo: {evo_feature.shape}")
#             self.log_error(sequence_name)
#             # Return placeholder data
#             node_features, graph, evo_feature = self.generate_placeholder()
#             return sequence_name, sequence, label, EC1.astype(int), node_features, graph, evo_feature
#
#         return sequence_name, sequence, label, EC1.astype(int), node_features, graph, evo_feature
#
#     def log_error(self, sequence_name):
#         with open(self.error_log, 'a') as f:
#             f.write(f"{sequence_name}\n")
#
#     def generate_placeholder(self):
#         # Generate placeholder empty features to avoid shape mismatch
#         dummy_node_features = np.zeros((2, 64), dtype=np.float32)
#         dummy_graph = np.zeros((2, 160), dtype=np.float32)
#         dummy_evo_feature = np.zeros((2, 1024), dtype=np.float32)
#         return dummy_node_features, dummy_graph, dummy_evo_feature
#
#     def __len__(self):
#         return len(self.labels)


class GINLayer(nn.Module):
    def __init__(self, nhidden):
        super(GINLayer, self).__init__()
        self.linear1 = nn.Linear(nhidden, nhidden)
        self.linear2 = nn.Linear(nhidden, nhidden)
        self.relu = nn.ReLU()

    def forward(self, node_feat, adj):
        neighbor_agg = torch.matmul(adj, node_feat)
        h = self.relu(self.linear1((1 + 0.1) * node_feat + neighbor_agg))
        h = self.linear2(h)
        return h


class GraphAttentionLayer(nn.Module):
    def __init__(self, nhidden):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = nhidden
        self.out_features = nhidden

        # Learnable parameters: attention mechanism
        self.W = nn.Parameter(torch.zeros(size=(nhidden, nhidden)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * nhidden, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)  # Linear transformation

        # Self-attention mechanism
        N = h.size()[0]  # Number of nodes
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # Masked attention scores
        attention = F.softmax(attention, dim=1)  # Attention coefficients
        h_prime = torch.matmul(attention, h)  # Linear combination using attention scores

        return h_prime


class GraphConvolution(nn.Module):
    def __init__(self, nhidden):
        super(GraphConvolution, self).__init__()

        self.nhidden = nhidden
        self.projection = nn.Linear(self.nhidden, self.nhidden)

        self.weight = Parameter(torch.FloatTensor(self.nhidden, self.nhidden))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.nhidden)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        seq_fea = torch.matmul(input, self.weight)
        output = torch.spmm(adj, seq_fea)
        return output


class CNNModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNModel, self).__init__()

        self.convs = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=5, stride=1, padding=2)
        self.fcs = nn.Linear(input_dim, output_dim)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        pro_fea = torch.unsqueeze(x, 0).permute(0, 2, 1)
        layer_inner = self.convs(pro_fea)
        layer_inner = self.act_fn(layer_inner)

        layer_inner = nn.MaxPool1d(3, stride=1, padding=1)(layer_inner)
        layer_inner = torch.squeeze(layer_inner)

        layer_inner = torch.sum(layer_inner, dim=1)
        layer_inner = self.fcs(layer_inner)
        out_fea = nn.Sigmoid()(layer_inner)

        return out_fea


class predict_ec(nn.Module):
    def __init__(self, hidden_dim):
        super(predict_ec, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1024)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        x = torch.mean(x, dim=0)
        x = torch.relu(self.fc4(x))
        return x


class GCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, dropout):  # NLAYER = 3 nfeat = 64 hidden_dim=512
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden))

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))  # fcs[0]: (N, 64) -> (N, 512)
        self.fcs.append(nn.Linear(1024, nhidden))  # fcs[1]: (B, 1024) -> (B, 512)
        self.fcs.append(nn.Linear(nlayers * nhidden, 1280))  # fcs[2]: (N, 3*512) -> (N, 512)
        self.fcs.append(nn.Linear(2 * nhidden, nhidden))  # fcs[3]: (B, 1024) -> (B, 512)
        self.transformer_block = TransformerBlock(1280, 1280, 4)  # Transformer block
        self.inter_feature_transform = nn.Linear(1280, 512)  # Added linear layer

        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, adj, evo_fea, inter_feature):  # x is node_feature; adj is graph
        """
        x: (N, 64)              - Initial feature of each node
        adj: (N, N)             - Graph adjacency matrix
        evo_fea: (B, 1024)      - Global sequence feature of each graph, e.g., from ProteinBERT
        """
        _layers = []
        local_fea = self.act_fn(self.fcs[0](x))  # (N, 64) -> (N, 512)

        # Stacked graph convolution layers: layers 2 to 4 (NLAYER=3)
        for i, con in enumerate(self.convs):
            local_fea = self.act_fn(con(local_fea, adj))  # Each layer keeps the shape (N, 512)
            _layers.append(local_fea)

        local_fea = self.act_fn(
            self.fcs[2](torch.cat(_layers, 1)))  # [(N, 512), (N, 512), (N, 512)] -> (N, 1536) -> (N, 512)

        global_fea = F.dropout(evo_fea, self.dropout, training=self.training)  # (B, 1024)
        global_fea = self.act_fn(self.fcs[1](global_fea))  # (B, 1024) -> (B, 512)

        inter_fea = inter_feature.unsqueeze(1).repeat(1, global_fea.size(0), 1)  # Broadcast to each node
        inter_fea = inter_fea.squeeze(0)  # Remove the first dimension
        # inter_fea = self.inter_feature_transform(inter_fea)  # (B, 1280) -> (B, 512)

        local_fea = self.transformer_block(local_fea, inter_fea)
        local_fea = self.inter_feature_transform(local_fea)  # (B, 1280) -> (B, 512)

        profeas = self.act_fn(
            self.fcs[-1](torch.cat([global_fea, local_fea], 1)))  # (B, 512) + (N, 512) = (B, 1024) -> (B, 512); B equals N

        return profeas

    # def forward(self, x, adj, evo_fea):
    #     """
    #     x: (N, 65) -> The last dimension is the is_local label (0/1)
    #     adj: (N, N)
    #     evo_fea: (N, 1024)
    #     """
    #     # Split is_local and node features
    #     is_local = x[:, -1]  # (N,)
    #     x = x[:, :-1]  # (N, 64) Remove the last dimension as the real node features
    #
    #     _layers = []
    #     local_fea = self.act_fn(self.fcs[0](x))  # (N, 64) -> (N, 512)
    #
    #     # Process is_local weights
    #     is_local = is_local.unsqueeze(1)  # (N,) -> (N, 1)
    #     weights = is_local * 1.0 + (1 - is_local) * 0  # (N, 1)
    #
    #     # Multi-layer graph convolution and within-layer weighting
    #     for con in self.convs:
    #         local_fea = self.act_fn(con(local_fea, adj))  # (N, 512)
    #         local_fea = local_fea * weights  # Use weighting to emphasize active sites
    #         _layers.append(local_fea)
    #
    #     # Feature aggregation
    #     local_fea = self.act_fn(self.fcs[2](torch.cat(_layers, dim=1)))  # (N, 1536) -> (N, 512)
    #
    #     # Process global sequence features
    #     global_fea = F.dropout(evo_fea, self.dropout, training=self.training)  # (N, 1024)
    #     global_fea = self.act_fn(self.fcs[1](global_fea))  # (N, 512)
    #
    #     # Concatenate features for classification
    #     profeas = self.act_fn(self.fcs[-1](torch.cat([global_fea, local_fea], dim=1)))  # (N, 1024) -> (N, 512)
    #     return profeas


class GraphCAI(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout):
        super(GraphCAI, self).__init__()

        self.gcn = GCN(nlayers=nlayers, nfeat=nfeat+2560, nhidden=nhidden, dropout=dropout)  # esm 2560 1152
        self.inter_model = inter_model(26203, 1280)
        self.criterion = nn.CrossEntropyLoss()

        self.projection = nn.Linear(nhidden, nhidden // 2)
        self.projection1 = nn.Linear(nhidden // 2, nfeat)
        self.projection2 = nn.Linear(nfeat, nclass)
        self.act_fn = nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
        self.predict_ec = predict_ec(nhidden)

    def forward(self, x, adj, evo_fea, inter):
        inter_feature = self.inter_model(inter)
        enz_feas = self.gcn(x.float(), adj, evo_fea, inter_feature)
        inner_layer = self.act_fn(self.projection1(self.act_fn(self.projection(enz_feas))))
        output = self.projection2(inner_layer)
        ec_output = self.predict_ec(enz_feas)
        return output, ec_output
