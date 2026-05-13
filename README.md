# GraphCAI

GraphCAI is a graph neural network-based framework for catalytic residue prediction in enzymes. It integrates residue-level sequence features, structure-derived contact graphs, pretrained protein language model embeddings, atom-level descriptors, and InterPro-derived functional domain features to perform residue-level binary classification.

## Project Overview

The project predicts whether each residue in an enzyme sequence is catalytic or non-catalytic. Protein structures are represented as residue contact graphs, where nodes correspond to residues and edges are constructed based on spatial distance thresholds. Graph neural network modules are then used to learn structural and functional residue representations.

## Main Features

- Residue-level catalytic site prediction
- Structure-based residue contact graph construction
- Integration of PSSM, HHM, sequence, ESM, and InterPro features
- Training, validation, and independent test evaluation scripts
- Support for benchmark datasets such as NN, PC, HA-superfamily, and EF-fold
