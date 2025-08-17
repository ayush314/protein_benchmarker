"""
Constants for protein structure processing.
"""

import torch

# Atom ordering for PDB to OpenFold conversion
ATOM_NUMBERING = {
    "N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "OG": 5, "CG": 6, "CD1": 7,
    "CD2": 8, "CE1": 9, "CE2": 10, "CZ": 11, "OD1": 12, "ND2": 13, "CG1": 14,
    "CG2": 15, "CD": 16, "CE": 17, "NZ": 18, "OD2": 19, "OE1": 20, "NE2": 21,
    "OE2": 22, "OH": 23, "NE": 24, "NH1": 25, "NH2": 26, "OG1": 27, "SD": 28,
    "ND1": 29, "SG": 30, "NE1": 31, "CE3": 32, "CZ2": 33, "CZ3": 34, "CH2": 35,
    "OXT": 36
}

# OpenFold atom types (from openfold.np.residue_constants.atom_types)
OPENFOLD_ATOM_TYPES = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD",
    "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "NZ", "CZ", "CZ2", "CZ3", "OH", "CH2",
    "NH1", "NH2", "OXT"
]

# Conversion tensors
PDB_TO_OPENFOLD_INDEX_TENSOR = torch.tensor([
    ATOM_NUMBERING[atom] for atom in OPENFOLD_ATOM_TYPES
])

OPENFOLD_TO_PDB_INDEX_TENSOR = torch.tensor([
    OPENFOLD_ATOM_TYPES.index(atom) for atom in ATOM_NUMBERING.keys()
]) 