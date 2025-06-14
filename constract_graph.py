from rdkit import Chem
from typing import List

def get_atom_featues(atom:Chem.atom):
    possible_atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Unknown']

    atom_type_one_hot = [0.0] * len(possible_atom_types)

    try:
        atom_type_one_hot[possible_atom_types.index(atom.GetSymbol())] = 1.0
    except ValueError:
        atom_type_one_hot[-1] = 1.0

    hybridizations =  [
        Chem.rdchem.HybridizationType.SP, 
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, 
        Chem.rdchem.HybridizationType.SP3D, 
        Chem.rdchem.HybridizationType.SP3D2,
        'Unknown'
    ]
    hybridizations_as_one_hot = [0.0]*len(hybridizations)
    try:
        hybridizations_as_one_hot[hybridizations.index(atom.GetHybridization())] = 1.0
    except ValueError:
        hybridizations_as_one_hot[-1] = 1.0

    formal_charge = float(atom.GetFormalCharge())

    is_aromatic = float(atom.GetIsAromatic())

    degree = float(atom.GetDegree())

    num_hydrogens = float(atom.GetTotalNumHs())

    is_chiral = float(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED)

    features = atom_type_one_hot + [formal_charge, is_aromatic, degree, num_hydrogens, is_chiral] + hybridization_one_hot

    return features


def get_bond_features(bond:Chem.Bond):
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
        'Unknown'
    ]
    bond_types_one_hot = [0.0] * len(bond_types)
    try:
        bond_types_one_hot[bond_types.index(bond.GetBondType())] = 1.0
    except ValueError:
        bond_types_one_hot[-1] = 1
    is_conjugated = float(bond.GetIsConjugated())
    is_in_ring = float(bond.IsInRing())
    features = bond_types_one_hot + [is_conjugated,is_in_ring]
    return features



def smiles_to_graph_data(smiles_string,y_label):
    mol = Chem.MolFromSmiles(smiles_string)

    if mol is None:
        return None

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(get_atom_featues(atom))
    x = torch.tensor(np.array(atom_features_list),dtype=torch.float)

    edge_index = []
    edge_features_list = []

    if mol.GetNumBonds() > 0:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_index.append((i,j))
            edge_index.append((j,i))

            bon_features = get_bond_features(bond)
            edge_features_list.append(bon_features)
            edge_features_list.append(bon_features)
        edge_index = torch.tensor(np.array(edge_indices).T, dtype=torch.long) # transpose so when we pass it geometric lib it will work
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.float)

    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float)
    y = torch.tensor([y_label], dtype=torch.float) # Shape: [1, num_labels]
    return Data(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y)

