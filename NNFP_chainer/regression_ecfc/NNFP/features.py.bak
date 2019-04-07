#coding : utf-8
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from util import one_of_k_encoding, one_of_k_encoding_unk

def atom_features_from_ecfp(atom):
    #print atom
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                       'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                       'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
                                       'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                       'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()]
					)


def atom_get(x, allowable_set):
	"""Maps inputs not in the allowable set to the last element."""
	if x not in allowable_set:
		return -1
	
	
	for i in range(len(allowable_set)):
		if allowable_set[i] == x: 
			return i

def get_degree(x):
	return int(x)

def get_totalnumHs(x):
	return int(x)

def get_implicitValamce(x):
	return int(x)

def get_isaromatic(x):
	return 1 if x else 0


'''

def atom_features_from_ecfp(atom):
    return   np.array([atom_get(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                       'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                       'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
                                       'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                       'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']),
                    get_degree(atom.GetDegree()),
                    get_totalnumHs(atom.GetTotalNumHs()),
                    get_implicitValamce(atom.GetImplicitValence()),
                    get_isaromatic(atom.GetIsAromatic())]
					)
'''

def atom_features_from_fcfp(mol):
	com = AllChem.RemoveHs(mol) 
	gl = AllChem.GetFeatureInvariants(com)
	def to_bin(x):
		ff = (map(int, list(format(x, 'b').zfill(6)))) #FCFP has 6 features
		return ff
	gl = map(to_bin, gl)
	return np.array(gl)

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])

def num_atom_features_from_ecfp():
	# Return length of feature vector using a very simple molecule.
	m = Chem.MolFromSmiles('CC')
	alist = m.GetAtoms()
	a = alist[0]
	return len(atom_features_from_ecfp(a))

def num_atom_features_from_fcfp():
	# Return length of feature vector using a very simple molecule.
	a = Chem.MolFromSmiles('CC')
	return len(atom_features_from_fcfp(a)[0])


def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))

