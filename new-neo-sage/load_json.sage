"""
JSON loading utilities for Neo protocol data structures.

This module provides functions to load:
- NeoParams: protocol parameters
- CcsStructure: Customizable Constraint System
- McsInstance/McsWitness: Module Committed Statement
- MeInstance: Module Error (folded output)
"""

import json

def load_field(F):
    """
    Helper to convert JSON field element to Sage field element.
    Assumes field elements are stored as u64 strings.
    """
    def convert(s):
        return F(int(s))
    return convert

def load_vector(json_vec, convert_fn):
    """Load a vector from JSON using conversion function."""
    return [convert_fn(x) for x in json_vec]

def load_matrix(json_mat, convert_fn):
    """
    Load a matrix from JSON.
    Expected format: {"rows": n, "cols": m, "data": [...]}
    Data is row-major order.
    """
    rows = json_mat["rows"]
    cols = json_mat["cols"]
    data = [convert_fn(x) for x in json_mat["data"]]
    
    mat = Matrix(rows, cols)
    for r in range(rows):
        for c in range(cols):
            mat[r, c] = data[r * cols + c]
    return mat

def load_neo_params(json_data):
    """
    Load NeoParams from JSON.
    
    Returns: dict with keys:
        - n: lattice dimension
        - kappa: module rank
        - b: decomposition base
        - d: decomposition depth
        - q: field modulus
    """
    return {
        'n': json_data['n'],
        'kappa': json_data['kappa'],
        'b': json_data['b'],
        'd': json_data['d'],
        'q': int(json_data['q'])
    }

def load_ccs_structure(json_data, F):
    """
    Load CcsStructure from JSON.
    
    Args:
        json_data: JSON object with CCS structure
        F: Sage field
        
    Returns: dict with keys:
        - matrices: list of matrices M_j
        - f: sparse polynomial (as dict of {monomial: coefficient})
        - n: number of rows
        - m: number of columns
    """
    convert = load_field(F)
    
    matrices = [load_matrix(m, convert) for m in json_data['matrices']]
    
    f = {}
    for term in json_data['f']:
        monomial = tuple(term['monomial'])
        coeff = convert(term['coeff'])
        f[monomial] = coeff
    
    return {
        'matrices': matrices,
        'f': f,
        'n': json_data['n'],
        'm': json_data['m']
    }

def load_mcs_instance(json_data, F):
    """
    Load McsInstance from JSON.
    
    Args:
        json_data: JSON object with MCS instance
        F: Sage field
        
    Returns: dict with keys:
        - c: commitment (matrix d×κ)
        - x: public inputs (vector)
        - m_in: public input length
    """
    convert = load_field(F)
    
    return {
        'c': load_matrix(json_data['c'], convert),
        'x': load_vector(json_data['x'], convert),
        'm_in': json_data['m_in']
    }

def load_mcs_witness(json_data, F):
    """
    Load McsWitness from JSON.
    
    Args:
        json_data: JSON object with MCS witness
        F: Sage field
        
    Returns: dict with keys:
        - w: private witness (vector)
        - Z: decomposition matrix (d×m)
    """
    convert = load_field(F)
    
    return {
        'w': load_vector(json_data['w'], convert),
        'Z': load_matrix(json_data['Z'], convert)
    }

def load_me_instance(json_data, F, K):
    """
    Load MeInstance from JSON.
    
    Args:
        json_data: JSON object with ME instance
        F: Sage base field
        K: Sage extension field (for sumcheck)
        
    Returns: dict with keys:
        - c: commitment (matrix d×κ)
        - X: projected public inputs (matrix d×m_in)
        - r: random point (vector in K)
        - y: list of y_j vectors (each is a vector in K)
        - y_at_r: Y_j(r) scalars in K
        - m_in: public input length
    """
    convert_f = load_field(F)
    convert_k = load_field(K)
    
    return {
        'c': load_matrix(json_data['c'], convert_f),
        'X': load_matrix(json_data['X'], convert_f),
        'r': load_vector(json_data['r'], convert_k),
        'y': [load_vector(y_j, convert_k) for y_j in json_data['y']],
        'y_at_r': load_vector(json_data['y_at_r'], convert_k),
        'm_in': json_data['m_in']
    }

def load_pi_ccs_input(filepath):
    """
    Load complete π-CCS folding input from JSON file.
    
    Args:
        filepath: path to JSON file
        
    Returns: dict with keys:
        - params: NeoParams
        - ccs: CcsStructure
        - mcs_instances: list of MCS instances
        - mcs_witnesses: list of MCS witnesses
        - me_instances: list of ME instances
        - me_witnesses: list of witness matrices Z
        - field: Sage field F
        - ext_field: Sage extension field K
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    q = int(data['field']['q'])
    F = GF(q)
    
    ext_deg = data['field']['ext_deg']
    if ext_deg == 1:
        K = F
    else:
        raise NotImplementedError(f"Extension fields not yet supported (degree {ext_deg})")
    
    params = load_neo_params(data['params'])
    ccs = load_ccs_structure(data['ccs'], F)
    
    mcs_instances = [load_mcs_instance(mcs, F) for mcs in data['mcs_instances']]
    mcs_witnesses = [load_mcs_witness(mcs, F) for mcs in data['mcs_witnesses']]
    
    me_instances = [load_me_instance(me, F, K) for me in data['me_instances']]
    me_witnesses = [load_matrix(z, load_field(F)) for z in data['me_witnesses']]
    
    return {
        'params': params,
        'ccs': ccs,
        'mcs_instances': mcs_instances,
        'mcs_witnesses': mcs_witnesses,
        'me_instances': me_instances,
        'me_witnesses': me_witnesses,
        'field': F,
        'ext_field': K
    }

