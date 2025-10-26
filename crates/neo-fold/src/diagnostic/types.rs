//! Core diagnostic types

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Complete diagnostic snapshot of a failing constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintDiagnostic {
    /// Schema version for evolution
    pub schema: String,
    
    /// Redaction salt for secure witness hashing (32 bytes hex)
    /// Used with keyed BLAKE3 to prevent preimage attacks
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redaction_salt: Option<String>,
    
    /// Field configuration
    pub field: FieldConfig,
    
    /// Folding pipeline context
    pub folding: FoldingContext,
    
    /// CCS structure & failing constraint
    pub structure: StructureInfo,
    
    /// Instance information (commitment, public inputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance: Option<InstanceInfo>,
    
    /// Evaluation at failure point
    pub eval: EvaluationInfo,
    
    /// Witness data (sparse, policy-controlled)
    pub witness: WitnessInfo,
    
    /// Transcript challenges for reproducibility
    pub transcript: TranscriptInfo,
    
    /// Test/step context
    pub context: ContextInfo,
    
    /// Optional symbol table for human-readable debugging
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbols: Option<SymbolTable>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldConfig {
    /// Modulus as decimal string (works for arbitrary size)
    pub q: String,
    
    /// Field name
    pub name: String,
    
    /// Extension degree (1 = base field)
    pub ext_deg: u32,
    
    /// Encoding format (always canonical for stability)
    pub encoding: String,
    
    /// For extension fields: generator polynomial coefficients
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ext_modulus: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldingContext {
    /// Which reduction phase
    pub phase: String,  // "PiCCS" | "PiRLC" | "PiDEC" | "PreFold"
    
    /// Decomposition base
    pub b: u64,
    
    /// Decomposition depth
    pub k: usize,
    
    /// Norm bound B = b^k
    #[serde(rename = "B")]
    pub norm_bound: u64,
    
    /// Strong sampling set expansion factor
    #[serde(rename = "T")]
    pub expansion_factor: u64,
    
    /// Challenge set identifier (for versioning)
    pub challenge_set: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureInfo {
    /// Stable canonical hash of CCS structure
    pub hash: String,
    
    /// Number of matrices (t)
    pub t: usize,
    
    /// Number of constraints (n)
    pub n: usize,
    
    /// Number of variables (m)
    pub m: usize,
    
    /// Length of public input vector (m_in)
    /// z = x || w where |x| = m_in, |w| = m - m_in
    pub m_in: usize,
    
    /// Failing row index
    pub row_index: usize,
    
    /// Sparse rows per matrix M_j for the failing constraint
    pub rows: Vec<SparseRow>,
    
    /// FULL STORAGE: Complete matrices (all n rows for all t matrices)
    pub full_matrices: Vec<FullMatrix>,
    
    /// Canonical polynomial representation
    pub f_canonical: PolyCanonical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseRow {
    /// Matrix index (j)
    pub matrix_j: usize,
    
    /// Non-zero column indices
    pub idx: Vec<usize>,
    
    /// Values at those indices (canonical LE bytes, hex-encoded)
    pub val: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolyCanonical {
    /// CCS polynomial as canonical terms
    pub terms: Vec<PolyTerm>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolyTerm {
    /// Coefficient (canonical LE bytes, hex-encoded)
    pub coeff: String,
    
    /// Variable indices with exponents: (var_index, exponent)
    /// This supports non-multilinear polynomials (e.g., y_0^2 * y_1)
    pub vars: Vec<(usize, u32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationInfo {
    /// Sum-check evaluation point r ∈ K^{log n}
    pub r: Vec<String>,
    
    /// Partial evaluations y_j = <M_j z, χ_r>
    pub y_at_r: Vec<String>,
    
    /// FULL STORAGE: Matrix-vector products for failing row
    /// y_j = M_j[row] · z for each matrix j
    pub matrix_products: Vec<MatrixProduct>,
    
    /// For R1CS: az, bz, cz products
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r1cs_products: Option<R1csProducts>,
    
    /// FULL STORAGE: All constraint evaluations (not just failing one)
    pub all_constraints: Vec<ConstraintEval>,
    
    /// Expected value (usually 0)
    pub expected: String,
    
    /// Actual value (residual)
    pub actual: String,
    
    /// Delta in canonical form
    pub delta: String,
    
    /// Delta in signed representation (for human readability)
    pub delta_signed: String,
}

/// Matrix-vector product for one matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixProduct {
    /// Matrix index j
    pub matrix_j: usize,
    
    /// Product value: M_j[row] · z
    pub value: String,
    
    /// Contributing terms: [(col_idx, mat_val, z_val, product)]
    pub terms: Vec<ProductTerm>,
}

/// Individual term in matrix-vector product
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductTerm {
    pub col_idx: usize,
    pub matrix_value: String,
    pub witness_value: String,
    pub product: String,
}

/// Evaluation of one constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintEval {
    /// Constraint row index
    pub row: usize,
    
    /// Whether this constraint passed
    pub passed: bool,
    
    /// Evaluation result
    pub value: String,
    
    /// Matrix products for this row
    pub y_values: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct R1csProducts {
    pub az: String,
    pub bz: String,
    pub cz: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessInfo {
    /// Witness disclosure policy used
    pub policy: super::witness::WitnessPolicy,
    
    /// FULL STORAGE: Complete combined witness z = x || w (all m elements)
    pub z_full: Vec<String>,
    
    /// Public inputs x (first m_in elements of z)
    pub x_public: Vec<String>,
    
    /// Private witness w (remaining m - m_in elements)
    pub w_private: Vec<String>,
    
    /// Decomposition matrix Z ∈ F^{d×m} (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z_decomposition: Option<DecompositionMatrix>,
    
    /// Gradient-based blame analysis (mathematically correct)
    pub gradient_blame: Vec<GradientContribution>,
    
    /// Full witness size
    pub size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    /// Indices
    pub idx: Vec<usize>,
    
    /// Values (canonical LE bytes, hex-encoded)
    pub val: Vec<String>,
}

/// Mathematically correct sensitivity analysis
/// For CCS: ∂r/∂z_i = Σ_j (∂f/∂y_j) * M_j[row, i]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientContribution {
    /// Witness index
    pub i: usize,
    
    /// Human-readable term (if symbol table available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    
    /// Partial derivative ∂r/∂z_i (canonical form)
    pub gradient: String,
    
    /// Absolute gradient (for ranking)
    pub gradient_abs: String,
    
    /// Current witness value z_i
    pub z_value: String,
    
    /// Product: (∂r/∂z_i) * z_i (contribution to residual)
    pub contribution: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptInfo {
    /// Hash of transcript state BEFORE sampling these challenges
    pub transcript_hash: String,
    
    /// Domain separators used (for reproducibility)
    pub domain_labels: Vec<String>,
    
    /// Sum-check challenges
    pub sumcheck: SumcheckChallenges,
    
    /// RLC challenges (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rlc: Option<RlcChallenges>,
    
    /// Decomposition challenges (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dec: Option<DecChallenges>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SumcheckChallenges {
    /// α ∈ K^{log d}
    pub alpha: Vec<String>,
    
    /// β ∈ K^{log(dn)}
    pub beta: Vec<String>,
    
    /// γ ∈ K
    pub gamma: String,
    
    /// Final evaluation point α' ∈ F^{log d}
    pub alpha_prime: Vec<String>,
    
    /// Final evaluation point r' ∈ F^{log n}
    pub r_prime: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlcChallenges {
    /// ρ_i matrices for i ∈ [k+1] (flattened, row-major)
    pub rho: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecChallenges {
    /// Decomposition-specific randomness
    pub params: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextInfo {
    /// Step index in IVC/NIVC chain
    pub step_idx: usize,
    
    /// Instruction or operation name
    pub instruction: String,
    
    /// Test name
    pub test: String,
    
    /// Human-readable failure reason
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
}

/// Full matrix storage (complete circuit reconstruction)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullMatrix {
    /// Matrix index (j)
    pub matrix_j: usize,
    
    /// Dimensions (n × m)
    pub shape: (usize, usize),
    
    /// All rows as sparse vectors
    pub rows: Vec<SparseRow>,
    
    /// Matrix density (percentage of non-zero entries)
    pub density: f64,
    
    /// Total non-zero entries
    pub nnz: usize,
}

/// Decomposition matrix Z ∈ F^{d×m}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionMatrix {
    /// Dimensions (d × m)
    pub shape: (usize, usize),
    
    /// Decomposition base
    pub base_b: u64,
    
    /// Rows of Z (each row is z decomposed in base b)
    pub rows: Vec<Vec<String>>,
}

/// Symbol table: maps witness indices to human-readable names
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolTable {
    pub symbols: HashMap<usize, String>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            symbols: HashMap::new(),
        }
    }
    
    pub fn add(&mut self, idx: usize, name: String) {
        self.symbols.insert(idx, name);
    }
    
    pub fn get(&self, idx: usize) -> Option<&String> {
        self.symbols.get(&idx)
    }
}

/// Instance information (commitment and public inputs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceInfo {
    /// Commitment to witness (Ajtai commitment coordinates)
    /// c = L(Z) where Z is the decomposition of z
    pub commitment: Vec<String>,
    
    /// Commitment dimensions (d × κ)
    pub commitment_shape: (usize, usize),
    
    /// Decomposition parameters
    pub base_b: u64,
    pub depth_d: usize,
    
    /// Public input length
    pub m_in: usize,
    
    /// Ajtai lattice parameters (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lattice_params: Option<LatticeParams>,
}

/// Ajtai lattice parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeParams {
    /// Lattice dimension
    pub n: usize,
    
    /// Module rank
    pub kappa: usize,
    
    /// Modulus q
    pub q: String,
    
    /// Norm bound B = b^d
    pub norm_bound: String,
}

