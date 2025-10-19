//! Core diagnostic types

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Complete diagnostic snapshot of a failing constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintDiagnostic {
    /// Schema version for evolution
    pub schema: String,
    
    /// Field configuration
    pub field: FieldConfig,
    
    /// Folding pipeline context
    pub folding: FoldingContext,
    
    /// CCS structure & failing constraint
    pub structure: StructureInfo,
    
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
    
    /// Failing row index
    pub row_index: usize,
    
    /// Sparse rows per matrix M_j (only for failing constraint)
    pub rows: Vec<SparseRow>,
    
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
    
    /// For R1CS: az, bz, cz products
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r1cs_products: Option<R1csProducts>,
    
    /// Expected value (usually 0)
    pub expected: String,
    
    /// Actual value (residual)
    pub actual: String,
    
    /// Delta in canonical form
    pub delta: String,
    
    /// Delta in signed representation (for human readability)
    pub delta_signed: String,
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
    
    /// Sparse witness (based on policy)
    pub z_sparse: SparseVector,
    
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

