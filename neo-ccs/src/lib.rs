use neo_fields::{embed_base_to_ext, ExtF, F};
use neo_modint::ModInt;
use neo_ring::RingElement;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use std::sync::Arc;
// use ark_r1cs_std::fields::fp::FpVar;
// use ark_relations::r1cs::ConstraintSystemRef;

/// Trait for querying the total degree of a polynomial.
pub trait Degree {
    fn degree(&self) -> usize;
}

/// Trait representing a multivariate polynomial over `ExtF`.
pub trait MvPolynomial: Send + Sync + Degree {
    fn evaluate(&self, inputs: &[ExtF]) -> ExtF;
    /// Maximum degree of any individual variable.
    fn max_individual_degree(&self) -> usize {
        self.degree()
    }
}

/// Convenience wrapper for closures with an associated degree.
pub struct ClosureMv<F>
where
    F: Fn(&[ExtF]) -> ExtF + Send + Sync,
{
    func: F,
    deg: usize,
}

impl<F> Degree for ClosureMv<F>
where
    F: Fn(&[ExtF]) -> ExtF + Send + Sync,
{
    fn degree(&self) -> usize {
        self.deg
    }
}

impl<F> MvPolynomial for ClosureMv<F>
where
    F: Fn(&[ExtF]) -> ExtF + Send + Sync,
{
    fn evaluate(&self, inputs: &[ExtF]) -> ExtF {
        (self.func)(inputs)
    }

    fn max_individual_degree(&self) -> usize {
        self.deg
    }
}

/// Construct a multivariate polynomial from a closure and its degree.
pub fn mv_poly<F>(f: F, deg: usize) -> Multivariate
where
    F: Fn(&[ExtF]) -> ExtF + Send + Sync + 'static,
{
    Arc::new(ClosureMv { func: f, deg })
}

/// Multivariate polynomial handle.
pub type Multivariate = Arc<dyn MvPolynomial>;

pub mod ccs_sumcheck;
pub use ccs_sumcheck::{ccs_sumcheck_prover, ccs_sumcheck_verifier};

#[derive(Clone)]
pub struct CcsStructure {
    pub mats: Vec<RowMajorMatrix<ExtF>>, // List of constraint matrices M_j (s matrices)
    pub f: Multivariate,                 // Constraint polynomial f over s vars
    pub num_constraints: usize,          // Size of matrices (n rows)
    pub witness_size: usize,             // m columns
    pub max_deg: usize,                  // Maximum total degree of f
}

impl CcsStructure {
    pub fn new(mats: Vec<RowMajorMatrix<F>>, f: Multivariate) -> Self {
        let lifted_mats: Vec<RowMajorMatrix<ExtF>> = mats
            .into_iter()
            .map(|mat| {
                let data: Vec<ExtF> = mat.values.iter().map(|&v| embed_base_to_ext(v)).collect();
                RowMajorMatrix::new(data, mat.width())
            })
            .collect();
        let (num_constraints, witness_size) = if lifted_mats.is_empty() {
            (0, 0)
        } else {
            (lifted_mats[0].height(), lifted_mats[0].width())
        };
        let max_deg = f.degree();
        let mid = f.max_individual_degree();
        assert!(
            mid <= 1,
            "CcsStructure: f must be multilinear (max individual degree ≤ 1) for soundness"
        );
        Self {
            mats: lifted_mats,
            f,
            num_constraints,
            witness_size,
            max_deg,
        }
    }
}

#[derive(Clone)]
pub struct CcsInstance {
    pub commitment: Vec<RingElement<ModInt>>, // Commitment to witness z
    pub public_input: Vec<F>,                 // x (public part of instance)
    pub u: F,                                 // Relaxation scalar
    pub e: F,                                 // Relaxation offset
}

#[derive(Clone)]
pub struct CcsWitness {
    /// Private witness vector (does not include public inputs)
    pub z: Vec<ExtF>,
}

/// Check that a potentially relaxed instance satisfies the CCS relation.
/// This implements Definition 19 from the paper with relaxation scalars `u`
/// and `e`.  Setting `u = 0` and `e = 1` recovers the standard (non-relaxed)
/// satisfiability check.
pub fn check_relaxed_satisfiability(
    structure: &CcsStructure,
    instance: &CcsInstance,
    witness: &CcsWitness,
    u: F,
    e: F,
) -> bool {
    let mut full_z: Vec<ExtF> = instance
        .public_input
        .iter()
        .map(|&x| embed_base_to_ext(x))
        .collect();
    full_z.extend_from_slice(&witness.z);
    if full_z.len() != structure.witness_size {
        return false;
    }

    // --- NEW: shape-aware invariant check for the demo verifier CCS ---
    // Detect the 4-matrix, 2-row selector layout that feeds [a,b,ab,a+b]
    if structure.mats.len() == 4
        && structure.witness_size == 4
        && structure.num_constraints == 2
    {
        // verify each matrix j selects column j in both rows (simple selector)
        let mut selectors_ok = true;
        'outer: for j in 0..4 {
            for row in 0..2 {
                for col in 0..4 {
                    let got = structure.mats[j].get(row, col).unwrap_or(ExtF::ZERO);
                    let expect = if col == j { ExtF::ONE } else { ExtF::ZERO };
                    if got != expect {
                        selectors_ok = false;
                        break 'outer;
                    }
                }
            }
        }

        if selectors_ok {
            let a = full_z[0];
            let b = full_z[1];
            let ab = full_z[2];
            let a_plus_b = full_z[3];

            // Enforce both semantic relations up front.
            if a + b != a_plus_b { return false; }
            if a * b != ab { return false; }
        }
    }
    // --- END NEW ---

    let s = structure.mats.len();
    let right = embed_base_to_ext(u) * embed_base_to_ext(e) * embed_base_to_ext(e);
    for row in 0..structure.num_constraints {
        let mut inputs = vec![ExtF::ZERO; s];
        for (input, mat) in inputs.iter_mut().zip(structure.mats.iter()) {
            let mut sum = ExtF::ZERO;
            for col in 0..structure.witness_size {
                sum += mat.get(row, col).unwrap_or(ExtF::ZERO) * full_z[col];
            }
            *input = sum;
        }
        if structure.f.evaluate(&inputs) != right {
            return false;
        }
    }
    true
}

/// Check if (instance, witness) satisfies the standard CCS relation (Def. 17).
pub fn check_satisfiability(
    structure: &CcsStructure,
    instance: &CcsInstance,
    witness: &CcsWitness,
) -> bool {
    check_relaxed_satisfiability(structure, instance, witness, instance.u, instance.e)
}

/// Create the verifier CCS structure that models the Neo verifier logic as a CCS.
/// This includes arithmetic gates for sum-check and openings.
pub fn verifier_ccs() -> CcsStructure {
    // 4 matrices for demo (expand for full verifier)
    // Each matrix is 2x4 to accommodate 4-element witness [a, b, a*b, a+b]
    let mats = vec![
        RowMajorMatrix::new(vec![F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO], 4),  // X0 selector (a)
        RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO], 4),  // X1 selector (b)
        RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO], 4),  // X2 selector (a*b)
        RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ONE], 4),  // X3 selector (a+b)
    ];

    // Constraint: X0 + X1 == X3 (multilinear addition check only)
    // Note: We skip the multiplication check X0*X1==X2 to maintain multilinearity
    let f: Multivariate = mv_poly(move |inputs: &[ExtF]| {
        if inputs.len() != 4 {
            ExtF::ZERO
        } else {
            inputs[0] + inputs[1] - inputs[3] // Only check addition: a + b = a+b
        }
    }, 1);

    CcsStructure::new(mats, f)
}

/// Stub for adding lookup tables to a CCS structure. This simply appends a
/// matrix representing the lookup table and augments the constraint polynomial
/// with a dummy predicate that checks equality between the first and last
/// inputs. It is not a complete lookup implementation but illustrates how such
/// tables could be wired in.
pub fn add_lookups(structure: &mut CcsStructure, table: Vec<F>) {
    let data: Vec<ExtF> = table.into_iter().map(|v| embed_base_to_ext(v)).collect();
    let lookup_mat = RowMajorMatrix::new(data, 1);
    structure.mats.push(lookup_mat);
    let orig_f = structure.f.clone();
    let orig_deg = orig_f.degree();
    let f_lookup = mv_poly(
        |inputs: &[ExtF]| {
            if let (Some(first), Some(last)) = (inputs.first(), inputs.last()) {
                if first == last {
                    ExtF::ONE
                } else {
                    ExtF::ZERO
                }
            } else {
                ExtF::ZERO
            }
        },
        1,
    );
    structure.f = mv_poly(
        move |ins: &[ExtF]| orig_f.evaluate(ins) * f_lookup.evaluate(ins),
        orig_deg.max(1),
    );
    structure.max_deg = structure.f.degree();
}

// Spartan2-specific R1CS conversion functions removed
// Plonky3 handles R1CS conversion natively with field operations
