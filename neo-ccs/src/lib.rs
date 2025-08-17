use neo_fields::{embed_base_to_ext, ExtF, F};
use neo_modint::ModInt;
use neo_ring::RingElement;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use std::sync::Arc;

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
struct ClosureMv<F>
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

pub mod sumcheck;
pub use sumcheck::{ccs_sumcheck_prover, ccs_sumcheck_verifier};

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

#[cfg(test)]
mod tests {
    use super::*;
    use neo_fields::from_base;
    use neo_sumcheck::{fiat_shamir::fiat_shamir_challenge, FnOracle};
    use p3_matrix::dense::RowMajorMatrix;

    #[test]
    fn test_satisfiability() {
        let _n = 2; // Num constraints
        let m = 3; // Witness size
        let a = RowMajorMatrix::new(vec![F::ONE, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO], m); // A matrix
        let b = RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ONE, F::ZERO], m); // B
        let c = RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ONE], m); // C
        let mats = vec![a, b, c];

        // f = X0 * X1 - X2
        let f = mv_poly(
            |inputs: &[ExtF]| {
                if inputs.len() != 3 {
                    ExtF::ZERO
                } else {
                    inputs[0] * inputs[1] - inputs[2]
                }
            },
            2,
        );

        let structure = CcsStructure::new(mats, f);

        // Valid witness: z = [1, 2, 2] for both rows (1*2 -2 =0, 1*2 -2=0)
        let witness = CcsWitness {
            z: vec![
                from_base(F::ONE),
                from_base(F::from_u64(2)),
                from_base(F::from_u64(2)),
            ],
        };

        let instance = CcsInstance {
            commitment: vec![], // Stub, not checked here
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };

        assert!(check_satisfiability(&structure, &instance, &witness));

        // Invalid: change to [1,2,3]
        let bad_witness = CcsWitness {
            z: vec![
                from_base(F::ONE),
                from_base(F::from_u64(2)),
                from_base(F::from_u64(3)),
            ],
        };
        assert!(!check_satisfiability(&structure, &instance, &bad_witness));
    }

    #[test]
    fn test_high_deg_f() {
        let m = 1;
        let mat = RowMajorMatrix::new(vec![F::ONE], m);
        let mats = vec![mat];
        let f = mv_poly(
            |inputs: &[ExtF]| {
                if inputs.len() == 1 {
                    inputs[0] * inputs[0] * inputs[0]
                } else {
                    ExtF::ZERO
                }
            },
            3,
        );
        let structure = CcsStructure::new(mats, f);
        let instance = CcsInstance {
            commitment: vec![],
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };
        let witness = CcsWitness {
            z: vec![from_base(F::from_u64(2))],
        };
        let mut transcript = vec![];
        let mut oracle = FnOracle::new(|_: &[ExtF]| vec![]);
        let (msgs, _) = ccs_sumcheck_prover(
            &structure,
            &instance,
            &witness,
            1,
            &mut oracle,
            &mut transcript,
        )
        .expect("sumcheck");
        let mut current = from_base(F::from_u64(8));
        for (uni, _) in &msgs {
            current = uni.eval(fiat_shamir_challenge(&mut vec![]));
        }
        assert_eq!(current, from_base(F::from_u64(8)));
    }

    #[test]
    fn test_public_inputs() {
        let m = 1;
        let mat = RowMajorMatrix::new(vec![F::ZERO], m);
        let mats = vec![mat];
        let f = mv_poly(|inputs: &[ExtF]| inputs[0], 1);
        let structure = CcsStructure::new(mats, f);
        let instance = CcsInstance {
            commitment: vec![],
            public_input: vec![F::ONE],
            u: F::ZERO,
            e: F::ONE,
        };
        let witness = CcsWitness { z: vec![] };
        let mut transcript = vec![];
        let mut oracle = FnOracle::new(|_: &[ExtF]| vec![]);
        let (msgs, _) = ccs_sumcheck_prover(
            &structure,
            &instance,
            &witness,
            1,
            &mut oracle,
            &mut transcript,
        )
        .expect("sumcheck");
        let mut current = from_base(F::ONE);
        for (uni, _) in &msgs {
            current = uni.eval(fiat_shamir_challenge(&mut vec![]));
        }
        assert_eq!(current, from_base(F::ONE));
    }
}
