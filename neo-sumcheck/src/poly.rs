use crate::ExtF;
use p3_field::PrimeCharacteristicRing;

/// Trait for multivariate polynomials that can be evaluated at points
/// All polynomials are defined over the extension field ExtF
pub trait UnivPoly {
    /// Evaluate the polynomial at a given point in the Boolean hypercube or extension
    fn evaluate(&self, point: &[ExtF]) -> ExtF;
    /// Return the number of variables (degree in terms of number of variables)
    fn degree(&self) -> usize;
    /// Return the maximum degree of any individual variable
    fn max_individual_degree(&self) -> usize;
}

/// Represents a multilinear polynomial through its evaluations on the Boolean hypercube
/// The evaluations are provided in lexicographic order over {0,1}^ℓ where ℓ = log₂(len)
/// Automatically pads to the next power of 2 if the input length is not a power of 2
#[derive(Clone)]
pub struct MultilinearEvals {
    /// Polynomial evaluations at all points in the Boolean hypercube, padded to 2^ℓ
    pub evals: Vec<ExtF>,
}

impl MultilinearEvals {
    /// Create a new multilinear polynomial from evaluations
    ///
    /// # Arguments
    /// * `original_evals` - Evaluations of the polynomial, will be zero-padded to next power of 2
    ///
    /// # Example
    /// For evaluations [f(0,0), f(0,1), f(1,0), f(1,1)] representing a 2-variable polynomial
    pub fn new(mut original_evals: Vec<ExtF>) -> Self {
        let len = original_evals.len();
        let ell = (len as f64).log2().ceil() as usize;
        let padded_len = 1 << ell;
        if len > padded_len {
            panic!("Input too large for multilinear extension");
        }
        original_evals.resize(padded_len, ExtF::ZERO);
        Self {
            evals: original_evals,
        }
    }
}

impl UnivPoly for MultilinearEvals {
    /// Evaluate the multilinear polynomial at an arbitrary point
    ///
    /// Uses the multilinear extension formula:
    /// f(x₁,...,xₗ) = Σᵢ f(i) · ∏ⱼ ((1-xⱼ) if iⱼ=0, xⱼ if iⱼ=1)
    /// where i ranges over all points in the Boolean hypercube {0,1}^ℓ
    fn evaluate(&self, point: &[ExtF]) -> ExtF {
        assert_eq!(point.len(), self.degree());
        let mut result = ExtF::ZERO;

        // Sum over all Boolean hypercube points
        for (idx, &val) in self.evals.iter().enumerate() {
            let mut term = val; // f(idx as binary vector)

            // Compute the Lagrange basis polynomial for this hypercube point
            for (j, &x) in point.iter().enumerate() {
                let bit_j = (idx >> j) & 1; // j-th bit of idx
                if bit_j == 1 {
                    term *= x; // If bit is 1, multiply by xⱼ
                } else {
                    term *= ExtF::ONE - x; // If bit is 0, multiply by (1-xⱼ)
                }
            }
            result += term;
        }
        result
    }

    /// Return the number of variables ℓ where the polynomial is over {0,1}^ℓ
    fn degree(&self) -> usize {
        self.evals.len().trailing_zeros() as usize
    }

    /// Multilinear polynomials have individual degree 1 in each variable
    fn max_individual_degree(&self) -> usize {
        1
    }
}

/// Blanket implementation allowing any closure to be used as a polynomial
/// This is convenient for defining polynomials inline without creating new structs
impl<FN> UnivPoly for FN
where
    FN: Fn(&[ExtF]) -> ExtF,
{
    fn evaluate(&self, point: &[ExtF]) -> ExtF {
        self(point)
    }

    /// Default degree of 0; should be overridden for specific use cases
    fn degree(&self) -> usize {
        0 // Default; override if needed
    }

    /// Default max individual degree of 0; should be overridden for specific use cases
    fn max_individual_degree(&self) -> usize {
        0 // Default; override if needed
    }
}
