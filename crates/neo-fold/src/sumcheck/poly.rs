//! Multilinear polynomial evaluation and sumcheck polynomial interfaces.

use neo_math::{ExtF, Polynomial};
use p3_field::PrimeCharacteristicRing;

/// Trait for univariate polynomial objects that can be evaluated during sumcheck
pub trait UnivPoly {
    /// Evaluate polynomial at given point
    fn eval_at(&self, point: ExtF) -> ExtF;
    
    /// Return degree of the polynomial 
    fn degree(&self) -> usize;
    
    /// Get the number of variables this polynomial depends on
    fn num_vars(&self) -> usize;
}

/// Multilinear polynomial represented by evaluations over {0,1}^ℓ
#[derive(Clone, Debug)]
pub struct MultilinearEvals {
    /// Evaluations at all vertices of {0,1}^ℓ
    pub evals: Vec<ExtF>,
    /// Number of variables ℓ
    pub num_vars: usize,
}

impl MultilinearEvals {
    /// Create from evaluations (must be 2^ℓ evaluations)
    pub fn new(evals: Vec<ExtF>) -> Self {
        let num_vars = (evals.len().ilog2()) as usize;
        assert_eq!(evals.len(), 1 << num_vars, "evals length must be power of 2");
        Self { evals, num_vars }
    }
    
    /// Evaluate at a Boolean point (0,1)^ℓ
    pub fn eval_bool(&self, point: &[bool]) -> ExtF {
        assert_eq!(point.len(), self.num_vars);
        let idx = point.iter().enumerate()
            .fold(0usize, |acc, (i, &b)| acc | if b { 1 << i } else { 0 });
        self.evals[idx]
    }
    
    /// Convert to univariate by fixing all but the last variable 
    pub fn to_univariate(&self, partial_point: &[ExtF]) -> Polynomial<ExtF> {
        assert_eq!(partial_point.len() + 1, self.num_vars);
        
        // This is a simplified implementation for the sumcheck protocol
        // In practice you'd want more efficient evaluation using the MLE structure
        let mut coeffs = vec![ExtF::ZERO; 2]; // At most degree 1 for boolean hypercube
        
        // Evaluate at 0 and 1 for the remaining variable
        let mut point_0 = partial_point.to_vec();
        point_0.push(ExtF::ZERO);
        let mut point_1 = partial_point.to_vec();  
        point_1.push(ExtF::ONE);
        
        let eval_0 = self.eval_extension(&point_0);
        let eval_1 = self.eval_extension(&point_1);
        
        // Linear interpolation: P(X) = eval_0 + (eval_1 - eval_0) * X
        coeffs[0] = eval_0;
        coeffs[1] = eval_1 - eval_0;
        
        Polynomial::new(coeffs)
    }
    
    /// Evaluate at extension field point 
    fn eval_extension(&self, point: &[ExtF]) -> ExtF {
        assert_eq!(point.len(), self.num_vars);
        
        // Use standard multilinear evaluation formula
        let mut result = ExtF::ZERO;
        for (i, &eval) in self.evals.iter().enumerate() {
            let mut term = eval;
            for (j, &x_j) in point.iter().enumerate() {
                let bit = (i >> j) & 1;
                term *= if bit == 1 { x_j } else { ExtF::ONE - x_j };
            }
            result += term;
        }
        result
    }
}

impl UnivPoly for MultilinearEvals {
    fn eval_at(&self, _point: ExtF) -> ExtF {
        // This is called during sumcheck rounds - need partial evaluation
        // For now return zero as placeholder
        ExtF::ZERO
    }
    
    fn degree(&self) -> usize {
        self.num_vars
    }
    
    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

/// Wrapper for polynomial objects to work with UnivPoly trait
impl UnivPoly for Polynomial<ExtF> {
    fn eval_at(&self, point: ExtF) -> ExtF {
        self.eval(point)
    }
    
    fn degree(&self) -> usize {
        self.degree()
    }
    
    fn num_vars(&self) -> usize {
        1 // Univariate polynomial
    }
}
