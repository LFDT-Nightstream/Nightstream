use p3_field::Field;

/// A sparse multivariate polynomial over `F` with `t` indeterminates.
///
/// Each `Term` stores a coefficient `coeff` and per-variable exponents `exps[j]` for j∈[0..t).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SparsePoly<F> {
    t: usize,
    terms: Vec<Term<F>>,
}

impl<F> SparsePoly<F> {
    /// Construct with explicit arity and terms.
    pub fn new(t: usize, terms: Vec<Term<F>>) -> Self { Self { t, terms } }

    /// Number of variables.
    pub fn arity(&self) -> usize { self.t }

    /// Terms.
    pub fn terms(&self) -> &[Term<F>] { &self.terms }
    
    /// Maximum degree of the polynomial (highest sum of exponents in any term).
    pub fn max_degree(&self) -> u32 {
        self.terms.iter()
            .map(|term| term.exps.iter().sum::<u32>())
            .max()
            .unwrap_or(0)
    }
}

/// A single term: `coeff * ∏_j x_j^{exps[j]}`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Term<F> {
    /// Coefficient
    pub coeff: F,
    /// Exponents for each variable
    pub exps: Vec<u32>,
}

impl<F: Field> SparsePoly<F> {
    /// Evaluate at a point `x` of length `t`. Panics if lengths mismatch.
    pub fn eval(&self, x: &[F]) -> F {
        assert_eq!(x.len(), self.t);
        let mut acc = F::ZERO;
        for term in &self.terms {
            debug_assert_eq!(term.exps.len(), self.t);
            let mut m = term.coeff;
            for (xi, &pow) in x.iter().zip(term.exps.iter()) {
                if pow == 0 { continue; }
                let mut p = *xi;
                let mut e = pow - 1;
                // fast-ish pow for small exponents
                while e > 0 {
                    p *= *xi;
                    e -= 1;
                }
                m *= p;
            }
            acc += m;
        }
        acc
    }
}
