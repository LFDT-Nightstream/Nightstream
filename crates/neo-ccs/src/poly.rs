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

    /// Return a new polynomial with one dummy variable inserted at the front (index 0).
    /// All terms are updated by shifting exponents to the right and inserting 0 at position 0.
    pub fn insert_var_at_front(&self) -> Self
    where
        F: Clone,
    {
        let new_t = self.t + 1;
        let mut new_terms = Vec::with_capacity(self.terms.len());
        for term in &self.terms {
            let mut exps = Vec::with_capacity(new_t);
            exps.push(0u32);
            exps.extend(term.exps.iter().copied());
            new_terms.push(Term { coeff: term.coeff.clone(), exps });
        }
        SparsePoly { t: new_t, terms: new_terms }
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
    
    /// Evaluate with inputs in an extension field K (coeffs in base F).
    pub fn eval_in_ext<K: Field + From<F>>(&self, x: &[K]) -> K {
        assert_eq!(x.len(), self.t);
        let mut acc = K::ZERO;
        for term in &self.terms {
            let mut m = K::from(term.coeff);
            for (xi, &pow) in x.iter().zip(term.exps.iter()) {
                if pow == 0 { continue; }
                let mut p = *xi;
                for _ in 1..pow { p *= *xi; } // small exponents in CCS polynomials
                m *= p;
            }
            acc += m;
        }
        acc
    }
}
