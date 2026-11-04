use p3_field::Field;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SparsePoly<F> {
    t: usize,
    terms: Vec<Term<F>>,
}

impl<F> SparsePoly<F> {
    pub fn new(t: usize, terms: Vec<Term<F>>) -> Self {
        Self { t, terms }
    }

    pub fn arity(&self) -> usize {
        self.t
    }

    pub fn terms(&self) -> &[Term<F>] {
        &self.terms
    }

    pub fn max_degree(&self) -> u32 {
        self.terms
            .iter()
            .map(|term| term.exps.iter().copied().fold(0u32, u32::saturating_add))
            .max()
            .unwrap_or(0)
    }

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
            new_terms.push(Term {
                coeff: term.coeff.clone(),
                exps,
            });
        }
        SparsePoly {
            t: new_t,
            terms: new_terms,
        }
    }

    pub fn append_zero_vars(&self, extra: usize) -> Self
    where
        F: Clone,
    {
        if extra == 0 {
            return self.clone();
        }
        let new_t = self.t + extra;
        let mut new_terms = Vec::with_capacity(self.terms.len());
        for term in &self.terms {
            let mut exps = term.exps.clone();
            exps.extend(std::iter::repeat_n(0u32, extra));
            new_terms.push(Term {
                coeff: term.coeff.clone(),
                exps,
            });
        }
        SparsePoly {
            t: new_t,
            terms: new_terms,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Term<F> {
    pub coeff: F,
    pub exps: Vec<u32>,
}

impl<F: Field> SparsePoly<F> {
    pub fn eval(&self, x: &[F]) -> F {
        assert_eq!(x.len(), self.t);
        let mut acc = F::ZERO;
        for term in &self.terms {
            debug_assert_eq!(term.exps.len(), self.t);
            let mut m = term.coeff;
            for (xi, &pow) in x.iter().zip(term.exps.iter()) {
                if pow == 0 {
                    continue;
                }
                let mut p = *xi;
                let mut e = pow - 1;
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

    pub fn eval_in_ext<K: Field + From<F>>(&self, x: &[K]) -> K {
        assert_eq!(x.len(), self.t);
        let mut acc = K::ZERO;
        for term in &self.terms {
            let mut m = K::from(term.coeff);
            for (xi, &pow) in x.iter().zip(term.exps.iter()) {
                if pow == 0 {
                    continue;
                }
                let mut p = *xi;
                for _ in 1..pow {
                    p *= *xi;
                }
                m *= p;
            }
            acc += m;
        }
        acc
    }
}
