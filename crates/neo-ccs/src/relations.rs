use p3_field::{Field, PrimeCharacteristicRing};

use crate::{
    error::{CcsError, RelationError},
    matrix::Mat,
    poly::SparsePoly,
    sparse::{CcsMatrix, CscMat},
};
use neo_math::D;

/// A validated CCS structure. The fields are private, so a structure keeps
/// the shape that `new`, `new_sparse` or `new_verifier_artifact_header`
/// checked.
#[derive(Clone, Debug)]
pub struct CcsStructure<F> {
    matrices: Vec<CcsMatrix<F>>,
    f: SparsePoly<F>,
    n: usize,
    m: usize,
    domain_variables: Option<usize>,
}

impl<F: Field> CcsStructure<F> {
    pub fn matrices(&self) -> &[CcsMatrix<F>] {
        &self.matrices
    }

    pub fn f(&self) -> &SparsePoly<F> {
        &self.f
    }

    /// The constraint rows.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The witness columns.
    pub fn m(&self) -> usize {
        self.m
    }

    pub fn domain_variables(&self) -> Option<usize> {
        self.domain_variables
    }

    pub fn new(matrices: Vec<Mat<F>>, f: SparsePoly<F>) -> Result<Self, RelationError>
    where
        F: p3_field::PrimeCharacteristicRing + Copy + Eq + Send + Sync,
    {
        if matrices.is_empty() {
            return Err(RelationError::InvalidStructure);
        }
        let n = matrices[0].rows();
        let m = matrices[0].cols();
        for mj in matrices.iter() {
            if mj.rows() != n || mj.cols() != m {
                return Err(RelationError::InvalidStructure);
            }
            if mj.rows() == 0 || mj.cols() == 0 {
                return Err(RelationError::InvalidStructure);
            }
        }
        let t = matrices.len();
        if f.arity() != t {
            return Err(RelationError::PolyArity {
                poly_arity: f.arity(),
                t,
            });
        }
        validate_polynomial(&f, t)?;

        let matrices = matrices
            .into_iter()
            .map(|mj| {
                if mj.is_identity_hint() {
                    CcsMatrix::Identity { n: mj.rows() }
                } else {
                    CcsMatrix::Csc(CscMat::from_dense_row_major(&mj))
                }
            })
            .collect();

        Ok(Self {
            matrices,
            f,
            n,
            m,
            domain_variables: None,
        })
    }

    pub fn new_sparse(matrices: Vec<CcsMatrix<F>>, f: SparsePoly<F>) -> Result<Self, RelationError> {
        if matrices.is_empty() {
            return Err(RelationError::InvalidStructure);
        }
        let n = matrices[0].rows();
        let m = matrices[0].cols();
        for mj in matrices.iter() {
            if matches!(mj, CcsMatrix::VerifierArtifact { .. }) {
                return Err(RelationError::Message(
                    "verifier-artifact matrices require the artifact-header constructor".into(),
                ));
            }
            if mj.rows() != n || mj.cols() != m {
                return Err(RelationError::InvalidStructure);
            }
            if mj.rows() == 0 || mj.cols() == 0 {
                return Err(RelationError::InvalidStructure);
            }
            if !mj.has_canonical_csc() {
                return Err(RelationError::Message(
                    "CCS sparse matrix is not in canonical CSC form".into(),
                ));
            }
        }
        let t = matrices.len();
        if f.arity() != t {
            return Err(RelationError::PolyArity {
                poly_arity: f.arity(),
                t,
            });
        }
        validate_polynomial(&f, t)?;
        Ok(Self {
            matrices,
            f,
            n,
            m,
            domain_variables: None,
        })
    }

    pub fn new_verifier_artifact_header(
        n: usize,
        m: usize,
        matrix_count: usize,
        f: SparsePoly<F>,
    ) -> Result<Self, RelationError> {
        if n == 0 || m == 0 || matrix_count == 0 || padded_carrier_width(m).is_none() {
            return Err(RelationError::InvalidStructure);
        }
        if f.arity() != matrix_count {
            return Err(RelationError::PolyArity {
                poly_arity: f.arity(),
                t: matrix_count,
            });
        }
        validate_polynomial(&f, matrix_count)?;
        Ok(Self {
            matrices: vec![CcsMatrix::VerifierArtifact { rows: n, cols: m }; matrix_count],
            f,
            n,
            m,
            domain_variables: None,
        })
    }

    pub fn with_domain_variables(mut self, variables: usize) -> Result<Self, RelationError> {
        self.domain_variables = Some(variables);
        self.validate_domain()?;
        Ok(self)
    }

    pub fn domain_rows(&self) -> usize {
        self.domain_variables
            .map_or(self.n, |variables| self.n.max(1 << variables))
    }

    pub fn cube_variables(&self) -> usize {
        self.domain_rows()
            .max(self.m.div_ceil(D) * D)
            .next_power_of_two()
            .max(2)
            .trailing_zeros() as usize
    }

    fn validate_domain(&self) -> Result<(), RelationError> {
        let Some(variables) = self.domain_variables else {
            return Ok(());
        };
        let size = u32::try_from(variables)
            .ok()
            .and_then(|shift| 1usize.checked_shl(shift))
            .ok_or(RelationError::InvalidStructure)?;
        let carrier = padded_carrier_width(self.m).ok_or(RelationError::InvalidStructure)?;
        if self.n > size || carrier > size {
            return Err(RelationError::InvalidStructure);
        }
        Ok(())
    }

    pub fn is_verifier_artifact_header(&self) -> bool {
        !self.matrices.is_empty()
            && self
                .matrices
                .iter()
                .all(|matrix| matches!(matrix, CcsMatrix::VerifierArtifact { .. }))
    }

    pub fn validate(&self) -> Result<(), RelationError> {
        if self.matrices.is_empty() || self.n == 0 || self.m == 0 || padded_carrier_width(self.m).is_none() {
            return Err(RelationError::InvalidStructure);
        }
        let artifact_header = self.is_verifier_artifact_header();
        if !artifact_header
            && self
                .matrices
                .iter()
                .any(|matrix| matches!(matrix, CcsMatrix::VerifierArtifact { .. }))
        {
            return Err(RelationError::InvalidStructure);
        }
        for matrix in &self.matrices {
            if matrix.rows() != self.n || matrix.cols() != self.m || (!artifact_header && !matrix.has_canonical_csc()) {
                return Err(RelationError::InvalidStructure);
            }
        }
        if self.f.arity() != self.matrices.len() {
            return Err(RelationError::PolyArity {
                poly_arity: self.f.arity(),
                t: self.matrices.len(),
            });
        }
        self.validate_domain()?;
        validate_polynomial(&self.f, self.matrices.len())
    }

    pub fn t(&self) -> usize {
        self.matrices.len()
    }

    pub fn max_degree(&self) -> u32 {
        self.f.max_degree()
    }
}

fn validate_polynomial<F>(polynomial: &SparsePoly<F>, arity: usize) -> Result<(), RelationError> {
    for term in polynomial.terms() {
        if term.exps.len() != arity {
            return Err(RelationError::Message(
                "polynomial term exponent count does not match its arity".into(),
            ));
        }
        if term
            .exps
            .iter()
            .try_fold(0u32, |degree, exponent| degree.checked_add(*exponent))
            .is_none()
        {
            return Err(RelationError::Message(
                "polynomial term total degree exceeds u32".into(),
            ));
        }
    }
    Ok(())
}

impl CcsStructure<neo_math::Fq> {
    pub fn transform_matrices_superneo(&self) -> Result<Self, RelationError> {
        if !self.m.is_multiple_of(D) {
            return Err(RelationError::Message(format!(
                "superneo matrix transform requires m multiple of D={}, got m={}",
                D, self.m
            )));
        }

        let bar = neo_math::superneo_bar_matrix();
        let mut out = Vec::with_capacity(self.matrices.len());
        for mj in &self.matrices {
            out.push(transform_ccs_matrix_superneo(mj, bar)?);
        }
        Ok(CcsStructure {
            domain_variables: self.domain_variables,
            ..CcsStructure::new_sparse(out, self.f.clone())?
        })
    }
}

fn transform_ccs_matrix_superneo(
    src: &CcsMatrix<neo_math::Fq>,
    bar: &[[neo_math::Fq; D]; D],
) -> Result<CcsMatrix<neo_math::Fq>, RelationError> {
    use neo_math::Fq;

    let nrows = src.rows();
    let ncols = src.cols();
    if !ncols.is_multiple_of(D) {
        return Err(RelationError::Message(format!(
            "superneo matrix transform requires ncols multiple of D={}, got ncols={}",
            D, ncols
        )));
    }

    let mut triplets: Vec<(usize, usize, Fq)> = Vec::new();
    match src {
        CcsMatrix::Identity { n } => {
            if *n != ncols {
                return Err(RelationError::Message(
                    "identity sentinel must be square before superneo transform".into(),
                ));
            }
            triplets.reserve(nrows * D);
            for r in 0..nrows {
                let block = r / D;
                let local = r % D;
                let base = block * D;
                for i in 0..D {
                    let coeff = bar[i][local];
                    if coeff != Fq::ZERO {
                        triplets.push((r, base + i, coeff));
                    }
                }
            }
        }
        CcsMatrix::Csc(m) => {
            triplets.reserve(m.vals().len() * D);
            for c in 0..m.ncols() {
                let block = c / D;
                let local = c % D;
                let base = block * D;
                for k in m.column_range(c) {
                    let r = m.row_index(k);
                    let v = m.vals()[k];
                    for i in 0..D {
                        let coeff = v * bar[i][local];
                        if coeff != Fq::ZERO {
                            triplets.push((r, base + i, coeff));
                        }
                    }
                }
            }
        }
        CcsMatrix::CscWithGeometricRuns { csc, geometric_runs } => {
            triplets.reserve(csc.vals().len() * D);
            for c in 0..csc.ncols() {
                let block = c / D;
                let local = c % D;
                let base = block * D;
                for k in csc.column_range(c) {
                    let r = csc.row_index(k);
                    let v = csc.vals()[k];
                    for i in 0..D {
                        let coeff = v * bar[i][local];
                        if coeff != Fq::ZERO {
                            triplets.push((r, base + i, coeff));
                        }
                    }
                }
            }
            for run in geometric_runs {
                run.for_each_term(|r, c, v| {
                    let block = c / D;
                    let local = c % D;
                    let base = block * D;
                    for (i, bar_row) in bar.iter().enumerate() {
                        let coeff = v * bar_row[local];
                        if coeff != Fq::ZERO {
                            triplets.push((r, base + i, coeff));
                        }
                    }
                });
            }
        }
        CcsMatrix::VerifierArtifact { .. } => {
            return Err(RelationError::Message(
                "SuperNeo matrix transformation requires materialized matrix content".into(),
            ));
        }
    }

    Ok(CcsMatrix::Csc(CscMat::from_triplets(triplets, nrows, ncols)))
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LaneCommitments<C> {
    pub ops: C,
    pub is: C,
    pub fs: C,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CcsClaim<C, F> {
    pub c: C,
    pub x: Vec<F>,
    pub m_in: usize,
    #[serde(default = "Option::default")]
    pub adv: Option<LaneCommitments<C>>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "F: serde::Serialize",
    deserialize = "F: serde::Deserialize<'de> + p3_field::PrimeCharacteristicRing + Clone + Eq"
))]
#[allow(non_snake_case)]
pub struct CcsWitness<F> {
    pub Z: Mat<F>,
}

impl<F: Copy> CcsWitness<F> {
    pub fn private_len(&self, m_in: usize, total: usize) -> Option<usize> {
        let private = total.checked_sub(m_in)?;
        self.Z
            .rows()
            .checked_mul(self.Z.cols())
            .is_some_and(|len| len >= total)
            .then_some(private)
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct V1_2Evaluations<K> {
    pub eval_k: Vec<K>,
    pub eval_a: Vec<Vec<K>>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(bound(
    serialize = "C: serde::Serialize, F: serde::Serialize, K: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>, F: serde::Deserialize<'de> + p3_field::PrimeCharacteristicRing + Clone + Eq, K: serde::Deserialize<'de>"
))]
#[allow(non_snake_case)]
pub struct CeClaim<C, F, K> {
    pub c: C,
    pub X: Mat<F>,
    pub r: Vec<K>,
    pub eval_k: Vec<K>,
    pub eval_a: Vec<Vec<K>>,
    pub m_in: usize,
    pub fold_digest: [u8; 32],
    #[serde(default = "Option::default")]
    pub adv: Option<LaneCommitments<C>>,
}

impl<C, F, K> CeClaim<C, F, K> {
    pub fn evaluation_families(&self) -> impl Iterator<Item = &[K]> {
        std::iter::once(self.eval_k.as_slice()).chain(self.eval_a.iter().map(Vec::as_slice))
    }

    pub fn evaluation_family_count(&self) -> usize {
        1 + self.eval_a.len()
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "F: serde::Serialize",
    deserialize = "F: serde::Deserialize<'de> + p3_field::PrimeCharacteristicRing + Clone + Eq"
))]
#[allow(non_snake_case)]
pub struct CeWitness<F> {
    pub Z: Mat<F>,
}

#[inline]
pub fn superneo_public_x_cols(m_in: usize) -> Option<usize> {
    m_in.is_multiple_of(D).then_some(m_in / D)
}

fn padded_carrier_width(m: usize) -> Option<usize> {
    m.div_ceil(D).checked_mul(D)
}

pub fn check_ccs_rowwise_zero<F: Field>(s: &CcsStructure<F>, x: &[F], w: &[F]) -> Result<(), CcsError> {
    if x.len() + w.len() != s.m {
        return Err(CcsError::Len {
            context: "z = x||w length",
            expected: s.m,
            got: x.len() + w.len(),
        });
    }
    let mut z = x.to_vec();
    z.extend_from_slice(w);

    let mut mz: Vec<Vec<F>> = Vec::with_capacity(s.t());
    for mj in &s.matrices {
        let mut v = vec![F::ZERO; s.n];
        mj.add_mul_into(&z, &mut v, s.n);
        mz.push(v);
    }

    for i in 0..s.n {
        let mut point = Vec::with_capacity(s.t());
        for j in 0..s.t() {
            point.push(mz[j][i]);
        }
        let val = s.f.eval(&point);
        if val != F::ZERO {
            return Err(CcsError::RowFail { row: i });
        }
    }
    Ok(())
}

pub fn check_ccs_rowwise_relaxed<F: Field>(
    s: &CcsStructure<F>,
    x: &[F],
    w: &[F],
    u: Option<&[F]>,
    e: Option<F>,
) -> Result<(), CcsError> {
    let e = e.unwrap_or(F::ONE);
    let zero_u: Vec<F>;
    let u = match u {
        Some(u) => {
            if u.len() != s.n {
                return Err(CcsError::Len {
                    context: "u (slack)",
                    expected: s.n,
                    got: u.len(),
                });
            }
            u
        }
        None => {
            zero_u = vec![F::ZERO; s.n];
            &zero_u
        }
    };
    if x.len() + w.len() != s.m {
        return Err(CcsError::Len {
            context: "z = x||w length",
            expected: s.m,
            got: x.len() + w.len(),
        });
    }
    let mut z = x.to_vec();
    z.extend_from_slice(w);

    let mut mz: Vec<Vec<F>> = Vec::with_capacity(s.t());
    for mj in &s.matrices {
        let mut v = vec![F::ZERO; s.n];
        mj.add_mul_into(&z, &mut v, s.n);
        mz.push(v);
    }

    for i in 0..s.n {
        let mut point = Vec::with_capacity(s.t());
        for j in 0..s.t() {
            point.push(mz[j][i]);
        }
        let val = s.f.eval(&point);
        if val != e * u[i] {
            return Err(CcsError::RowFail { row: i });
        }
    }
    Ok(())
}
