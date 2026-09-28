use p3_field::{Field, PrimeCharacteristicRing};

use crate::{
    error::{CcsError, RelationError},
    matrix::Mat,
    poly::SparsePoly,
    sparse::{CcsMatrix, CscMat},
    traits::SModuleHomomorphism,
    utils::tensor_point,
};
use neo_math::{superneo_bar_block, KExtensions, Rq, D, F as GoldiF};

/// CCS structure: matrices {M_j} and a sparse polynomial `f` in `t` variables.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CcsStructure<F> {
    /// M_j ∈ F^{n×m}, j = 0..t-1
    pub matrices: Vec<CcsMatrix<F>>,
    /// Degree-`<u` polynomial in t variables.
    pub f: SparsePoly<F>,
    /// n (rows)
    pub n: usize,
    /// m (cols)
    pub m: usize,
}

impl<F: Field> CcsStructure<F> {
    /// Create a CCS structure; validates matrix shapes & polynomial arity.
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

        Ok(Self { matrices, f, n, m })
    }

    /// Create a CCS structure from sparse matrices (CSC / identity).
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
        Ok(Self { matrices, f, n, m })
    }

    /// Create a matrix-content-free CCS header for a separately verified
    /// evaluator artifact.
    pub fn new_verifier_artifact_header(
        n: usize,
        m: usize,
        matrix_count: usize,
        f: SparsePoly<F>,
    ) -> Result<Self, RelationError> {
        if n == 0 || m == 0 || matrix_count == 0 {
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
        })
    }

    /// Whether all matrix content is owned by a verifier artifact.
    pub fn is_verifier_artifact_header(&self) -> bool {
        !self.matrices.is_empty()
            && self
                .matrices
                .iter()
                .all(|matrix| matches!(matrix, CcsMatrix::VerifierArtifact { .. }))
    }

    /// Recheck all structure invariants at a public boundary.
    pub fn validate(&self) -> Result<(), RelationError> {
        if self.matrices.is_empty() || self.n == 0 || self.m == 0 {
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
        validate_polynomial(&self.f, self.matrices.len())
    }

    /// Number of matrices (arity of `f`).
    pub fn t(&self) -> usize {
        self.matrices.len()
    }

    /// Maximum degree of the CCS polynomial.
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
    /// SuperNeo matrix transform `M -> bar(M)` applied row-wise.
    ///
    /// The field-column dimension `m` must be divisible by `D` so rows can be partitioned
    /// into `d`-coefficient ring blocks.
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
        CcsStructure::new_sparse(out, self.f.clone())
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
    let mut transformed_blocks = Vec::new();
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
            triplets.reserve(m.vals.len() * D);
            for c in 0..m.ncols {
                let block = c / D;
                let local = c % D;
                let base = block * D;
                for k in m.column_range(c) {
                    let r = m.row_index(k);
                    let v = m.vals[k];
                    for i in 0..D {
                        let coeff = v * bar[i][local];
                        if coeff != Fq::ZERO {
                            triplets.push((r, base + i, coeff));
                        }
                    }
                }
            }
        }
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            triplets.reserve(csc.vals.len() * D);
            for c in 0..csc.ncols {
                let block = c / D;
                let local = c % D;
                let base = block * D;
                for k in csc.column_range(c) {
                    let r = csc.row_index(k);
                    let v = csc.vals[k];
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
            transformed_blocks.extend(
                blocks
                    .iter()
                    .map(|block| block.with_superneo_transformed_columns()),
            );
        }
        CcsMatrix::VerifierArtifact { .. } => {
            return Err(RelationError::Message(
                "SuperNeo matrix transformation requires materialized matrix content".into(),
            ));
        }
    }

    let csc = CscMat::from_triplets(triplets, nrows, ncols);
    if transformed_blocks.is_empty() {
        Ok(CcsMatrix::Csc(csc))
    } else {
        CcsMatrix::csc_with_seeded_phi81(csc, transformed_blocks)
            .map_err(|error| RelationError::Message(error.to_string()))
    }
}

/// Nebula split-witness lane commitments in the `adv` tuple.
///
/// Exactly three commitments, one per memory lane, each under its own
/// Ajtai matrix (`ops` under `A_ops`; `is` and `fs` under a shared
/// `A_mem`, which is what makes cross-segment boundary equality
/// meaningful). The all-or-nothing shape is deliberate: a claim either
/// carries a complete tuple or none (`Option<LaneCommitments<C>>`), so a
/// partial tuple is unrepresentable rather than merely invalid.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LaneCommitments<C> {
    /// Ops-lane commitment (`A_ops · embed(lane_ops)`).
    pub ops: C,
    /// Initial-scan-lane commitment (`A_mem · embed(lane_is)`).
    pub is: C,
    /// Final-scan-lane commitment (`A_mem · embed(lane_fs)`).
    pub fs: C,
}

/// CCS claim: (c, x) with public inputs x ⊂ z.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CcsClaim<C, F> {
    /// Commitment to Z (Ajtai over decomposition).
    pub c: C,
    /// Public inputs x ∈ F^{m_in}; z = x || w.
    pub x: Vec<F>,
    /// m_in
    pub m_in: usize,
    /// Nebula lane-commitment tuple; `None` for non-Nebula claims.
    /// Folds component-wise beside `c` and is opened by the terminal
    /// decider against its lane slices.
    #[serde(default = "Option::default")]
    pub adv: Option<LaneCommitments<C>>,
}

/// CCS witness: w and its decomposition Z = Decomp_b(z).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "F: serde::Serialize",
    deserialize = "F: serde::Deserialize<'de> + p3_field::PrimeCharacteristicRing + Clone + Eq"
))]
#[allow(non_snake_case)]
pub struct CcsWitness<F> {
    /// Private witness w ∈ F^{m - m_in}.
    pub w: Vec<F>,
    /// Z ∈ F^{d×m}: decomposition matrix of z = x || w.
    pub Z: Mat<F>,
}

impl<F: Copy> CcsWitness<F> {
    /// Validate the private-witness geometry without materializing a second
    /// copy of a packed assignment.
    pub fn private_len(&self, m_in: usize, total: usize) -> Option<usize> {
        let private = total.checked_sub(m_in)?;
        if self.w.len() == private {
            return Some(private);
        }
        (self.w.is_empty()
            && self
                .Z
                .rows()
                .checked_mul(self.Z.cols())
                .is_some_and(|len| len >= total))
        .then_some(private)
    }

    /// Borrow an explicit private witness or reconstruct it from the
    /// authoritative packed assignment `Z`.
    pub fn private_values(&self, m_in: usize, total: usize) -> Option<std::borrow::Cow<'_, [F]>> {
        let private = self.private_len(m_in, total)?;
        if self.w.len() == private {
            return Some(std::borrow::Cow::Borrowed(&self.w));
        }
        let rows = self.Z.rows();
        let values = (m_in..total)
            .map(|column| self.Z[(column % rows, column / rows)])
            .collect();
        Some(std::borrow::Cow::Owned(values))
    }
}

/// Separate SuperNeo v1.1 evaluation families.
#[allow(non_camel_case_types)]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct V1_1Evaluations<K> {
    /// Paper `Eval_K`: the Pad evaluation family.
    pub eval_k: Vec<K>,
    /// Paper `Eval_A`: one family for each genuine CCS matrix.
    pub eval_a: Vec<Vec<K>>,
}

/// SuperNeo v1.1 CE claim: `(c, X, r, Eval_K, Eval_A, aux_openings)`.
///
/// `eval_k` is the Pad evaluation family. `eval_a` contains only the genuine
/// CCS-matrix evaluation families. Pad is never stored as matrix zero.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(bound(
    serialize = "C: serde::Serialize, F: serde::Serialize, K: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>, F: serde::Deserialize<'de> + p3_field::PrimeCharacteristicRing + Clone + Eq, K: serde::Deserialize<'de>"
))]
#[allow(non_snake_case)]
pub struct CeClaim<C, F, K> {
    /// Commitment to Z.
    pub c: C,
    /// Exact coefficient embedding `X = L_x(Z) ∈ F^{d×(m_in/d)}`.
    /// Valid protocol claims require `m_in % d == 0`.
    pub X: Mat<F>,
    /// r ∈ K^{log n}
    pub r: Vec<K>,
    /// Paper `Eval_K`: the coefficient-complete Pad evaluation in `R_K`.
    ///
    /// Callers may store either:
    /// - the unpadded length `d` (= `Z.rows()`), or
    /// - the Ajtai-padded length `2^{ell_d}` (typically `D.next_power_of_two()`),
    ///   in which case the tail must be all zeros.
    pub eval_k: Vec<K>,
    /// Paper `Eval_A`: one coefficient-complete evaluation for each genuine
    /// CCS matrix. The outer length is exactly `structure.t()`.
    pub eval_a: Vec<Vec<K>>,
    /// m_in
    pub m_in: usize,
    /// **SECURITY**: Transcript-derived digest binding this ME to the folding proof
    pub fold_digest: [u8; 32],
    /// Nebula lane-commitment tuple; `None` for non-Nebula claims.
    /// Mixed by the same public ρ/`b`-power arithmetic as `c` through
    /// Π_RLC/Π_DEC. The reductions do not inspect its semantics.
    #[serde(default = "Option::default")]
    pub adv: Option<LaneCommitments<C>>,
}

impl<C, F, K> CeClaim<C, F, K> {
    /// Iterate the paper output message order: `Eval_K`, then each `Eval_A`.
    pub fn evaluation_families(&self) -> impl Iterator<Item = &[K]> {
        std::iter::once(self.eval_k.as_slice()).chain(self.eval_a.iter().map(Vec::as_slice))
    }

    /// Number of v1_1 evaluation families, including the separate Pad family.
    pub fn evaluation_family_count(&self) -> usize {
        1 + self.eval_a.len()
    }
}

impl<C, F, K: Copy> CeClaim<C, F, K> {
    /// Derive constant coefficients for codec and legacy wire adapters. These
    /// values are never stored as independent claim authority.
    pub fn evaluation_constant_terms(&self) -> Option<Vec<K>> {
        self.evaluation_families()
            .map(|family| family.first().copied())
            .collect()
    }
}

/// CE witness: Z.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "F: serde::Serialize",
    deserialize = "F: serde::Deserialize<'de> + p3_field::PrimeCharacteristicRing + Clone + Eq"
))]
#[allow(non_snake_case)]
pub struct CeWitness<F> {
    /// Z ∈ F^{d×m}
    pub Z: Mat<F>,
}

/// Storage width for the coefficient embedding of `m_in` field elements.
/// Protocol claims additionally require `m_in % D == 0`, which makes this
/// exact division for every valid public input.
#[inline]
pub fn superneo_public_x_cols(m_in: usize) -> usize {
    m_in.div_ceil(D)
}

fn validate_superneo_witness_mat_for_expected_m<F: Field>(z: &Mat<F>, expected_m: usize) -> Result<(), CcsError> {
    if z.rows() != D {
        return Err(CcsError::Dim {
            context: "Z rows (expected D)",
            expected: (D, expected_m),
            got: (z.rows(), z.cols()),
        });
    }
    if expected_m == 0 {
        return Err(CcsError::Relation("expected_m must be > 0".into()));
    }
    let want_cols = expected_m.div_ceil(D);
    if z.cols() == want_cols {
        let pad_end = want_cols
            .checked_mul(D)
            .ok_or_else(|| CcsError::Relation("witness padding overflow".into()))?;
        for c in expected_m..pad_end {
            let blk = c / D;
            let off = c % D;
            if z[(off, blk)] != F::ZERO {
                return Err(CcsError::Relation(
                    format!("non-zero padded coefficient at logical index {c} (blk={blk}, off={off})").into(),
                ));
            }
        }
        return Ok(());
    }
    Err(CcsError::Dim {
        context: "Z shape vs SuperNeo packed width",
        expected: (D, want_cols),
        got: (z.rows(), z.cols()),
    })
}

fn matrix_entry_base_f<F: Field + Copy + Into<GoldiF>>(mat: &CcsMatrix<F>, row: usize, col: usize) -> GoldiF {
    if row >= mat.rows() || col >= mat.cols() {
        return GoldiF::ZERO;
    }
    match mat {
        CcsMatrix::Identity { .. } => {
            if row == col {
                GoldiF::ONE
            } else {
                GoldiF::ZERO
            }
        }
        CcsMatrix::Csc(csc) => {
            let mut acc = GoldiF::ZERO;
            for idx in csc.column_range(col) {
                if csc.row_index(idx) == row {
                    acc += csc.vals[idx].into();
                }
            }
            acc
        }
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            let mut acc = GoldiF::ZERO;
            for idx in csc.column_range(col) {
                if csc.row_index(idx) == row {
                    acc += csc.vals[idx].into();
                }
            }
            for block in blocks {
                acc += block.entry::<GoldiF>(row, col);
            }
            for run in geometric_runs {
                acc += run.entry(row, col).into();
            }
            acc
        }
        CcsMatrix::VerifierArtifact { .. } => {
            panic!("raw matrix entry access is unavailable for a verifier-artifact matrix")
        }
    }
}

/// Build identity-first SuperNeo ring-coefficient linear forms for one CE point `r`.
///
/// Returns `forms[j][col][rho]` such that the virtual padded identity is at
/// `j = 0` and structure matrix `j - 1` follows it. Each ring row satisfies
/// `y_ring[j][rho] = Σ_col forms[j][col][rho] * z[col]`, where `col` ranges
/// over witness columns padded up to the next multiple of `D`.
pub fn build_superneo_ring_forms<
    F: Field + PrimeCharacteristicRing + Copy + Into<GoldiF>,
    K: Field + From<F> + KExtensions + Copy,
>(
    s: &CcsStructure<F>,
    r: &[K],
) -> Result<Vec<Vec<[K; D]>>, CcsError> {
    let n_pad = s.n.max(s.m.div_ceil(D) * D).next_power_of_two();
    let ell = n_pad.trailing_zeros() as usize;
    if r.len() != ell {
        return Err(CcsError::Len {
            context: "r (extension point)",
            expected: ell,
            got: r.len(),
        });
    }

    let chi_r = tensor_point::<K>(r);
    let m_eff = s.m.div_ceil(D) * D;
    let block_count = m_eff / D;
    let mut out = Vec::with_capacity(s.t() + 1);

    let mut identity_forms = vec![[K::ZERO; D]; m_eff];
    for (row, &weight) in chi_r.iter().take(m_eff).enumerate() {
        if weight == K::ZERO {
            continue;
        }
        let block = row / D;
        let mut identity_row = [GoldiF::ZERO; D];
        identity_row[row % D] = GoldiF::ONE;
        let identity_bar = Rq(superneo_bar_block(identity_row));
        for witness_lane in 0..D {
            let mut basis = [GoldiF::ZERO; D];
            basis[witness_lane] = GoldiF::ONE;
            let shifted = identity_bar.mul(&Rq(basis));
            let slot = &mut identity_forms[block * D + witness_lane];
            for rho in 0..D {
                slot[rho] += weight.scale_base(shifted.0[rho]);
            }
        }
    }
    out.push(identity_forms);

    for matrix in &s.matrices {
        let mut forms = vec![[K::ZERO; D]; m_eff];
        for (row, &weight) in chi_r.iter().take(s.n).enumerate() {
            if weight == K::ZERO {
                continue;
            }
            for blk in 0..block_count {
                let base = blk * D;
                let mut a = [GoldiF::ZERO; D];
                for (i, coeff) in a.iter_mut().enumerate() {
                    *coeff = matrix_entry_base_f(matrix, row, base + i);
                }
                let a_bar = Rq(superneo_bar_block(a));
                for i in 0..D {
                    let mut basis = [GoldiF::ZERO; D];
                    basis[i] = GoldiF::ONE;
                    let shifted = a_bar.mul(&Rq(basis));
                    let slot = &mut forms[base + i];
                    for rho in 0..D {
                        slot[rho] += weight.scale_base(shifted.0[rho]);
                    }
                }
            }
        }
        out.push(forms);
    }

    Ok(out)
}

/// Check `c == L(Z)` for CCS claim.
/// Note: The critical Z == Decomp_b(z) check is now handled in the folding pipeline
/// where both neo-ccs and neo-ajtai dependencies are available.
pub fn check_ccs_claim_opening<F: Field, C, L: SModuleHomomorphism<F, C>>(
    l: &L,
    inst: &CcsClaim<C, F>,
    wit: &CcsWitness<F>,
) -> Result<Vec<F>, CcsError>
where
    C: PartialEq,
{
    // shape sanity
    let m = inst.m_in + wit.w.len();
    validate_superneo_witness_mat_for_expected_m(&wit.Z, m)?;
    // z = x || w
    if inst.x.len() != inst.m_in {
        return Err(CcsError::Len {
            context: "x (public)",
            expected: inst.m_in,
            got: inst.x.len(),
        });
    }
    let mut z = inst.x.clone();
    z.extend_from_slice(&wit.w);

    // === COMMITMENT BINDING ===
    let c_star = l.commit(&wit.Z);
    if c_star != inst.c {
        return Err(CcsError::Relation("c != L(Z)".into()));
    }

    Ok(z)
}

/// **MUST**: Verify CCS satisfiability `f(M z) = 0` **row-wise** with public inputs `x`.
///
/// This matches Def. 17's condition `f(Mg_1 z, …, Mg_t z) ∈ ZS_n` by simply
/// checking that for each row i, `f((M_1 z)[i], …, (M_t z)[i]) == 0`.
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

    // Compute M_j z for every j
    let mut mz: Vec<Vec<F>> = Vec::with_capacity(s.t());
    for mj in &s.matrices {
        let mut v = vec![F::ZERO; s.n];
        mj.add_mul_into(&z, &mut v, s.n);
        mz.push(v);
    }

    // Row-wise: for each i, evaluate f( (M_1 z)[i], ..., (M_t z)[i] ) == 0
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

/// **MUST**: Verify **relaxed CCS** `f(M z) = e * u` row-wise (defaults `u=0`, `e=1`).
///
/// This corresponds to the usual relaxed CCS used in Nova/HyperNova/Neo.
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

    // M_j z for every j
    let mut mz: Vec<Vec<F>> = Vec::with_capacity(s.t());
    for mj in &s.matrices {
        let mut v = vec![F::ZERO; s.n];
        mj.add_mul_into(&z, &mut v, s.n);
        mz.push(v);
    }

    // Row-wise: f( (M_1 z)[i], ..., (M_t z)[i] ) == e * u[i]
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
