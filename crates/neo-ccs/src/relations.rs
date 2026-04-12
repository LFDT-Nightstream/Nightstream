use p3_field::{Field, PrimeCharacteristicRing};

use neo_math::{superneo_bar_block, KExtensions, Rq, D, F as GoldiF};
use neo_params::NeoParams;

use crate::{
    error::{CcsError, RelationError},
    matrix::Mat,
    poly::SparsePoly,
    sparse::{CcsMatrix, CscMat},
    traits::SModuleHomomorphism,
    utils::tensor_point,
};

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
        Ok(Self { matrices, f, n, m })
    }

    /// Number of matrices (arity of `f`).
    pub fn t(&self) -> usize {
        self.matrices.len()
    }

    /// Maximum degree of the CCS polynomial.
    pub fn max_degree(&self) -> u32 {
        self.f.max_degree()
    }

    /// Ensure the first matrix is the identity I_n, as assumed by paper's NC semantics.
    /// If not, insert I_n at index 0 and shift the polynomial arity/variables accordingly.
    pub fn ensure_identity_first(&self) -> Result<Self, RelationError>
    where
        F: p3_field::PrimeCharacteristicRing + Copy + Eq + Clone,
    {
        // If not square, we cannot insert a true identity; leave structure unchanged.
        if self.n != self.m {
            return Ok(self.clone());
        }
        let is_id0 = self
            .matrices
            .first()
            .map(|m0| m0.is_identity())
            .unwrap_or(false);
        if is_id0 {
            return Ok(self.clone());
        }
        // Insert identity at position 0
        let mut matrices = self.matrices.clone();
        matrices.insert(0, CcsMatrix::Identity { n: self.n });
        // Shift polynomial variables by inserting a dummy variable at the front
        let f = self.f.insert_var_at_front();
        Ok(CcsStructure {
            matrices,
            f,
            n: self.n,
            m: self.m,
        })
    }

    /// Owned variant of `ensure_identity_first` that avoids cloning when `M₀` is already identity.
    ///
    /// This is useful in hot paths where callers already own a `CcsStructure` and only need to
    /// normalize it (if necessary) for Ajtai/NC semantics.
    pub fn ensure_identity_first_owned(mut self) -> Result<Self, RelationError>
    where
        F: p3_field::PrimeCharacteristicRing + Copy + Eq + Clone,
    {
        // If not square, we cannot insert a true identity; leave structure unchanged.
        if self.n != self.m {
            return Ok(self);
        }
        let is_id0 = self
            .matrices
            .first()
            .map(|m0| m0.is_identity())
            .unwrap_or(false);
        if is_id0 {
            return Ok(self);
        }
        self.matrices.insert(0, CcsMatrix::Identity { n: self.n });
        self.f = self.f.insert_var_at_front();
        Ok(self)
    }

    /// **STRICT** validation: Assert that M₀ = I_n for Ajtai/NC pipeline.
    ///
    /// The Ajtai norm constraint (NC) layer assumes the first matrix is the identity
    /// for digit-range checks. If this invariant is violated, the sumcheck will fail
    /// with a mysterious error later. This function fails fast with a clear error message.
    ///
    /// # Errors
    /// - Returns error if n ≠ m (non-square CCS cannot have square identity)
    /// - Returns error if matrices list is empty
    /// - Returns error if M₀ is not the identity matrix I_n
    ///
    /// # Example
    /// ```ignore
    /// // Before using CCS in Ajtai/NC pipeline:
    /// ccs.assert_m0_is_identity_for_nc()?;
    /// ```
    pub fn assert_m0_is_identity_for_nc(&self) -> Result<(), RelationError>
    where
        F: p3_field::PrimeCharacteristicRing + Copy + Eq,
    {
        // Check 1: Square CCS required for identity to even make sense
        if self.n != self.m {
            return Err(RelationError::Message(format!(
                "Ajtai NC requires square CCS (n_constraints == n_vars), got {}×{}. \
                 You may need to pad your R1CS to square dimensions.",
                self.n, self.m
            )));
        }

        // Check 2: Must have at least one matrix
        if self.matrices.is_empty() {
            return Err(RelationError::Message(
                "Ajtai NC expects at least one matrix (M₀) in CCS".into(),
            ));
        }

        // Check 3: M₀ must be the identity matrix
        if !self.matrices[0].is_identity() {
            return Err(RelationError::Message(
                "Ajtai NC requires M₀ = I_n (identity matrix). \
                 Your CCS has a non-identity first matrix. \
                 This usually happens with rectangular R1CS or when r1cs_to_ccs \
                 doesn't produce identity-first form. \
                 Try: (1) ensure n==m in R1CS, or (2) call ensure_identity_first() \
                 before this check."
                    .into(),
            ));
        }

        Ok(())
    }
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
                let s = m.col_ptr[c];
                let e = m.col_ptr[c + 1];
                for k in s..e {
                    let r = m.row_idx[k];
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
    }

    Ok(CcsMatrix::Csc(CscMat::from_triplets(triplets, nrows, ncols)))
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
}

/// CCS witness: w and its decomposition Z = Decomp_b(z).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[allow(non_snake_case)]
pub struct CcsWitness<F> {
    /// Private witness w ∈ F^{m - m_in}.
    pub w: Vec<F>,
    /// Z ∈ F^{d×m}: decomposition matrix of z = x || w.
    pub Z: Mat<F>,
}

/// CE claim: (c, X, r, {y_ring_j}, ct, aux_openings).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[allow(non_snake_case)]
pub struct CeClaim<C, F, K> {
    /// Commitment to Z.
    pub c: C,
    /// X = L_x(Z) ∈ F^{d×m_in}
    pub X: Mat<F>,
    /// r ∈ K^{log n}
    pub r: Vec<K>,
    /// s_col ∈ K^{log m}: column-domain point used for the digit-range (NC) check.
    ///
    /// Legacy (square/identity-first) pipelines may leave this empty.
    #[serde(default)]
    pub s_col: Vec<K>,
    /// Ring-digit rows per CCS matrix output (j=0..t-1).
    ///
    /// Callers may store either:
    /// - the unpadded length `d` (= `Z.rows()`), or
    /// - the Ajtai-padded length `2^{ell_d}` (typically `D.next_power_of_two()`),
    ///   in which case the tail must be all zeros.
    pub y_ring: Vec<Vec<K>>,
    /// Scalar view of `y_ring`.
    ///
    /// In SuperNeo embedding, core entries are constant terms of each `y_ring[j]`.
    /// Existing pipelines may append additional scalar openings to this vector.
    pub ct: Vec<K>,
    /// Additional scalar openings that are not core CCS matrix outputs.
    ///
    /// This field is the CE-native home for sidecar/Route-A openings.
    #[serde(default)]
    pub aux_openings: Vec<K>,
    /// y_zcol := Z · χ_{s_col} ∈ K^{d} (digit rows, typically padded to 2^{ell_d}).
    ///
    /// Legacy (square/identity-first) pipelines may leave this empty.
    #[serde(default)]
    pub y_zcol: Vec<K>,
    /// m_in
    pub m_in: usize,
    /// **SECURITY**: Transcript-derived digest binding this ME to the folding proof
    pub fold_digest: [u8; 32],
    /// **PATTERN A**: Pre-commitment coordinates for linear link constraints
    /// c_step_coords[i] are the coordinates of the pre-commitment (with ρ=0 for EV part)
    /// Used to enforce: c_full[i] - c_step_coords[i] = ⟨L_i, U⟩ where U = ρ·y_step
    pub c_step_coords: Vec<F>,
    /// Pattern A: Offset where ρ-dependent part starts in witness vector (unused in Pattern B)
    pub u_offset: usize,
    /// Pattern A: Length of the ρ-dependent part (unused in Pattern B)
    pub u_len: usize,
}

/// CE witness: Z.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[allow(non_snake_case)]
pub struct CeWitness<F> {
    /// Z ∈ F^{d×m}
    pub Z: Mat<F>,
}

/// Witness matrix interpretation strategy.
///
/// Currently only SuperNeo packed layout is supported, but this enum
/// exists to support future witness formats without changing the
/// validation and extraction call sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WitnessLayout {
    SuperneoPacked,
}

fn witness_layout_for_expected_m<F: Field>(z: &Mat<F>, expected_m: usize) -> Result<WitnessLayout, CcsError> {
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
        return Ok(WitnessLayout::SuperneoPacked);
    }
    Err(CcsError::Dim {
        context: "Z shape vs SuperNeo packed width",
        expected: (D, want_cols),
        got: (z.rows(), z.cols()),
    })
}

#[inline]
fn witness_get<F: Field + Copy>(z: &Mat<F>, layout: WitnessLayout, rho: usize, col: usize) -> F {
    match layout {
        WitnessLayout::SuperneoPacked => {
            let blk = col / D;
            let off = col % D;
            if off == rho {
                z[(rho, blk)]
            } else {
                F::ZERO
            }
        }
    }
}

fn project_x_from_witness_layout<F: Field + Copy>(z: &Mat<F>, layout: WitnessLayout, m_in: usize) -> Mat<F> {
    let mut x = Mat::zero(D, m_in, F::ZERO);
    for rho in 0..D {
        for c in 0..m_in {
            x[(rho, c)] = witness_get(z, layout, rho, c);
        }
    }
    x
}

#[inline]
fn ct_from_y_digits_for_ccs_m<Kf: Field>(y_digits: &[Kf], expected_m: usize) -> Kf {
    debug_assert!(expected_m > 0);
    y_digits.first().copied().unwrap_or(Kf::ZERO)
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
            let s = csc.col_ptr[col];
            let e = csc.col_ptr[col + 1];
            match csc.row_idx[s..e].binary_search(&row) {
                Ok(idx) => csc.vals[s + idx].into(),
                Err(_) => GoldiF::ZERO,
            }
        }
    }
}

/// Build SuperNeo ring-coefficient linear forms for one CE point `r`.
///
/// Returns `forms[j][col][rho]` such that for each matrix `j`, the ring row
/// satisfies `y_ring[j][rho] = Σ_col forms[j][col][rho] * z[col]`, where `col`
/// ranges over logical witness columns padded up to the next multiple of `D`.
pub fn build_superneo_ring_forms<
    F: Field + PrimeCharacteristicRing + Copy + Into<GoldiF>,
    K: Field + From<F> + KExtensions + Copy,
>(
    s: &CcsStructure<F>,
    r: &[K],
) -> Result<Vec<Vec<[K; D]>>, CcsError> {
    let n_pad = s.n.next_power_of_two();
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
    let mut out = Vec::with_capacity(s.t());

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
                    let shifted = a_bar.mul_by_monomial(i);
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
    let _layout = witness_layout_for_expected_m(&wit.Z, m)?;
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

/// Check `X == L_x(Z)` and CE output consistency.
pub fn check_ce_consistency<F: Field, K: Field + From<F>, C, L: SModuleHomomorphism<F, C>>(
    _params: &NeoParams,
    s: &CcsStructure<F>,
    l: &L,
    inst: &CeClaim<C, F, K>,
    wit: &CeWitness<F>,
) -> Result<(), CcsError>
where
    C: PartialEq,
{
    let z_layout = witness_layout_for_expected_m(&wit.Z, s.m)?;

    // X = L_x(Z)
    let x_star = project_x_from_witness_layout(&wit.Z, z_layout, inst.m_in);
    if x_star.as_slice() != inst.X.as_slice() {
        return Err(CcsError::Relation("X != L_x(Z)".into()));
    }
    // c == L(Z) (always true in Π_CCS/Π_RLC composition; enforce here)
    let c_star = l.commit(&wit.Z);
    if c_star != inst.c {
        return Err(CcsError::Relation("c != L(Z)".into()));
    }

    // y_j == Z M_j^T r^b
    // Allow arbitrary n by deriving ℓ from the next power of two.
    // χ_r is length 2^ℓ, and we consume only the first n entries.
    let n_pad = s.n.next_power_of_two();
    let ell = n_pad.trailing_zeros() as usize;
    if inst.r.len() != ell {
        return Err(CcsError::Len {
            context: "r (extension point)",
            expected: ell,
            got: inst.r.len(),
        });
    }

    // Optional NC channel: y_zcol == Z · χ_{s_col} (column-domain).
    //
    // This is only checked when both `s_col` and `y_zcol` are present, so legacy callers
    // can omit these fields without failing consistency checks.
    if !(inst.s_col.is_empty() && inst.y_zcol.is_empty()) {
        if inst.s_col.is_empty() || inst.y_zcol.is_empty() {
            return Err(CcsError::Relation(
                "incomplete NC channel: expected both s_col and y_zcol".into(),
            ));
        }

        // Column-domain length is derived from CCS width `m` (not `n`).
        let m_pad = s.m.next_power_of_two().max(2);
        let ell_m = m_pad.trailing_zeros() as usize;
        if inst.s_col.len() != ell_m {
            return Err(CcsError::Len {
                context: "s_col (column extension point)",
                expected: ell_m,
                got: inst.s_col.len(),
            });
        }

        // Ajtai padding length for digit rows (matches `1 << ell_d` used by Π_CCS dims).
        let d_pad = D.next_power_of_two();
        let ell_d = d_pad.trailing_zeros() as usize;
        let d_pad = 1usize << ell_d;
        if inst.y_zcol.len() != d_pad {
            return Err(CcsError::Len {
                context: "y_zcol (padded digit rows)",
                expected: d_pad,
                got: inst.y_zcol.len(),
            });
        }

        // Compute y_zcol = Z · χ_{s_col}.
        let chi_s = crate::utils::tensor_point::<K>(&inst.s_col);
        let mut y_star = vec![K::ZERO; D];
        for rho in 0..D {
            let mut acc = K::ZERO;
            for c in 0..s.m {
                acc += K::from(witness_get(&wit.Z, z_layout, rho, c)) * chi_s[c];
            }
            y_star[rho] = acc;
        }
        y_star.resize(d_pad, K::ZERO);

        if y_star.as_slice() != inst.y_zcol.as_slice() {
            return Err(CcsError::Relation("y_zcol != Z · χ_{s_col}".into()));
        }
    }
    let rb = tensor_point::<K>(&inst.r); // K^n

    // for each j: v := M_j^T r^b ∈ K^m; then y_j = Z v ∈ K^d
    if inst.y_ring.len() != s.t() {
        return Err(CcsError::Len {
            context: "|y_ring|",
            expected: s.t(),
            got: inst.y_ring.len(),
        });
    }

    // Ajtai padding length for digit rows (matches `1 << ell_d` used by Π_CCS dims).
    let d_pad = D.next_power_of_two();

    for (j, mj) in s.matrices.iter().enumerate() {
        // v = M_j^T r^b (consume only the first n rows of χ_r)
        let mut v_k_m = vec![K::ZERO; s.m];
        mj.add_mul_transpose_into(&rb, &mut v_k_m, s.n);
        // y*_j = Z v_k_m
        let mut y_star = vec![K::ZERO; D];
        for rho in 0..D {
            let mut acc = K::ZERO;
            for c in 0..s.m {
                acc += K::from(witness_get(&wit.Z, z_layout, rho, c)) * v_k_m[c];
            }
            y_star[rho] = acc;
        }
        let yj = &inst.y_ring[j];
        let d = y_star.len();
        if yj.len() < d {
            return Err(CcsError::Len {
                context: "y_ring[j] (digit row)",
                expected: d,
                got: yj.len(),
            });
        }
        if yj.len() != d && yj.len() != d_pad {
            return Err(CcsError::Len {
                context: "y_ring[j] (digit row)",
                expected: d_pad,
                got: yj.len(),
            });
        }
        if y_star.as_slice() != &yj[..d] {
            return Err(CcsError::Relation("y_j != Z M_j^T r^b".into()));
        }
        if yj[d..].iter().any(|&x| x != K::ZERO) {
            return Err(CcsError::Relation("y_j != Z M_j^T r^b".into()));
        }
    }

    // Core CE invariant (SuperNeo-only): `ct[j] == y_ring[j][0]`.
    if inst.ct.len() < s.t() {
        return Err(CcsError::Len {
            context: "ct (core entries)",
            expected: s.t(),
            got: inst.ct.len(),
        });
    }
    for j in 0..s.t() {
        let want = ct_from_y_digits_for_ccs_m(&inst.y_ring[j], s.m);
        if inst.ct[j] != want {
            return Err(CcsError::Relation("ct[j] != y_ring[j][0]".into()));
        }
    }

    Ok(())
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
