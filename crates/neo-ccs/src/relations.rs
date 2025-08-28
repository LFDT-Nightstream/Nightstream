use p3_field::Field;

use crate::{
    error::{CcsError, RelationError},
    matrix::{Mat, MatRef},
    poly::SparsePoly,
    traits::SModuleHomomorphism,
    utils::{validate_power_of_two, tensor_point, mat_vec_mul_ff, mat_vec_mul_fk},
};

/// CCS structure: matrices {M_j} and a sparse polynomial `f` in `t` variables.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CcsStructure<F> {
    /// M_j ∈ F^{n×m}, j = 0..t-1
    pub matrices: Vec<Mat<F>>,
    /// Degree-`<u` polynomial in t variables.
    pub f: SparsePoly<F>,
    /// n (rows)
    pub n: usize,
    /// m (cols)
    pub m: usize,
}

impl<F: Field> CcsStructure<F> {
    /// Create a CCS structure; validates matrix shapes & polynomial arity.
    pub fn new(matrices: Vec<Mat<F>>, f: SparsePoly<F>) -> Result<Self, RelationError> {
        if matrices.is_empty() { return Err(RelationError::InvalidStructure); }
        let n = matrices[0].rows();
        let m = matrices[0].cols();
        for (_j, mj) in matrices.iter().enumerate() {
            if mj.rows() != n || mj.cols() != m {
                return Err(RelationError::InvalidStructure);
            }
            if mj.rows() == 0 || mj.cols() == 0 {
                return Err(RelationError::InvalidStructure);
            }
        }
        let t = matrices.len();
        if f.arity() != t {
            return Err(RelationError::PolyArity { poly_arity: f.arity(), t });
        }
        Ok(Self { matrices, f, n, m })
    }

    /// Number of matrices (arity of `f`).
    pub fn t(&self) -> usize { self.matrices.len() }
}

/// MCS instance: (c, x) with public inputs x ⊂ z (see Def. 17).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct McsInstance<C, F> {
    /// Commitment to Z (Ajtai over decomposition).
    pub c: C,
    /// Public inputs x ∈ F^{m_in}; z = x || w.
    pub x: Vec<F>,
    /// m_in
    pub m_in: usize,
}

/// MCS witness: w and its decomposition Z = Decomp_b(z) (we need Z for consistency checks).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[allow(non_snake_case)]
pub struct McsWitness<F> {
    /// Private witness w ∈ F^{m - m_in}.
    pub w: Vec<F>,
    /// Z ∈ F^{d×m}: decomposition matrix of z = x || w.
    pub Z: Mat<F>,
}

/// ME instance: (c, X, r, {y_j}). See Def. 18.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[allow(non_snake_case)]
pub struct MeInstance<C, F, K> {
    /// Commitment to Z.
    pub c: C,
    /// X = L_x(Z) ∈ F^{d×m_in}
    pub X: Mat<F>,
    /// r ∈ K^{log n}
    pub r: Vec<K>,
    /// y_j ∈ K^{d} for j=0..t-1 (stored as a vector-of-vectors length t, each len d).
    pub y: Vec<Vec<K>>,
    /// m_in
    pub m_in: usize,
}

/// ME witness: Z.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[allow(non_snake_case)]
pub struct MeWitness<F> {
    /// Z ∈ F^{d×m}
    pub Z: Mat<F>,
}

/// Check `c == L(Z)` for MCS. Also return `z = x || w` for downstream use.
pub fn check_mcs_opening<F: Field, C, L: SModuleHomomorphism<F, C>>(
    l: &L,
    inst: &McsInstance<C, F>,
    wit: &McsWitness<F>,
) -> Result<Vec<F>, CcsError>
where C: PartialEq
{
    // shape sanity
    let m = inst.m_in + wit.w.len();
    if wit.Z.cols() != m {
        return Err(CcsError::Dim{
            context: "Z (cols) vs m_in + |w|",
            expected: (wit.Z.rows(), m),
            got: (wit.Z.rows(), wit.Z.cols()),
        });
    }
    // z = x || w
    if inst.x.len() != inst.m_in {
        return Err(CcsError::Len{ context: "x (public)", expected: inst.m_in, got: inst.x.len()});
    }
    let mut z = inst.x.clone();
    z.extend_from_slice(&wit.w);

    // c == L(Z)
    let c_star = l.commit(&wit.Z);
    if c_star != inst.c {
        return Err(CcsError::Relation("c != L(Z)".into()));
    }
    Ok(z)
}

/// Check `X == L_x(Z)` and `y_j == Z M_j^T r^b` for ME (Def. 18).
pub fn check_me_consistency<F: Field, K: Field + From<F>, C, L: SModuleHomomorphism<F, C>>(
    s: &CcsStructure<F>,
    l: &L,
    inst: &MeInstance<C, F, K>,
    wit: &MeWitness<F>,
) -> Result<(), CcsError>
where C: PartialEq
{
    // X = L_x(Z)
    let x_star = l.project_x(&wit.Z, inst.m_in);
    if x_star.as_slice() != inst.X.as_slice() {
        return Err(CcsError::Relation("X != L_x(Z)".into()));
    }
    // c == L(Z) (always true in Π_CCS/Π_RLC composition; enforce here)
    let c_star = l.commit(&wit.Z);
    if c_star != inst.c {
        return Err(CcsError::Relation("c != L(Z)".into()));
    }

    // y_j == Z M_j^T r^b
    if !validate_power_of_two(s.n) {
        return Err(CcsError::NNotPowerOfTwo { n: s.n });
    }
    let ell = s.n.trailing_zeros() as usize;
    if inst.r.len() != ell {
        return Err(CcsError::Len {
            context: "r (extension point)",
            expected: ell,
            got: inst.r.len(),
        });
    }
    let rb = tensor_point::<K>(&inst.r); // K^n

    // for each j: v := M_j^T r^b ∈ K^m; then y_j = Z v ∈ K^d
    if inst.y.len() != s.t() { return Err(CcsError::Len{ context: "|y|", expected: s.t(), got: inst.y.len() }); }

    for (j, mj) in s.matrices.iter().enumerate() {
        // v = M_j^T r^b  →  v[c] = Σ_r M_j[r,c] * rb[r]
        let v_k_m = {
            // multiply transpose-by-vector without materializing M_j^T
            let mut v = vec![K::ZERO; s.m];
            for r in 0..s.n {
                let rb_r = rb[r];
                let row = mj.row(r);
                for c in 0..s.m {
                    let a_k: K = row[c].into();
                    v[c] += a_k * rb_r;
                }
            }
            v
        };
        // y*_j = Z v_k_m
        let z_ref = MatRef::from_mat(&wit.Z);
        let y_star = mat_vec_mul_fk::<F, K>(z_ref.data, z_ref.rows, z_ref.cols, &v_k_m);
        if y_star != inst.y[j] {
            return Err(CcsError::Relation("y_j != Z M_j^T r^b".into()));
        }
    }
    Ok(())
}

/// **MUST**: Verify CCS satisfiability `f(M z) = 0` **row-wise** with public inputs `x`.
///
/// This matches Def. 17's condition `f(Mg_1 z, …, Mg_t z) ∈ ZS_n` by simply
/// checking that for each row i, `f((M_1 z)[i], …, (M_t z)[i]) == 0`.
pub fn check_ccs_rowwise_zero<F: Field>(
    s: &CcsStructure<F>,
    x: &[F],
    w: &[F],
) -> Result<(), CcsError> {
    if x.len() + w.len() != s.m {
        return Err(CcsError::Len{ context: "z = x||w length", expected: s.m, got: x.len()+w.len() });
    }
    let mut z = x.to_vec(); z.extend_from_slice(w);

    // Compute M_j z for every j
    let mut mz: Vec<Vec<F>> = Vec::with_capacity(s.t());
    for mj in &s.matrices {
        let v = mat_vec_mul_ff::<F>(mj.as_slice(), s.n, s.m, &z);
        mz.push(v);
    }

    // Row-wise: for each i, evaluate f( (M_1 z)[i], ..., (M_t z)[i] ) == 0
    for i in 0..s.n {
        let mut point = Vec::with_capacity(s.t());
        for j in 0..s.t() { point.push(mz[j][i]); }
        let val = s.f.eval(&point);
        if val != F::ZERO { return Err(CcsError::RowFail{ row: i }); }
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
            if u.len() != s.n { return Err(CcsError::Len{ context: "u (slack)", expected: s.n, got: u.len() }); }
            u
        },
        None => {
            zero_u = vec![F::ZERO; s.n];
            &zero_u
        }
    };
    if x.len() + w.len() != s.m {
        return Err(CcsError::Len{ context: "z = x||w length", expected: s.m, got: x.len()+w.len() });
    }
    let mut z = x.to_vec(); z.extend_from_slice(w);

    // M_j z for every j
    let mut mz: Vec<Vec<F>> = Vec::with_capacity(s.t());
    for mj in &s.matrices {
        let v = mat_vec_mul_ff::<F>(mj.as_slice(), s.n, s.m, &z);
        mz.push(v);
    }

    // Row-wise: f( (M_1 z)[i], ..., (M_t z)[i] ) == e * u[i]
    for i in 0..s.n {
        let mut point = Vec::with_capacity(s.t());
        for j in 0..s.t() { point.push(mz[j][i]); }
        let val = s.f.eval(&point);
        if val != e * u[i] { return Err(CcsError::RowFail{ row: i }); }
    }
    Ok(())
}
