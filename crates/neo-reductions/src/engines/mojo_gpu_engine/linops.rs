use neo_ccs::Mat;
// use neo_math::{F, K};
use p3_field::{Field, PrimeCharacteristicRing};

/// Trait for linear algebra operations, abstracting over CPU/GPU backends.
pub trait LinOps<Ff, Kf> 
where 
    Ff: Field + PrimeCharacteristicRing + Copy,
    Kf: From<Ff> + Copy + core::ops::AddAssign + core::ops::Mul<Output = Kf> + core::ops::MulAssign + core::ops::Sub<Output = Kf>,
{
    /// General Matrix-Matrix Multiplication (GEMM): out = alpha * a * b + beta * out
    /// a: d x k
    /// b: k x m
    /// out: d x m
    fn gemm(
        &self,
        a: &Mat<Ff>,
        b: &Mat<Ff>,
        out: &mut Mat<Ff>,
        alpha: Ff,
        beta: Ff,
    );

    /// Matrix-Vector Multiplication with Field matrix and Extension vector: y = a * x
    /// a: d x m (elements in F)
    /// x: m (elements in K)
    /// y: d (elements in K)
    fn gemv_FK(
        &self,
        a: &Mat<Ff>,
        x: &[Kf],
        y: &mut [Kf],
    );
    
    /// Matrix-Matrix Multiplication with Field matrix and Extension matrix: out = a * b
    /// a: d x k (elements in F)
    /// b: k x m (elements in K) - represented as Mat<K> or similar? 
    /// For now, let's assume we might need to multiply F-matrix by K-matrix (e.g. Z_i * V).
    /// But Z_i is usually F. V is K.
    /// Matrix-Vector Multiplication with Transposed Field matrix and Extension vector: y = a^T * x
    /// a: m x d (elements in F) -> a^T is d x m
    /// x: m (elements in K)
    /// y: d (elements in K)
    fn gemv_transpose_FK(
        &self,
        a: &Mat<Ff>,
        x: &[Kf],
        y: &mut [Kf],
    );

    /// Matrix-Matrix Multiplication with Field matrix and Extension matrix: out = a * b
    fn gemm_FK(
        &self,
        a: &Mat<Ff>,
        b: &Mat<Kf>,
        out: &mut Mat<Kf>,
    );
}

/// CPU implementation of `LinOps`.
///
/// Note: This implementation assumes that `Kf::default()` returns the additive identity (zero).
/// This is true for `K` (Goldilocks extension) but should be kept in mind for generic usage.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuLinOps;

impl<Ff, Kf> LinOps<Ff, Kf> for CpuLinOps
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    Kf: From<Ff> + Copy + core::ops::AddAssign + core::ops::Mul<Output = Kf> + core::ops::MulAssign + core::ops::Sub<Output = Kf> + Default,
{
    fn gemm(
        &self,
        a: &Mat<Ff>,
        b: &Mat<Ff>,
        out: &mut Mat<Ff>,
        alpha: Ff,
        beta: Ff,
    ) {
        assert_eq!(a.cols(), b.rows(), "gemm: dimensions mismatch a.cols != b.rows");
        assert_eq!(out.rows(), a.rows(), "gemm: dimensions mismatch out.rows != a.rows");
        assert_eq!(out.cols(), b.cols(), "gemm: dimensions mismatch out.cols != b.cols");

        let d = a.rows();
        let k = a.cols();
        let m = b.cols();

        // Naive O(d*k*m)
        for r in 0..d {
            for c in 0..m {
                let mut acc = Ff::ZERO;
                for i in 0..k {
                    acc += a[(r, i)] * b[(i, c)];
                }
                
                if beta == Ff::ZERO {
                    out[(r, c)] = alpha * acc;
                } else {
                    out[(r, c)] = alpha * acc + beta * out[(r, c)];
                }
            }
        }
    }

    fn gemv_FK(
        &self,
        a: &Mat<Ff>,
        x: &[Kf],
        y: &mut [Kf],
    ) {
        assert_eq!(a.cols(), x.len(), "gemv_FK: dimensions mismatch a.cols != x.len");
        assert_eq!(a.rows(), y.len(), "gemv_FK: dimensions mismatch a.rows != y.len");

        let rows = a.rows();
        let cols = a.cols();

        for r in 0..rows {
            let mut acc = Kf::default(); // K::ZERO
            for c in 0..cols {
                let val_a = a[(r, c)];
                if val_a != Ff::ZERO {
                    acc += Kf::from(val_a) * x[c];
                }
            }
            y[r] = acc;
        }
    }

    fn gemv_transpose_FK(
        &self,
        a: &Mat<Ff>,
        x: &[Kf],
        y: &mut [Kf],
    ) {
        // y = a^T * x
        // a is rows x cols. a^T is cols x rows.
        // x is rows (length of a's rows).
        // y is cols (length of a's cols).
        assert_eq!(a.rows(), x.len(), "gemv_transpose_FK: dimensions mismatch a.rows != x.len");
        assert_eq!(a.cols(), y.len(), "gemv_transpose_FK: dimensions mismatch a.cols != y.len");

        let rows = a.rows();
        let cols = a.cols();

        // Initialize y to zero
        for val in y.iter_mut() { *val = Kf::default(); }

        for r in 0..rows {
            let x_val = x[r];
            for c in 0..cols {
                let val_a = a[(r, c)];
                if val_a != Ff::ZERO {
                    y[c] += Kf::from(val_a) * x_val;
                }
            }
        }
    }

    fn gemm_FK(
        &self,
        a: &Mat<Ff>,
        b: &Mat<Kf>,
        out: &mut Mat<Kf>,
    ) {
        assert_eq!(a.cols(), b.rows(), "gemm_FK: dimensions mismatch a.cols != b.rows");
        assert_eq!(out.rows(), a.rows(), "gemm_FK: dimensions mismatch out.rows != a.rows");
        assert_eq!(out.cols(), b.cols(), "gemm_FK: dimensions mismatch out.cols != b.cols");

        let d = a.rows();
        let k = a.cols();
        let m = b.cols();

        for r in 0..d {
            for c in 0..m {
                let mut acc = Kf::default();
                for i in 0..k {
                    let val_a = a[(r, i)];
                    if val_a != Ff::ZERO {
                        acc += Kf::from(val_a) * b[(i, c)];
                    }
                }
                out[(r, c)] = acc;
            }
        }
    }
}
