use crate::matrix::Mat;

/// Minimal interface needed from Ajtai (or any S-module homomorphic) commitment.
///
/// We intentionally keep this trait tiny; `neo-ajtai` will implement it.
/// - `commit(Z)` returns the commitment `c`.
pub trait SModuleHomomorphism<F, C> {
    /// Commit to a `d × m` matrix `Z`.
    fn commit(&self, z: &Mat<F>) -> C;

    /// Commit to many `d × m` matrices.
    ///
    /// Default implementation commits one-by-one. Backends can override to batch
    /// work when many matrices share the same commitment parameters.
    fn commit_many(&self, zs: &[&Mat<F>]) -> Vec<C> {
        zs.iter().map(|z| self.commit(z)).collect()
    }
}
