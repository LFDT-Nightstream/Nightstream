use crate::matrix::Mat;

pub trait SModuleHomomorphism<F, C> {
    fn commit(&self, z: &Mat<F>) -> C;

    fn commit_many(&self, zs: &[&Mat<F>]) -> Vec<C> {
        zs.iter().map(|z| self.commit(z)).collect()
    }
}
