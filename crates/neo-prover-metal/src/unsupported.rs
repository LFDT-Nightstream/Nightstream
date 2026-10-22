//! Type-compatible unavailable backend for non-Apple targets and shaderless builds.

use super::*;

pub struct MetalSession;

pub struct MetalGkrLayer<'a>(std::marker::PhantomData<&'a mut [u64]>);

impl MetalSession {
    pub fn new() -> Result<Self, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn keep_production_key(&mut self) {}

    pub(crate) fn release_production_key_rows(&self) {}

    pub fn dft_columns(&self, _values: &mut [u64], _width: usize, _generator: u64) -> Result<(), MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn gkr_layer<'a>(
        &self,
        _trees: Vec<Vec<&'a mut [u64]>>,
        _eq: &'a mut [u64],
    ) -> Result<MetalGkrLayer<'a>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn gkr_message(
        &self,
        _layer: &MetalGkrLayer<'_>,
        _weights: &[[u64; 3]],
        _mu: [u64; 3],
        _degree: usize,
    ) -> Result<Vec<[u64; 3]>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn gkr_fold(&self, _layer: &mut MetalGkrLayer<'_>, _r: [u64; 3]) -> Result<(), MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn poseidon2_merkle_layers(
        &self,
        _rows: usize,
        _width: usize,
        _fill: impl FnOnce(&mut [u64]),
    ) -> Result<Vec<Vec<PoseidonDigest>>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn poseidon2_merkle_leaves(&self, _values: &[u64], _width: usize) -> Result<Vec<PoseidonDigest>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn activity(&self) -> MetalActivity {
        MetalActivity::default()
    }
}
