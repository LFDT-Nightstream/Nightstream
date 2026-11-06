//! Device kernels: one file per Metal shader file of `neo-prover-metal`,
//! each a `#[cuda_module]`. `field` holds the arithmetic that they share.

use std::sync::Arc;

use cuda_core::CudaContext;
use cuda_host::EmbeddedModuleError;

pub(crate) mod assignments;
pub(crate) mod carried;
pub(crate) mod dec_forms;
pub(crate) mod decomposition;
pub(crate) mod dft;
pub(crate) mod field;
pub(crate) mod gkr;
pub(crate) mod joint;
pub(crate) mod nightstream;
pub(crate) mod production_commitment;
pub(crate) mod relation;

#[cfg(test)]
#[path = "../../tests/unit/field.rs"]
mod field_tests;

/// Every kernel module of the crate, over one loaded CUDA module: the crate
/// embeds one PTX bundle that holds every kernel.
pub(crate) struct Kernels {
    pub(crate) nightstream: nightstream::nightstream_kernels::LoadedModule,
    pub(crate) production_commitment: production_commitment::production_commitment_kernels::LoadedModule,
    pub(crate) dft: dft::dft_kernels::LoadedModule,
    pub(crate) gkr: gkr::gkr_kernels::LoadedModule,
    pub(crate) joint: joint::joint_kernels::LoadedModule,
    pub(crate) assignments: assignments::assignment_kernels::LoadedModule,
    pub(crate) carried: carried::carried_kernels::LoadedModule,
    pub(crate) relation: relation::relation_kernels::LoadedModule,
    pub(crate) dec_forms: dec_forms::dec_form_kernels::LoadedModule,
    pub(crate) decomposition: decomposition::decomposition_kernels::LoadedModule,
}

impl Kernels {
    pub(crate) fn load(context: &Arc<CudaContext>) -> Result<Self, EmbeddedModuleError> {
        // `load` references the bundle's link anchor, which keeps the bundle
        // in every binary that links this crate.
        let nightstream = nightstream::nightstream_kernels::load(context)?;
        let module = nightstream.as_cuda_module().clone();
        Ok(Self {
            production_commitment: production_commitment::production_commitment_kernels::from_module(module.clone())?,
            dft: dft::dft_kernels::from_module(module.clone())?,
            gkr: gkr::gkr_kernels::from_module(module.clone())?,
            joint: joint::joint_kernels::from_module(module.clone())?,
            assignments: assignments::assignment_kernels::from_module(module.clone())?,
            carried: carried::carried_kernels::from_module(module.clone())?,
            relation: relation::relation_kernels::from_module(module.clone())?,
            dec_forms: dec_forms::dec_form_kernels::from_module(module.clone())?,
            decomposition: decomposition::decomposition_kernels::from_module(module)?,
            nightstream,
        })
    }
}
