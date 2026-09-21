//! Device evaluation of circuit-owned rows, without a lifecycle-crate dependency.

use std::sync::Arc;

use neo_ccs::{Mat, V1_1Evaluations};
use neo_math::{F, K};
use neo_reductions::{
    optimized_engine::{PaperJointOracleBackend, PaperJointOracleInput, PaperJointRoundOracle},
    superneo_eval::SuperneoEvalCache,
    PiCcsError,
};

#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
use crate::session::{MetalJointMatrixPlan, MetalPaperJointOracle};
use crate::{oracle_error, MetalActivity, MetalError, MetalSession};

/// Retains the device session and matrix plan across folds of the same circuit.
pub struct MetalRowProver {
    session: MetalSession,
    #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
    plan: Option<MetalJointMatrixPlan>,
}

impl MetalRowProver {
    pub fn new() -> Result<Self, MetalError> {
        Ok(Self {
            session: MetalSession::new()?,
            #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
            plan: None,
        })
    }

    pub fn activity(&self) -> MetalActivity {
        self.session.activity()
    }

    #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
    fn prepare(&mut self, cache: Arc<SuperneoEvalCache>) -> Result<(), MetalError> {
        if self
            .plan
            .as_ref()
            .is_none_or(|plan| !plan.matches(cache.as_ref()))
        {
            self.plan = Some(self.session.prepare_joint_matrix_plan(cache)?);
        }
        Ok(())
    }

    /// Compute every supplied child opening on the device. Unsupported shapes
    /// return an error; callers do not receive an implicit host fallback.
    pub fn child_openings(
        &mut self,
        cache: Arc<SuperneoEvalCache>,
        witnesses: &[Mat<F>],
        point: &[K],
        assignment_width: usize,
    ) -> Result<Vec<V1_1Evaluations<K>>, PiCcsError> {
        #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
        {
            self.prepare(cache).map_err(oracle_error)?;
            self.session
                .eval_joint_dec_openings(
                    self.plan.as_ref().expect("prepared matrix plan"),
                    witnesses,
                    point,
                    assignment_width,
                )
                .map_err(oracle_error)?
                .ok_or_else(|| {
                    oracle_error(MetalError::Shape(
                        "device child openings are unavailable for this shape",
                    ))
                })
        }
        #[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
        {
            let _ = (cache, witnesses, point, assignment_width);
            Err(oracle_error(MetalError::Unavailable))
        }
    }
}

impl PaperJointOracleBackend for MetalRowProver {
    fn create<'a>(
        &'a mut self,
        input: PaperJointOracleInput<'a>,
    ) -> Result<Box<dyn PaperJointRoundOracle + 'a>, PiCcsError> {
        #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
        {
            self.prepare(Arc::clone(&input.cache))
                .map_err(oracle_error)?;
            Ok(Box::new(MetalPaperJointOracle::new(
                &self.session,
                self.plan.as_ref().expect("prepared matrix plan"),
                input,
            )?))
        }
        #[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
        {
            let _ = input;
            Err(oracle_error(MetalError::Unavailable))
        }
    }
}
