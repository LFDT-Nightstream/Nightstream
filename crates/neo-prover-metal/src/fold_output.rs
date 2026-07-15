//! Cross-fold ownership for resident Pi_DEC child witnesses.

use std::any::Any;
use std::sync::{Arc, Mutex};

use neo_fold_clean::paper::nifs::{
    DeferredNifsProofMaterializer, DeferredNifsRunningMaterializer, Error, NifsProof, NifsProofCarrier,
    NifsRunningCarrier,
};
use neo_fold_clean::paper::{pi_ccs, pi_dec, pi_rlc};
use neo_fold_clean::RunningInstance;
use neo_math::F;

use crate::MetalResidentWitnessSnapshot;

pub(crate) struct MetalDeferredNifsProof {
    state: Mutex<MetalDeferredNifsProofState>,
}

enum MetalDeferredNifsProofState {
    Pending {
        pi_ccs: pi_ccs::Proof,
        output: Arc<MetalFoldOutput>,
    },
    Ready(NifsProof),
    Failed,
}

impl MetalDeferredNifsProof {
    fn new(pi_ccs: pi_ccs::Proof, output: Arc<MetalFoldOutput>) -> Self {
        Self {
            state: Mutex::new(MetalDeferredNifsProofState::Pending { pi_ccs, output }),
        }
    }
}

impl DeferredNifsProofMaterializer for MetalDeferredNifsProof {
    fn materialize(&self) -> Result<NifsProof, Error> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| backend_unavailable("deferred Metal NIFS proof lock poisoned"))?;
        if let MetalDeferredNifsProofState::Ready(proof) = &*state {
            return Ok(proof.clone());
        }
        let pending = std::mem::replace(&mut *state, MetalDeferredNifsProofState::Failed);
        let MetalDeferredNifsProofState::Pending { pi_ccs, output } = pending else {
            return Err(backend_unavailable(
                "deferred Metal NIFS proof materialization already failed",
            ));
        };
        let proof = NifsProof {
            pi_ccs,
            pi_rlc: pi_rlc::Proof {
                combined: output.parent_authority()?.clone(),
            },
            pi_dec: pi_dec::Proof {
                children: output.running.claims.clone(),
            },
        };
        *state = MetalDeferredNifsProofState::Ready(proof.clone());
        Ok(proof)
    }
}

pub(crate) fn metal_proof_carrier(
    pi_ccs: pi_ccs::Proof,
    pi_rlc: pi_rlc::Proof,
    pi_dec: pi_dec::Proof,
    output: Arc<MetalFoldOutput>,
) -> Result<NifsProofCarrier, Error> {
    if output.parent_authority()? != &pi_rlc.combined || output.running.claims != pi_dec.children {
        return Err(backend_unavailable(
            "Metal proof and running carriers disagree on post-fold authority",
        ));
    }
    Ok(NifsProofCarrier::deferred(Arc::new(MetalDeferredNifsProof::new(
        pi_ccs, output,
    ))))
}

pub(crate) struct MetalFoldOutput {
    running: RunningInstance,
    resident_id: Option<u64>,
    witness_snapshot: Option<MetalResidentWitnessSnapshot>,
    parent_digest: [F; 4],
}

impl MetalFoldOutput {
    pub(crate) fn new(
        running: RunningInstance,
        resident_id: Option<u64>,
        witness_snapshot: Option<MetalResidentWitnessSnapshot>,
        parent_digest: [F; 4],
    ) -> Self {
        Self {
            running,
            resident_id,
            witness_snapshot,
            parent_digest,
        }
    }

    fn parent_authority(&self) -> Result<&neo_fold_clean::CeClaim, Error> {
        self.running
            .parent_authority
            .as_ref()
            .ok_or_else(|| backend_unavailable("Metal fold output is missing its Pi_RLC parent"))
    }

    fn materialize(&self) -> Result<RunningInstance, Error> {
        let mut running = self.running.clone();
        if let Some(snapshot) = self.witness_snapshot.as_ref() {
            running.witnesses = snapshot
                .materialize()
                .map_err(|_| backend_unavailable("materialize resident Metal witnesses"))?;
        }
        Ok(running)
    }
}

pub(crate) struct MetalRunningCarrier {
    output: Arc<MetalFoldOutput>,
}

impl MetalRunningCarrier {
    pub(crate) fn new(output: Arc<MetalFoldOutput>) -> Self {
        Self { output }
    }

    pub(crate) fn resident_id(&self) -> Option<u64> {
        self.output.resident_id
    }

    pub(crate) fn parent_digest(&self) -> [F; 4] {
        self.output.parent_digest
    }
}

impl DeferredNifsRunningMaterializer for MetalRunningCarrier {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn materialize(&self) -> Result<RunningInstance, Error> {
        self.output.materialize()
    }

    fn materialize_prover_input(&self) -> Result<RunningInstance, Error> {
        Ok(self.output.running.clone())
    }
}

pub(crate) fn metal_running_carrier(carrier: Option<&NifsRunningCarrier>) -> Option<&MetalRunningCarrier> {
    let NifsRunningCarrier::Deferred(materializer) = carrier? else {
        return None;
    };
    materializer.as_any().downcast_ref()
}

fn backend_unavailable(reason: &'static str) -> Error {
    Error::BackendUnavailable {
        backend: "metal",
        reason,
    }
}
