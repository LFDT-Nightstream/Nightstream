//! Owns the terminal chunk-step Spartan setup/prove/verify path used only by
//! `ivc_snark`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use spartan2::traits::snark::R1CSSNARKTrait;

use crate::rv64im::chunk_step_ivc::{
    build_rv64im_chunk_step_ivc_published_target, Rv64imChunkStepIvcStatement, Rv64imChunkStepIvcWitness,
};
use crate::rv64im::main_relation_spartan::{
    build_rv64im_chunk_step_ivc_circuit, chunk_step_ivc_spartan_public_values, rv64im_chunk_step_ivc_cache_key,
    Rv64imSpartan2DeciderProverKey, Rv64imSpartan2DeciderSnark, Rv64imSpartan2DeciderVerifierKey,
};
use crate::rv64im::SimpleKernelError;

pub(super) type Rv64imChunkStepIvcSpartanKeyPair =
    Arc<(Rv64imSpartan2DeciderProverKey, Rv64imSpartan2DeciderVerifierKey)>;

static RV64IM_CHUNK_STEP_IVC_SETUP_CACHE: OnceLock<Mutex<HashMap<[u8; 32], Rv64imChunkStepIvcSpartanKeyPair>>> =
    OnceLock::new();

pub(super) fn setup_rv64im_chunk_step_ivc_spartan(
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<(Rv64imSpartan2DeciderProverKey, Rv64imSpartan2DeciderVerifierKey), SimpleKernelError> {
    let circuit = build_rv64im_chunk_step_ivc_circuit(statement, witness)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider prepare failed: {err}")))?;
    Rv64imSpartan2DeciderSnark::setup(circuit)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider setup failed: {err}")))
}

pub(super) fn setup_rv64im_chunk_step_ivc_spartan_cached(
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<Rv64imChunkStepIvcSpartanKeyPair, SimpleKernelError> {
    let circuit = build_rv64im_chunk_step_ivc_circuit(statement, witness)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider prepare failed: {err}")))?;
    let cache_key = rv64im_chunk_step_ivc_cache_key(&circuit)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider setup failed: {err}")))?;
    let cache = RV64IM_CHUNK_STEP_IVC_SETUP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(keys) = cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM terminal decider setup cache poisoned".into()))?
        .get(&cache_key)
        .cloned()
    {
        return Ok(keys);
    }

    let keys = Arc::new(
        Rv64imSpartan2DeciderSnark::setup(circuit)
            .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider setup failed: {err}")))?,
    );
    cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM terminal decider setup cache poisoned".into()))?
        .insert(cache_key, keys.clone());
    Ok(keys)
}

pub(super) fn prove_rv64im_chunk_step_ivc_spartan(
    pk: &Rv64imSpartan2DeciderProverKey,
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<Vec<u8>, SimpleKernelError> {
    let circuit = build_rv64im_chunk_step_ivc_circuit(statement, witness)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider prepare failed: {err}")))?;
    let prep = Rv64imSpartan2DeciderSnark::prep_prove(pk, circuit.clone(), false)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider prepare failed: {err}")))?;
    let proof = Rv64imSpartan2DeciderSnark::prove(pk, circuit, &prep, false)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider prove failed: {err}")))?;
    bincode::serialize(&proof)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider encode failed: {err}")))
}

pub(super) fn verify_rv64im_chunk_step_ivc_spartan(
    vk: &Rv64imSpartan2DeciderVerifierKey,
    statement: &Rv64imChunkStepIvcStatement,
    snark_data: &[u8],
) -> Result<(), SimpleKernelError> {
    let published_target = build_rv64im_chunk_step_ivc_published_target(statement)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider verify failed: {err}")))?;
    let proof: Rv64imSpartan2DeciderSnark = bincode::deserialize(snark_data)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider decode failed: {err}")))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider verify failed: {err}")))?;
    if public_values != chunk_step_ivc_spartan_public_values(&published_target) {
        return Err(SimpleKernelError::Bridge(
            "RV64IM terminal decider public IO mismatch".into(),
        ));
    }
    Ok(())
}
