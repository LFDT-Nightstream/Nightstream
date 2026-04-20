//! Owns the terminal chunk-step Spartan setup/prove/verify path used only by
//! `ivc_snark`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use ff::Field as _;
use serde::Deserialize;
use spartan2::bellpepper::r1cs::{SpartanShape as _, SpartanWitness as _};
use spartan2::bellpepper::solver::SatisfyingAssignment;
use spartan2::traits::transcript::TranscriptEngineTrait as _;

use super::chunk_step_circuit::{
    build_rv64im_chunk_step_ivc_circuit, chunk_step_ivc_spartan_public_values,
    debug_chunk_step_ivc_constraint_checkpoints, debug_locate_chunk_step_main_relation_stage,
    debug_locate_chunk_step_state_out_claims_stage, rv64im_chunk_step_ivc_cache_key,
};
use super::{
    R1CSSNARKTrait, Rv64imDeciderEngine, Rv64imDeciderProverKey, Rv64imDeciderSnark, Rv64imDeciderVerifierKey, ShapeCS,
    SplitR1CSShape,
};
use crate::rv64im::chunk_step_ivc::{
    build_rv64im_chunk_step_ivc_published_target, Rv64imChunkStepIvcStatement, Rv64imChunkStepIvcWitness,
};
use crate::rv64im::SimpleKernelError;

pub(super) type Rv64imChunkStepIvcSpartanKeyPair = Arc<(Rv64imDeciderProverKey, Rv64imDeciderVerifierKey)>;

static RV64IM_CHUNK_STEP_IVC_SETUP_CACHE: OnceLock<Mutex<HashMap<[u8; 32], Rv64imChunkStepIvcSpartanKeyPair>>> =
    OnceLock::new();

#[derive(Deserialize)]
struct WitnessVectorDecode<Ff> {
    is_small: bool,
    #[serde(rename = "W")]
    w: Vec<Ff>,
    r_w: spartan2::provider::pcs::merkle_mle_pc::HashMleBlind<Rv64imDeciderEngine>,
}

pub(super) fn setup_rv64im_chunk_step_ivc_spartan(
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<(Rv64imDeciderProverKey, Rv64imDeciderVerifierKey), SimpleKernelError> {
    let circuit = build_rv64im_chunk_step_ivc_circuit(statement, witness)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider prepare failed: {err}")))?;
    Rv64imDeciderSnark::setup(circuit)
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
        Rv64imDeciderSnark::setup(circuit)
            .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider setup failed: {err}")))?,
    );
    cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM terminal decider setup cache poisoned".into()))?
        .insert(cache_key, keys.clone());
    Ok(keys)
}

pub(super) fn prove_rv64im_chunk_step_ivc_spartan(
    pk: &Rv64imDeciderProverKey,
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<Vec<u8>, SimpleKernelError> {
    let circuit = build_rv64im_chunk_step_ivc_circuit(statement, witness)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider prepare failed: {err}")))?;
    let prep = Rv64imDeciderSnark::prep_prove(pk, circuit.clone(), false)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider prepare failed: {err}")))?;
    let proof = Rv64imDeciderSnark::prove(pk, circuit, &prep, false)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider prove failed: {err}")))?;
    bincode::serialize(&proof)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider encode failed: {err}")))
}

pub(super) fn verify_rv64im_chunk_step_ivc_spartan(
    vk: &Rv64imDeciderVerifierKey,
    statement: &Rv64imChunkStepIvcStatement,
    snark_data: &[u8],
) -> Result<(), SimpleKernelError> {
    let published_target = build_rv64im_chunk_step_ivc_published_target(statement)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider verify failed: {err}")))?;
    let proof: Rv64imDeciderSnark = bincode::deserialize(snark_data)
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

pub(super) fn debug_check_rv64im_chunk_step_ivc_spartan_circuit(
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<(), SimpleKernelError> {
    let circuit = build_rv64im_chunk_step_ivc_circuit(statement, witness)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider prepare failed: {err}")))?;
    let shape = ShapeCS::<Rv64imDeciderEngine>::r1cs_shape(&circuit)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider shape build failed: {err}")))?;
    let regular_shape = shape.to_regular_shape();
    let (ck, _) = SplitR1CSShape::commitment_key(&[&shape]).map_err(|err| {
        SimpleKernelError::Bridge(format!("RV64IM terminal decider commitment-key build failed: {err}"))
    })?;
    let mut precommitted = SatisfyingAssignment::<Rv64imDeciderEngine>::shared_witness(&shape, &ck, &circuit, false)
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM terminal decider shared-witness synthesis failed: {err}"
            ))
        })?;
    SatisfyingAssignment::<Rv64imDeciderEngine>::precommitted_witness(&mut precommitted, &shape, &ck, &circuit, false)
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM terminal decider precommitted-witness synthesis failed: {err}"
            ))
        })?;
    let mut transcript =
        <Rv64imDeciderEngine as spartan2::traits::Engine>::TE::new(b"neo.fold.next/rv64im/ivc_snark/debug");
    let (instance, witness) = SatisfyingAssignment::<Rv64imDeciderEngine>::r1cs_instance_and_witness(
        &mut precommitted,
        &shape,
        &ck,
        &circuit,
        false,
        &mut transcript,
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!("RV64IM terminal decider instance/witness build failed: {err}"))
    })?;
    let regular_instance = instance.to_regular_instance().map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM terminal decider regular-instance conversion failed: {err}"
        ))
    })?;
    regular_shape.is_sat(&ck, &regular_instance, &witness).map_err(|err| {
        let witness_fields = match decode_witness_vector(&witness) {
            Ok(fields) => fields,
            Err(decode_err) => {
                return SimpleKernelError::Bridge(format!(
                    "RV64IM terminal decider circuit is unsatisfied: {err}; additionally failed to decode witness vector: {decode_err}"
                ))
            }
        };
        let z = [
            witness_fields,
            vec![<Rv64imDeciderEngine as spartan2::traits::Engine>::Scalar::ONE],
            circuit.expected_public_values(),
        ]
        .concat();
        match regular_shape.multiply_vec(&z) {
            Ok((az, bz, cz)) => {
                if let Some((idx, (az_i, (bz_i, cz_i)))) = az
                    .iter()
                    .zip(bz.iter().zip(cz.iter()))
                    .enumerate()
                    .find(|(_, (az_i, (bz_i, cz_i)))| **az_i * **bz_i != **cz_i)
                {
                    let phase = match debug_chunk_step_ivc_constraint_checkpoints(&circuit) {
                        Ok(checkpoints) => {
                            let (phase_name, phase_row) = checkpoints.phase_for_row(idx);
                            if phase_name == "chunk_step" {
                                match debug_locate_chunk_step_main_relation_stage(&circuit, phase_row) {
                                    Ok(stage) => {
                                        format!("{phase_name} (phase_row={phase_row}, inner_stage={stage})")
                                    }
                                    Err(stage_err) => {
                                        format!("{phase_name} (phase_row={phase_row}, inner_stage_err={stage_err})")
                                    }
                                }
                            } else if phase_name == "state_out_claims" {
                                match debug_locate_chunk_step_state_out_claims_stage(&circuit) {
                                    Ok(detail) => {
                                        format!("{phase_name} (phase_row={phase_row}, detail={detail})")
                                    }
                                    Err(detail_err) => {
                                        format!("{phase_name} (phase_row={phase_row}, detail_err={detail_err})")
                                    }
                                }
                            } else {
                                format!("{phase_name} (phase_row={phase_row})")
                            }
                        }
                        Err(checkpoint_err) => format!("unknown_phase (checkpoint_err={checkpoint_err})"),
                    };
                    SimpleKernelError::Bridge(format!(
                        "RV64IM terminal decider circuit is unsatisfied: {err}; first failing row={idx} in {phase}, az={az_i:?}, bz={bz_i:?}, cz={cz_i:?}"
                    ))
                } else {
                    SimpleKernelError::Bridge(format!(
                        "RV64IM terminal decider circuit is unsatisfied: {err}"
                    ))
                }
            }
            Err(mul_err) => SimpleKernelError::Bridge(format!(
                "RV64IM terminal decider circuit is unsatisfied: {err}; additionally failed to compute residuals: {mul_err}"
            )),
        }
    })
}

fn decode_witness_vector(
    witness: &spartan2::R1CSWitness<Rv64imDeciderEngine>,
) -> Result<Vec<<Rv64imDeciderEngine as spartan2::traits::Engine>::Scalar>, SimpleKernelError> {
    let WitnessVectorDecode { is_small, w, r_w } =
        bincode::deserialize(&bincode::serialize(witness).map_err(|err| {
            SimpleKernelError::Bridge(format!("RV64IM terminal decider witness encode failed: {err}"))
        })?)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider witness decode failed: {err}")))?;
    let _ = is_small;
    let _ = r_w;
    Ok(w)
}
