use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use neo_transcript::{Poseidon2Transcript, Transcript};

use super::commitment::terminal_committed_boundary_public_values;
use super::types::{
    DirectCcsTerminalCommittedKeyPair, DirectCcsTerminalCommittedPerf, DirectCcsTerminalCommittedProof,
    DirectCcsTerminalCommittedRelation, SimpleKernelError,
};
use crate::construction2::Construction2PublicBoundary;
use crate::construction2_terminal::TerminalPrivateColumnEncoding;
use crate::spartan_backend::{
    NeoFoldDeciderProverKey, NeoFoldDeciderSnark, NeoFoldDeciderVerifierKey, R1CSSNARKTrait, SpartanF,
};

static DIRECT_CCS_TERMINAL_COMMITTED_SETUP_CACHE: OnceLock<
    Mutex<HashMap<[u8; 32], DirectCcsTerminalCommittedKeyPair>>,
> = OnceLock::new();

pub(crate) fn setup_direct_ccs_terminal_committed_relation(
    relation: &DirectCcsTerminalCommittedRelation,
    perf: DirectCcsTerminalCommittedPerf,
) -> Result<
    (
        NeoFoldDeciderProverKey,
        NeoFoldDeciderVerifierKey,
        DirectCcsTerminalCommittedPerf,
    ),
    SimpleKernelError,
> {
    let circuit = relation.committed_circuit();
    let (pk, vk) = NeoFoldDeciderSnark::setup(circuit)
        .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal committed-step setup failed: {err}")))?;
    let mut perf = perf;
    perf.sizes = pk.sizes();
    perf.nnz = pk.shape_debug_stats().total_nnz;
    Ok((pk, vk, perf))
}

pub(crate) fn setup_direct_ccs_terminal_committed_relation_cached(
    relation: &DirectCcsTerminalCommittedRelation,
    perf: DirectCcsTerminalCommittedPerf,
) -> Result<DirectCcsTerminalCommittedKeyPair, SimpleKernelError> {
    let cache_key = direct_terminal_committed_setup_cache_key(relation, &perf);
    let cache = DIRECT_CCS_TERMINAL_COMMITTED_SETUP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(keys) = cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("direct terminal committed-step setup cache poisoned".into()))?
        .get(&cache_key)
        .cloned()
    {
        return Ok(keys);
    }

    let (pk, vk, perf) = setup_direct_ccs_terminal_committed_relation(relation, perf)?;
    let keys = DirectCcsTerminalCommittedKeyPair {
        prover: Arc::new(pk),
        verifier: Arc::new(vk),
        perf,
    };
    cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("direct terminal committed-step setup cache poisoned".into()))?
        .insert(cache_key, keys.clone());
    Ok(keys)
}

fn direct_terminal_committed_setup_cache_key(
    relation: &DirectCcsTerminalCommittedRelation,
    perf: &DirectCcsTerminalCommittedPerf,
) -> [u8; 32] {
    let assignment = &relation.assignment;
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/direct_ccs/terminal_committed_setup_cache");
    tr.append_message(
        b"neo.fold.next/direct_ccs/terminal_committed_setup_cache/version",
        b"v1",
    );
    tr.append_u64s(
        b"neo.fold.next/direct_ccs/terminal_committed_setup_cache/shape",
        &[
            perf.constraints as u64,
            perf.public_inputs as u64,
            perf.committed_width as u64,
            perf.commitment_words as u64,
            perf.source_values as u64,
            perf.source_bit_values as u64,
            perf.source_u32_values as u64,
            perf.source_u64_values as u64,
            assignment.terminal_public_values.len() as u64,
            assignment.r2_public_values.len() as u64,
            assignment.witness_values.len() as u64,
            relation.public_boundary.commitment_d as u64,
            relation.public_boundary.commitment_kappa as u64,
        ],
    );
    let encodings = assignment
        .layout
        .source_encodings
        .iter()
        .map(|encoding| match encoding {
            TerminalPrivateColumnEncoding::UnusedPadding => 0,
            TerminalPrivateColumnEncoding::Bit => 1,
            TerminalPrivateColumnEncoding::U32 => 32,
            TerminalPrivateColumnEncoding::U64 => 64,
        })
        .collect::<Vec<_>>();
    tr.append_u64s(
        b"neo.fold.next/direct_ccs/terminal_committed_setup_cache/source_encodings",
        &encodings,
    );
    tr.digest32()
}

pub(crate) fn prove_direct_ccs_terminal_committed_relation(
    pk: &NeoFoldDeciderProverKey,
    relation: &DirectCcsTerminalCommittedRelation,
) -> Result<(DirectCcsTerminalCommittedProof, f64), SimpleKernelError> {
    let circuit = relation.committed_circuit();
    let prep = NeoFoldDeciderSnark::prep_prove(pk, circuit.clone(), false)
        .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal committed-step prepare failed: {err}")))?;
    let (proof, perf) = NeoFoldDeciderSnark::prove_with_perf(pk, circuit, &prep, false)
        .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal committed-step prove failed: {err}")))?;
    let snark_data = bincode::serialize(&proof).map_err(|err| {
        SimpleKernelError::Bridge(format!("direct terminal committed-step proof encoding failed: {err}"))
    })?;
    Ok((DirectCcsTerminalCommittedProof { snark_data }, perf.pcs_prove_ms))
}

pub(crate) fn verify_direct_ccs_terminal_committed_relation(
    vk: &NeoFoldDeciderVerifierKey,
    expected_terminal_public_values: &[SpartanF],
    expected_public_boundary: &Construction2PublicBoundary,
    proof: &DirectCcsTerminalCommittedProof,
) -> Result<(), SimpleKernelError> {
    let snark: NeoFoldDeciderSnark = bincode::deserialize(&proof.snark_data).map_err(|err| {
        SimpleKernelError::Bridge(format!("direct terminal committed-step proof decoding failed: {err}"))
    })?;
    let public_values = snark
        .verify(vk)
        .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal committed-step verify failed: {err}")))?;
    let expected_public_values =
        terminal_committed_expected_public_values(expected_terminal_public_values, expected_public_boundary);
    if public_values != expected_public_values.as_slice() {
        return Err(SimpleKernelError::Bridge(
            "direct terminal committed-step public IO is not exactly terminal F' values followed by the Construction-2 boundary"
                .into(),
        ));
    }
    Ok(())
}

fn terminal_committed_expected_public_values(
    expected_terminal_public_values: &[SpartanF],
    expected_public_boundary: &Construction2PublicBoundary,
) -> Vec<SpartanF> {
    let boundary_values = terminal_committed_boundary_public_values(expected_public_boundary);
    let mut expected_public_values = Vec::with_capacity(expected_terminal_public_values.len() + boundary_values.len());
    expected_public_values.extend_from_slice(expected_terminal_public_values);
    expected_public_values.extend(boundary_values);
    expected_public_values
}
