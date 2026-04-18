//! Owns circuit-derived fixed-step transcript snapshots for recursive payloads.
//!
//! This module replays the chunk NIFS.V body directly and then appends the
//! local chunk-done marker expected by the carried transcript snapshot.

use bellpepper_core::{
    test_cs::TestConstraintSystem, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable,
};
use neo_ccs::{CcsStructure, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_reductions::engines::utils::{
    bind_header_and_instance_digest_with_digest, bind_me_inputs as bind_me_inputs_native, build_dims_and_policy,
    digest_ccs_matrices_with_sparse_cache,
};
use neo_transcript::Poseidon2Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use spartan2::provider::goldi::F as SpartanF;
use std::io::{self, Write};
use std::time::Instant;

use super::chunk_step_recursive::Rv64imMainRecursionFPrimePayload;
use super::recursive_cover::{alloc_recursive_cover_claims, alloc_recursive_cover_state};
use super::{
    alloc_const_field_values, append_chunk_meta, debug_locate_rv64im_main_relation_chunk_stage,
    debug_profile_rv64im_main_relation_chunk_stage_progress, digest32_as_spartan_fields,
    synthesize_rv64im_chunk_nifs_verifier_body_with_synthetic_chunk_relation_io, Rv64imChunkBoundaryPlan,
    Rv64imClaimBundle, CHUNK_META_RAW_TAG, STEP_INDEX_RAW_TAG,
};
use crate::rv64im::final_relation::{Rv64imChunkFoldTranscriptSnapshot, RV64IM_CHUNK_DONE_RAW_TAG};
use crate::rv64im::kernel::{
    rv64im_cached_root_main_lane_context, rv64im_cached_root_main_lane_optimized_cache, SimpleKernelError,
};
use crate::rv64im::main_relation_circuit::pi_ccs::{
    bind_header_and_instance_digest, bind_me_inputs_with_native_claims,
};
use crate::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use crate::rv64im::main_relation_trace::{Rv64imMainCircuitChunkCover, Rv64imMainCircuitChunkReplaySurface};

struct WitnessOnlyCS<Scalar> {
    inputs: usize,
    aux: usize,
    namespace_depth: usize,
    _marker: core::marker::PhantomData<Scalar>,
}

impl<Scalar> WitnessOnlyCS<Scalar> {
    fn new() -> Self {
        Self {
            inputs: 1,
            aux: 0,
            namespace_depth: 0,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<Scalar: ff::PrimeField + Send> ConstraintSystem<Scalar> for WitnessOnlyCS<Scalar> {
    type Root = Self;

    fn new() -> Self {
        Self::new()
    }

    fn alloc<FN, A, AR>(&mut self, _annotation: A, value: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let _ = value()?;
        let var = Variable::new_unchecked(Index::Aux(self.aux));
        self.aux += 1;
        Ok(var)
    }

    fn alloc_input<FN, A, AR>(&mut self, _annotation: A, value: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let _ = value()?;
        let var = Variable::new_unchecked(Index::Input(self.inputs));
        self.inputs += 1;
        Ok(var)
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _annotation: A, _a: LA, _b: LB, _c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    {
    }

    fn push_namespace<NR, N>(&mut self, _name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        self.namespace_depth += 1;
    }

    fn pop_namespace(&mut self) {
        assert!(self.namespace_depth > 0);
        self.namespace_depth -= 1;
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}

pub(super) fn derive_rv64im_fixed_transcript_out_from_chunk_body(
    payload: &Rv64imMainRecursionFPrimePayload,
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    replay_chunk: &Rv64imMainCircuitChunkReplaySurface,
    live_state_in_claims: &[CeClaim<neo_ajtai::Commitment, F, K>],
    terminal_final_claims: &[CeClaim<neo_ajtai::Commitment, F, K>],
    terminal_handle_in: [u8; 32],
    trace_prefix: Option<&str>,
) -> Result<Rv64imChunkFoldTranscriptSnapshot, SimpleKernelError> {
    let (params, _, structure) = rv64im_cached_root_main_lane_context()?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM fixed transcript dims failed: {err}")))?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()))
        .try_into()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM fixed transcript matrix digest length mismatch".into()))?;
    derive_fixed_transcript_out_from_parts(
        params,
        structure,
        dims,
        &mat_digest,
        &payload.chunk_cover,
        transcript_in,
        replay_chunk,
        &payload.state_in_claims,
        live_state_in_claims,
        terminal_final_claims,
        terminal_handle_in,
        payload.boundary_plan,
        trace_prefix,
    )
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn emit_debug_timing(trace_prefix: Option<&str>, label: &str, elapsed_ms: f64) {
    if let Some(prefix) = trace_prefix {
        eprintln!("{prefix}.{label}={elapsed_ms:.2}ms");
        let _ = io::stderr().flush();
    }
}

fn should_debug_profile_fixed_transcript_stages() -> bool {
    matches!(
        std::env::var("NS_DEBUG_FIXED_TRANSCRIPT_STAGES").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn derive_fixed_transcript_out_from_parts(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: neo_reductions::engines::utils::Dims,
    mat_digest: &[Goldilocks; 4],
    cover_chunk: &Rv64imMainCircuitChunkCover,
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    replay_chunk: &Rv64imMainCircuitChunkReplaySurface,
    state_in_claims: &[CeClaim<neo_ajtai::Commitment, F, K>],
    live_state_in_claims: &[CeClaim<neo_ajtai::Commitment, F, K>],
    terminal_final_claims: &[CeClaim<neo_ajtai::Commitment, F, K>],
    terminal_handle_in: [u8; 32],
    boundary_plan: Rv64imChunkBoundaryPlan,
    trace_prefix: Option<&str>,
) -> Result<Rv64imChunkFoldTranscriptSnapshot, SimpleKernelError> {
    let total_started = Instant::now();
    let started = Instant::now();
    let mut cs = WitnessOnlyCS::<SpartanF>::new();
    emit_debug_timing(trace_prefix, "cs_init", elapsed_ms(started));
    let started = Instant::now();
    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "fixed_transcript_state_in"),
        state_in_claims,
        transcript_in,
        terminal_handle_in,
        "fixed_transcript_state_in",
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM fixed transcript state allocation failed: {err}")))?;
    emit_debug_timing(trace_prefix, "alloc_state_in", elapsed_ms(started));
    let started = Instant::now();
    let transcript_values = transcript_in
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut replayed_transcript = Poseidon2TranscriptCircuit::from_state(
        state_in_var.transcript_state.clone(),
        transcript_values,
        transcript_in.absorbed,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM fixed transcript state import failed: {err}")))?;
    emit_debug_timing(trace_prefix, "import_transcript_state", elapsed_ms(started));
    let started = Instant::now();
    let live_state_in_vars = alloc_recursive_cover_claims(
        &mut cs.namespace(|| "fixed_transcript_live_state_in"),
        state_in_claims,
        "fixed_transcript_live_state_in",
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM fixed transcript live-state allocation failed: {err}")))?;
    emit_debug_timing(trace_prefix, "alloc_live_state_in_claims", elapsed_ms(started));
    let started = Instant::now();
    let carried_claims = Rv64imClaimBundle::from_effective_claims(
        live_state_in_vars
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    emit_debug_timing(trace_prefix, "bundle_effective_claims", elapsed_ms(started));
    if should_debug_profile_fixed_transcript_stages() && replay_chunk.handoff.public_chunk.start_index == 0 {
        debug_profile_fixed_transcript_chunk_body(
            params,
            structure,
            dims,
            mat_digest,
            cover_chunk,
            transcript_in,
            replay_chunk,
            state_in_claims,
            live_state_in_claims,
            terminal_final_claims,
            terminal_handle_in,
            boundary_plan,
        )?;
    }
    let started = Instant::now();
    if let Err(err) = synthesize_rv64im_chunk_nifs_verifier_body_with_synthetic_chunk_relation_io(
        params,
        structure,
        dims,
        mat_digest,
        terminal_final_claims,
        &mut cs,
        0,
        cover_chunk,
        replay_chunk,
        &mut replayed_transcript,
        carried_claims,
        Some(live_state_in_claims),
        boundary_plan,
    ) {
        if let Err(prefix_err) = debug_check_fixed_transcript_prefix(
            params,
            structure,
            dims,
            mat_digest,
            transcript_in,
            replay_chunk,
            state_in_claims,
            live_state_in_claims,
        ) {
            return Err(prefix_err);
        }
        if let Err(stage_err) = debug_locate_fixed_transcript_chunk_stage(
            params,
            structure,
            dims,
            mat_digest,
            cover_chunk,
            transcript_in,
            replay_chunk,
            state_in_claims,
            live_state_in_claims,
            terminal_final_claims,
            boundary_plan,
        ) {
            return Err(stage_err);
        }
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM fixed transcript chunk replay failed: {err}"
        )));
    }
    emit_debug_timing(trace_prefix, "synthesize_chunk_body", elapsed_ms(started));
    let started = Instant::now();
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| "fixed_transcript_chunk_done"),
            &[
                SpartanF::from_canonical_u64(RV64IM_CHUNK_DONE_RAW_TAG),
                SpartanF::from_canonical_u64(1),
            ],
        )
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM fixed transcript chunk_done failed: {err}")))?;
    emit_debug_timing(trace_prefix, "append_chunk_done", elapsed_ms(started));
    emit_debug_timing(trace_prefix, "satisfaction_check", 0.0);
    emit_debug_timing(trace_prefix, "total", elapsed_ms(total_started));
    Ok(Rv64imChunkFoldTranscriptSnapshot {
        state: replayed_transcript
            .state_values()
            .map(|value| F::from_u64(value.to_canonical_u64())),
        absorbed: replayed_transcript.absorbed(),
    })
}

fn debug_profile_fixed_transcript_chunk_body(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: neo_reductions::engines::utils::Dims,
    mat_digest: &[Goldilocks; 4],
    cover_chunk: &Rv64imMainCircuitChunkCover,
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    replay_chunk: &Rv64imMainCircuitChunkReplaySurface,
    state_in_claims: &[CeClaim<neo_ajtai::Commitment, F, K>],
    live_state_in_claims: &[CeClaim<neo_ajtai::Commitment, F, K>],
    terminal_final_claims: &[CeClaim<neo_ajtai::Commitment, F, K>],
    terminal_handle_in: [u8; 32],
    boundary_plan: Rv64imChunkBoundaryPlan,
) -> Result<(), SimpleKernelError> {
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "fixed_transcript_profile_state_in"),
        state_in_claims,
        transcript_in,
        terminal_handle_in,
        "fixed_transcript_profile_state_in",
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM fixed transcript profile state allocation failed: {err}"
        ))
    })?;
    let transcript_values = transcript_in
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut transcript = Poseidon2TranscriptCircuit::from_state(
        state_in_var.transcript_state.clone(),
        transcript_values,
        transcript_in.absorbed,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM fixed transcript profile state import failed: {err}")))?;
    let synthetic_chunk_relation_digest = alloc_const_field_values(
        &mut cs.namespace(|| "fixed_transcript_profile_chunk_relation_digest"),
        &digest32_as_spartan_fields(replay_chunk.handoff.chunk_relation_digest),
        "fixed_transcript_profile_chunk_relation_digest",
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM fixed transcript profile relation digest allocation failed: {err}"
        ))
    })?;
    let mut synthetic_chunk_relation_cursor = 0usize;
    let live_state_in_vars = alloc_recursive_cover_claims(
        &mut cs.namespace(|| "fixed_transcript_profile_live_state_in"),
        state_in_claims,
        "fixed_transcript_profile_live_state_in",
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM fixed transcript profile live-state allocation failed: {err}"
        ))
    })?;
    let carried_claims = Rv64imClaimBundle::from_effective_claims(
        live_state_in_vars
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    debug_profile_rv64im_main_relation_chunk_stage_progress(
        params,
        structure,
        dims,
        mat_digest,
        terminal_final_claims,
        &mut cs,
        0,
        cover_chunk,
        replay_chunk,
        &synthetic_chunk_relation_digest,
        &mut synthetic_chunk_relation_cursor,
        &mut transcript,
        carried_claims,
        Some(live_state_in_claims),
        boundary_plan,
        false,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM fixed transcript stage profile failed: {err}")))?;
    let _ = live_state_in_claims;
    Ok(())
}

fn compare_transcript_state(
    label: &str,
    circuit: &Poseidon2TranscriptCircuit,
    native: &Poseidon2Transcript,
) -> Result<(), SimpleKernelError> {
    if circuit.absorbed() != native.absorbed() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM fixed transcript mismatch after {label}: absorbed {} != {}",
            circuit.absorbed(),
            native.absorbed()
        )));
    }
    let native_state = native.state();
    for (idx, (circuit_value, native_value)) in circuit
        .state_values()
        .iter()
        .zip(native_state.iter())
        .enumerate()
    {
        let expected = SpartanF::from_canonical_u64(native_value.as_canonical_u64());
        if *circuit_value != expected {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM fixed transcript mismatch after {label} at limb {idx}: {} != {}",
                circuit_value.to_canonical_u64(),
                expected.to_canonical_u64()
            )));
        }
    }
    Ok(())
}

fn enforce_transcript_state_against_native(
    cs: &mut TestConstraintSystem<SpartanF>,
    label: &str,
    circuit: &Poseidon2TranscriptCircuit,
    native: &Poseidon2Transcript,
) -> Result<(), SimpleKernelError> {
    let expected = native
        .state()
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    circuit
        .enforce_state_values(
            &mut cs.namespace(|| format!("{label}_state_eq")),
            &expected,
            &format!("{label}_state_eq"),
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM fixed transcript {label} state enforcement failed: {err}"
            ))
        })?;
    if !cs.is_satisfied() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM fixed transcript latent mismatch after {label}: {}",
            cs.which_is_unsatisfied().unwrap_or("unknown constraint")
        )));
    }
    Ok(())
}

fn append_chunk_meta_native(transcript: &mut Poseidon2Transcript, replay_chunk: &Rv64imMainCircuitChunkReplaySurface) {
    if replay_chunk.handoff.public_chunk.steps.len() == 1 {
        transcript.append_fields_raw(&[
            F::from_u64(STEP_INDEX_RAW_TAG),
            F::from_u64(replay_chunk.handoff.public_chunk.start_index as u64),
        ]);
    } else {
        transcript.append_fields_raw(&[
            F::from_u64(CHUNK_META_RAW_TAG),
            F::from_u64(replay_chunk.handoff.public_chunk.start_index as u64),
            F::from_u64(replay_chunk.handoff.public_chunk.steps.len() as u64),
        ]);
    }
}

fn debug_check_fixed_transcript_prefix(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: neo_reductions::engines::utils::Dims,
    mat_digest: &[Goldilocks; 4],
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    replay_chunk: &Rv64imMainCircuitChunkReplaySurface,
    state_in_claims: &[CeClaim<neo_ajtai::Commitment, F, K>],
    live_state_in_claims: &[CeClaim<neo_ajtai::Commitment, F, K>],
) -> Result<(), SimpleKernelError> {
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "fixed_transcript_prefix_state_in"),
        live_state_in_claims,
        transcript_in,
        replay_chunk.handoff.bridge_handoff_digest,
        "fixed_transcript_prefix_state_in",
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!("RV64IM fixed transcript prefix state allocation failed: {err}"))
    })?;
    let transcript_values = transcript_in
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut circuit = Poseidon2TranscriptCircuit::from_state(
        state_in_var.transcript_state.clone(),
        transcript_values,
        transcript_in.absorbed,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM fixed transcript prefix state import failed: {err}")))?;
    let mut native = Poseidon2Transcript::from_state_and_absorbed(transcript_in.state, transcript_in.absorbed);
    compare_transcript_state("init", &circuit, &native)?;
    enforce_transcript_state_against_native(&mut cs, "init", &circuit, &native)?;

    append_chunk_meta(
        &mut cs.namespace(|| "fixed_transcript_prefix_chunk_meta"),
        &mut circuit,
        &replay_chunk.handoff,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM fixed transcript prefix meta failed: {err}")))?;
    append_chunk_meta_native(&mut native, replay_chunk);
    compare_transcript_state("chunk_meta", &circuit, &native)?;
    enforce_transcript_state_against_native(&mut cs, "chunk_meta", &circuit, &native)?;

    bind_header_and_instance_digest(
        &mut cs.namespace(|| "fixed_transcript_prefix_bind_header"),
        &mut circuit,
        params,
        structure.n,
        structure.m,
        structure.t(),
        &structure.f,
        dims,
        mat_digest,
        &replay_chunk
            .handoff
            .public_chunk_instance_digest
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM fixed transcript prefix header binding failed: {err}")))?;
    bind_header_and_instance_digest_with_digest(
        &mut native,
        params,
        structure,
        dims,
        &mat_digest.map(|value| F::from_u64(value.as_canonical_u64())),
        &replay_chunk.handoff.public_chunk_instance_digest,
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM fixed transcript prefix native header binding failed: {err}"
        ))
    })?;
    compare_transcript_state("bind_header", &circuit, &native)?;
    enforce_transcript_state_against_native(&mut cs, "bind_header", &circuit, &native)?;

    let live_state_in_vars = alloc_recursive_cover_claims(
        &mut cs.namespace(|| "fixed_transcript_prefix_live_state_in"),
        state_in_claims,
        "fixed_transcript_prefix_live_state_in",
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM fixed transcript prefix live-state allocation failed: {err}"
        ))
    })?;
    let carried_claims = Rv64imClaimBundle::from_effective_claims(
        live_state_in_vars
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    bind_me_inputs_with_native_claims(
        &mut cs.namespace(|| "fixed_transcript_prefix_bind_me_inputs"),
        &mut circuit,
        carried_claims.effective_claims(),
        live_state_in_claims,
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM fixed transcript prefix circuit ME binding failed: {err}"
        ))
    })?;
    bind_me_inputs_native(&mut native, live_state_in_claims).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM fixed transcript prefix native ME binding failed: {err}"
        ))
    })?;
    compare_transcript_state("bind_me_inputs", &circuit, &native)?;
    enforce_transcript_state_against_native(&mut cs, "bind_me_inputs", &circuit, &native)?;
    Ok(())
}

fn debug_locate_fixed_transcript_chunk_stage(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: neo_reductions::engines::utils::Dims,
    mat_digest: &[Goldilocks; 4],
    cover_chunk: &Rv64imMainCircuitChunkCover,
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    replay_chunk: &Rv64imMainCircuitChunkReplaySurface,
    state_in_claims: &[CeClaim<neo_ajtai::Commitment, F, K>],
    live_state_in_claims: &[CeClaim<neo_ajtai::Commitment, F, K>],
    terminal_final_claims: &[CeClaim<neo_ajtai::Commitment, F, K>],
    boundary_plan: Rv64imChunkBoundaryPlan,
) -> Result<(), SimpleKernelError> {
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "fixed_transcript_stage_state_in"),
        live_state_in_claims,
        transcript_in,
        replay_chunk.handoff.bridge_handoff_digest,
        "fixed_transcript_stage_state_in",
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!("RV64IM fixed transcript staged state allocation failed: {err}"))
    })?;
    let transcript_values = transcript_in
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut transcript = Poseidon2TranscriptCircuit::from_state(
        state_in_var.transcript_state.clone(),
        transcript_values,
        transcript_in.absorbed,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM fixed transcript staged state import failed: {err}")))?;
    let synthetic_chunk_relation_digest = alloc_const_field_values(
        &mut cs.namespace(|| "fixed_transcript_stage_chunk_relation_digest"),
        &digest32_as_spartan_fields(replay_chunk.handoff.chunk_relation_digest),
        "fixed_transcript_stage_chunk_relation_digest",
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM fixed transcript staged relation digest allocation failed: {err}"
        ))
    })?;
    let mut synthetic_chunk_relation_cursor = 0usize;
    let live_state_in_vars = alloc_recursive_cover_claims(
        &mut cs.namespace(|| "fixed_transcript_stage_live_state_in"),
        state_in_claims,
        "fixed_transcript_stage_live_state_in",
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM fixed transcript staged live-state allocation failed: {err}"
        ))
    })?;
    let carried_claims = Rv64imClaimBundle::from_effective_claims(
        live_state_in_vars
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    debug_locate_rv64im_main_relation_chunk_stage(
        params,
        structure,
        dims,
        mat_digest,
        terminal_final_claims,
        &mut cs,
        0,
        cover_chunk,
        replay_chunk,
        &synthetic_chunk_relation_digest,
        &mut synthetic_chunk_relation_cursor,
        &mut transcript,
        carried_claims,
        boundary_plan,
        false,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM fixed transcript staged chunk replay failed: {err}")))?;
    Ok(())
}
