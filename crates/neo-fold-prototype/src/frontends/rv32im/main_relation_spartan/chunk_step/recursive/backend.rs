//! Owns backend relation packaging for the RV32IM recursive F' step.

use std::io::{self, Write};
use std::time::Instant;

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsWitness};
use neo_math::F;

use crate::rv32im::chunk::step_ivc::Rv32imChunkStepIvcRelation;
use crate::rv32im::construction2::Rv32imMainRecursionConstruction2FreshInstance;
use crate::rv32im::f_prime::{
    build_rv32im_main_recursion_backend_statement_from_parts_with_vk_fs, build_rv32im_main_recursion_f_prime_advices,
    evaluate_rv32im_main_recursion_f_prime_advice, Rv32imMainRecursionFPrimeAdvice,
};
use crate::rv32im::SimpleKernelError;

use super::super::super::Rv32imMainRecursionStepSpartanStatement;
use super::{
    build_rv32im_main_recursion_f_prime_payload_with_trace, build_rv32im_main_recursion_f_prime_payloads,
    build_rv32im_main_recursion_step_spartan_shape_from_advices, Rv32imMainRecursionFPrimePayload,
    Rv32imMainRecursionStepSpartanShape,
};

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn emit_debug_timing(trace_prefix: Option<&str>, label: &str, elapsed_ms: f64) {
    if let Some(prefix) = trace_prefix {
        eprintln!("{prefix}.{label}={elapsed_ms:.2}ms");
        let _ = io::stderr().flush();
    }
}

#[derive(Clone, Debug)]
pub struct Rv32imMainRecursionFPrimeBackendRelation {
    pub f_prime_advice: Rv32imMainRecursionFPrimeAdvice,
    pub spartan_statement: Rv32imMainRecursionStepSpartanStatement,
    pub construction2_u_next: Rv32imMainRecursionConstruction2FreshInstance,
    pub payload: Rv32imMainRecursionFPrimePayload,
}

#[derive(Clone, Debug, Default)]
pub struct Rv32imMainRecursionFPrimeBackendRelationBuildPerf {
    pub spartan_shape_ms: f64,
    pub payloads_ms: f64,
    pub statement_build_ms: f64,
    pub semantics_check_ms: f64,
    pub total_ms: f64,
    pub relation_count: usize,
}

pub(crate) fn build_rv32im_main_recursion_step_spartan_statement(
    f_prime_advice: &Rv32imMainRecursionFPrimeAdvice,
) -> Result<Rv32imMainRecursionStepSpartanStatement, SimpleKernelError> {
    Ok(build_rv32im_main_recursion_step_spartan_statement_and_construction2_output(f_prime_advice)?.0)
}

fn build_rv32im_main_recursion_step_spartan_statement_and_construction2_output(
    f_prime_advice: &Rv32imMainRecursionFPrimeAdvice,
) -> Result<
    (
        Rv32imMainRecursionStepSpartanStatement,
        Rv32imMainRecursionConstruction2FreshInstance,
    ),
    SimpleKernelError,
> {
    let step_image = evaluate_rv32im_main_recursion_f_prime_advice(f_prime_advice)?;
    let backend_statement = build_rv32im_main_recursion_backend_statement_from_parts_with_vk_fs(
        f_prime_advice.verifier_key_fs(),
        step_image.chunk_count(),
        step_image.folded_accumulator_digest(),
        *step_image.z_next(),
    );
    Ok((
        backend_statement.native_statement(),
        step_image.construction2_u_next().clone(),
    ))
}

pub fn debug_check_rv32im_main_recursion_f_prime_backend_relation_semantics(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<(), SimpleKernelError> {
    if !backend_relation
        .payload
        .matches_explicit_semantics(&backend_relation.f_prime_advice)
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM recursive-step backend relation payload explicit z/pc semantics drifted from the native F' advice"
                .into(),
        ));
    }
    let canonical_statement = build_rv32im_main_recursion_step_spartan_statement(&backend_relation.f_prime_advice)?;
    if backend_relation.spartan_statement != canonical_statement {
        return Err(SimpleKernelError::Bridge(
            "RV32IM recursive-step backend relation requires the canonical per-step Spartan statement derived from native F'"
                .into(),
        ));
    }
    Ok(())
}

pub fn build_rv32im_main_recursion_f_prime_backend_relations(
    relations: &[Rv32imChunkStepIvcRelation],
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
) -> Result<Vec<Rv32imMainRecursionFPrimeBackendRelation>, SimpleKernelError> {
    let f_prime_advices = build_rv32im_main_recursion_f_prime_advices(relations)?;
    let payloads = build_rv32im_main_recursion_f_prime_payloads(&f_prime_advices, spartan_shape)
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    if f_prime_advices.len() != payloads.len() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM recursive-step backend builder produced mismatched step and payload counts".into(),
        ));
    }
    f_prime_advices
        .into_iter()
        .zip(payloads)
        .map(|(f_prime_advice, payload)| {
            let (spartan_statement, construction2_u_next) =
                build_rv32im_main_recursion_step_spartan_statement_and_construction2_output(&f_prime_advice)?;
            let backend_relation = Rv32imMainRecursionFPrimeBackendRelation {
                f_prime_advice,
                spartan_statement,
                construction2_u_next,
                payload,
            };
            debug_check_rv32im_main_recursion_f_prime_backend_relation_semantics(&backend_relation)?;
            Ok(backend_relation)
        })
        .collect()
}

pub fn build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape(
    relations: &[Rv32imChunkStepIvcRelation],
) -> Result<
    (
        Rv32imMainRecursionStepSpartanShape,
        Vec<Rv32imMainRecursionFPrimeBackendRelation>,
    ),
    SimpleKernelError,
> {
    let advices = build_rv32im_main_recursion_f_prime_advices(relations)?;
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(relations, &advices)
}

pub fn build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(
    relations: &[Rv32imChunkStepIvcRelation],
    advices: &[Rv32imMainRecursionFPrimeAdvice],
) -> Result<
    (
        Rv32imMainRecursionStepSpartanShape,
        Vec<Rv32imMainRecursionFPrimeBackendRelation>,
    ),
    SimpleKernelError,
> {
    Ok(
        build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices_and_perf(
            relations, advices, None,
        )?
        .0,
    )
}

pub fn build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices_and_perf(
    relations: &[Rv32imChunkStepIvcRelation],
    advices: &[Rv32imMainRecursionFPrimeAdvice],
    trace_prefix: Option<&str>,
) -> Result<
    (
        (
            Rv32imMainRecursionStepSpartanShape,
            Vec<Rv32imMainRecursionFPrimeBackendRelation>,
        ),
        Rv32imMainRecursionFPrimeBackendRelationBuildPerf,
    ),
    SimpleKernelError,
> {
    let total_started = Instant::now();
    let started = Instant::now();
    let spartan_shape = build_rv32im_main_recursion_step_spartan_shape_from_advices(relations, advices)
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let spartan_shape_ms = elapsed_ms(started);
    emit_debug_timing(trace_prefix, "spartan_shape", spartan_shape_ms);
    let started = Instant::now();
    let mut payloads = Vec::with_capacity(advices.len());
    for (step_index, advice) in advices.iter().enumerate() {
        let payload_trace_prefix = trace_prefix.map(|prefix| format!("{prefix}.step_{step_index}_payload"));
        let payload_started = Instant::now();
        let payload = build_rv32im_main_recursion_f_prime_payload_with_trace(
            advice,
            &spartan_shape,
            payload_trace_prefix.as_deref(),
        )
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
        emit_debug_timing(
            trace_prefix,
            &format!("step_{step_index}_payload_total"),
            elapsed_ms(payload_started),
        );
        payloads.push(payload);
    }
    let payloads_ms = elapsed_ms(started);
    emit_debug_timing(trace_prefix, "payloads", payloads_ms);
    if advices.len() != payloads.len() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM recursive-step backend builder produced mismatched step and payload counts".into(),
        ));
    }
    let mut statement_build_ms = 0.0;
    let mut semantics_check_ms = 0.0;
    let backend_relations = advices
        .iter()
        .cloned()
        .zip(payloads)
        .enumerate()
        .map(|(step_index, (f_prime_advice, payload))| {
            let started = Instant::now();
            let (spartan_statement, construction2_u_next) =
                build_rv32im_main_recursion_step_spartan_statement_and_construction2_output(&f_prime_advice)?;
            let statement_ms = elapsed_ms(started);
            statement_build_ms += statement_ms;
            emit_debug_timing(
                trace_prefix,
                &format!("step_{step_index}_statement_build"),
                statement_ms,
            );
            let backend_relation = Rv32imMainRecursionFPrimeBackendRelation {
                f_prime_advice,
                spartan_statement,
                construction2_u_next,
                payload,
            };
            let started = Instant::now();
            debug_check_rv32im_main_recursion_f_prime_backend_relation_semantics(&backend_relation)?;
            let semantics_ms = elapsed_ms(started);
            semantics_check_ms += semantics_ms;
            emit_debug_timing(
                trace_prefix,
                &format!("step_{step_index}_semantics_check"),
                semantics_ms,
            );
            Ok(backend_relation)
        })
        .collect::<Result<Vec<_>, SimpleKernelError>>()?;
    let perf = Rv32imMainRecursionFPrimeBackendRelationBuildPerf {
        spartan_shape_ms,
        payloads_ms,
        statement_build_ms,
        semantics_check_ms,
        total_ms: elapsed_ms(total_started),
        relation_count: advices.len(),
    };
    emit_debug_timing(trace_prefix, "statement_build_total", statement_build_ms);
    emit_debug_timing(trace_prefix, "semantics_check_total", semantics_check_ms);
    emit_debug_timing(trace_prefix, "total", perf.total_ms);
    Ok(((spartan_shape, backend_relations), perf))
}

pub fn debug_trace_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(
    relations: &[Rv32imChunkStepIvcRelation],
    advices: &[Rv32imMainRecursionFPrimeAdvice],
    trace_prefix: &str,
) -> Result<
    (
        (
            Rv32imMainRecursionStepSpartanShape,
            Vec<Rv32imMainRecursionFPrimeBackendRelation>,
        ),
        Rv32imMainRecursionFPrimeBackendRelationBuildPerf,
    ),
    SimpleKernelError,
> {
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices_and_perf(
        relations,
        advices,
        Some(trace_prefix),
    )
}

fn ccs_claim_matches(left: &CcsClaim<Commitment, F>, right: &CcsClaim<Commitment, F>) -> bool {
    left.c == right.c && left.x == right.x && left.m_in == right.m_in
}

fn ccs_witness_matches(left: &CcsWitness<F>, right: &CcsWitness<F>) -> bool {
    left.w == right.w
        && left.Z.rows() == right.Z.rows()
        && left.Z.cols() == right.Z.cols()
        && left.Z.as_slice() == right.Z.as_slice()
}

pub fn debug_check_rv32im_chunk_step_recursive_effective_chunk_trace_matches_native(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<(), SimpleKernelError> {
    let native_trace = backend_relation.f_prime_advice.main_circuit_chunk_trace()?;
    let effective_replay_surface = backend_relation.payload.effective_chunk_replay_surface()?;
    let native_replay_surface = native_trace.replay_surface()?;

    if effective_replay_surface.handoff.public_chunk.start_index
        != native_replay_surface.handoff.public_chunk.start_index
        || effective_replay_surface.handoff.public_chunk.steps.len()
            != native_replay_surface.handoff.public_chunk.steps.len()
        || effective_replay_surface
            .handoff
            .public_chunk_instance_digest
            != native_replay_surface.handoff.public_chunk_instance_digest
        || effective_replay_surface.handoff.public_chunk_digest != native_replay_surface.handoff.public_chunk_digest
        || effective_replay_surface.handoff.bridge_handoff_digest != native_replay_surface.handoff.bridge_handoff_digest
        || effective_replay_surface.handoff.chunk_relation_digest != native_replay_surface.handoff.chunk_relation_digest
        || effective_replay_surface.fresh_claims.len() != native_replay_surface.fresh_claims.len()
        || effective_replay_surface.pi_ccs.ccs_outputs != native_replay_surface.pi_ccs.ccs_outputs
        || effective_replay_surface.pi_ccs.replay_proof != native_replay_surface.pi_ccs.replay_proof
        || effective_replay_surface.pi_rlc.parent != native_replay_surface.pi_rlc.parent
        || effective_replay_surface.pi_dec.children != native_replay_surface.pi_dec.children
        || effective_replay_surface.pi_ccs.public_challenges.alpha
            != native_replay_surface.pi_ccs.public_challenges.alpha
        || effective_replay_surface.pi_ccs.public_challenges.beta_a
            != native_replay_surface.pi_ccs.public_challenges.beta_a
        || effective_replay_surface.pi_ccs.public_challenges.beta_r
            != native_replay_surface.pi_ccs.public_challenges.beta_r
        || effective_replay_surface.pi_ccs.public_challenges.beta_m
            != native_replay_surface.pi_ccs.public_challenges.beta_m
        || effective_replay_surface.pi_ccs.public_challenges.gamma
            != native_replay_surface.pi_ccs.public_challenges.gamma
        || effective_replay_surface.pi_ccs.row_chals != native_replay_surface.pi_ccs.row_chals
        || effective_replay_surface.pi_ccs.alpha_prime != native_replay_surface.pi_ccs.alpha_prime
        || effective_replay_surface.pi_ccs.s_col != native_replay_surface.pi_ccs.s_col
        || effective_replay_surface.pi_ccs.alpha_prime_nc != native_replay_surface.pi_ccs.alpha_prime_nc
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM effective chunk replay surface recovered from the recursive payload does not match the native trace"
                .into(),
        ));
    }

    for (effective, native) in effective_replay_surface
        .fresh_claims
        .iter()
        .zip(native_replay_surface.fresh_claims.iter())
    {
        if !ccs_claim_matches(effective, native) {
            return Err(SimpleKernelError::Bridge(
                "RV32IM effective fresh claim recovered from the recursive payload does not match the native trace"
                    .into(),
            ));
        }
    }

    for (effective, native) in backend_relation
        .payload
        .fresh_witnesses
        .iter()
        .take(backend_relation.payload.step_shape.fresh_witness_count as usize)
        .zip(native_trace.fresh_witnesses.iter())
    {
        if !ccs_witness_matches(effective, native) {
            return Err(SimpleKernelError::Bridge(
                "RV32IM effective fresh witness recovered from the recursive payload does not match the native trace"
                    .into(),
            ));
        }
    }

    Ok(())
}
