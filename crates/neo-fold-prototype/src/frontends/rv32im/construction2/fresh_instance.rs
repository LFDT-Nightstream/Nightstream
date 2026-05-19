//! Owns Construction-2 fresh-instance builders for the RV32IM F' path.

use std::io::{self, Write};
use std::time::Instant;

use crate::rv32im::construction2::default::build_rv32im_main_recursion_construction2_default_full_width_from_ccs_shape;
use crate::rv32im::f_prime::{Rv32imEncodedPublicInput, Rv32imMainRecursionFPrimeAdvice, Rv32imVerifierKeyFs};
use crate::rv32im::SimpleKernelError;

use super::{
    build_rv32im_main_recursion_construction2_default_pair,
    build_rv32im_main_recursion_construction2_f_prime_ccs_shape, build_rv32im_main_recursion_construction2_x_i,
    elapsed_ms, rv32im_main_recursion_construction2_x_only_placeholder,
    validate_rv32im_main_recursion_construction2_advice,
    validate_rv32im_main_recursion_construction2_input_fresh_instance, Rv32imMainRecursionConstruction2FreshInstance,
};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Rv32imMainRecursionConstruction2FreshInstanceBuildPerf {
    pub pack_image_ms: f64,
    pub commit_ms: f64,
    pub total_ms: f64,
}

pub fn build_rv32im_main_recursion_construction2_default_fresh_instance(
    vk_fs: &Rv32imVerifierKeyFs,
    full_width: usize,
) -> Result<Rv32imMainRecursionConstruction2FreshInstance, SimpleKernelError> {
    Ok(
        build_rv32im_main_recursion_construction2_default_pair(vk_fs, full_width)?
            .u_perp()
            .clone(),
    )
}

pub(crate) fn build_rv32im_main_recursion_construction2_fresh_instance_with_input_and_x_i(
    advice: &Rv32imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv32imMainRecursionConstruction2FreshInstance,
    x_i: Rv32imEncodedPublicInput,
) -> Result<Rv32imMainRecursionConstruction2FreshInstance, SimpleKernelError> {
    Ok(
        build_rv32im_main_recursion_construction2_fresh_instance_with_input_and_x_i_with_perf(
            advice,
            current_input_fresh_instance,
            x_i,
        )?
        .0,
    )
}

pub(crate) fn build_rv32im_main_recursion_construction2_fresh_instance_with_input_and_x_i_with_perf(
    advice: &Rv32imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv32imMainRecursionConstruction2FreshInstance,
    x_i: Rv32imEncodedPublicInput,
) -> Result<
    (
        Rv32imMainRecursionConstruction2FreshInstance,
        Rv32imMainRecursionConstruction2FreshInstanceBuildPerf,
    ),
    SimpleKernelError,
> {
    validate_rv32im_main_recursion_construction2_advice(advice)?;
    let total_started = Instant::now();
    let mut perf = Rv32imMainRecursionConstruction2FreshInstanceBuildPerf::default();
    let started = Instant::now();
    validate_rv32im_main_recursion_construction2_input_fresh_instance(advice, current_input_fresh_instance)?;
    perf.pack_image_ms = elapsed_ms(started);
    let started = Instant::now();
    let fresh_instance = rv32im_main_recursion_construction2_x_only_placeholder(x_i);
    perf.commit_ms = elapsed_ms(started);
    perf.total_ms = elapsed_ms(total_started);
    Ok((fresh_instance, perf))
}

pub(crate) fn debug_trace_build_rv32im_main_recursion_construction2_fresh_instance_with_input_and_x_i(
    advice: &Rv32imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv32imMainRecursionConstruction2FreshInstance,
    x_i: Rv32imEncodedPublicInput,
    trace_prefix: &str,
) -> Result<Rv32imMainRecursionConstruction2FreshInstance, SimpleKernelError> {
    let started = Instant::now();
    let (fresh, perf) = build_rv32im_main_recursion_construction2_fresh_instance_with_input_and_x_i_with_perf(
        advice,
        current_input_fresh_instance,
        x_i,
    )?;
    eprintln!(
        "{trace_prefix}.validate_input_fresh_instance={:.2}ms",
        perf.pack_image_ms
    );
    eprintln!(
        "{trace_prefix}.x_only_non_authoritative_commitment_placeholder={:.2}ms",
        perf.commit_ms
    );
    eprintln!("{trace_prefix}.total={:.2}ms", elapsed_ms(started));
    let _ = io::stderr().flush();
    Ok(fresh)
}

pub fn build_rv32im_main_recursion_construction2_fresh_instance_with_input(
    advice: &Rv32imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv32imMainRecursionConstruction2FreshInstance,
) -> Result<Rv32imMainRecursionConstruction2FreshInstance, SimpleKernelError> {
    build_rv32im_main_recursion_construction2_fresh_instance_with_input_and_x_i(
        advice,
        current_input_fresh_instance,
        build_rv32im_main_recursion_construction2_x_i(advice)?,
    )
}

pub fn build_rv32im_main_recursion_construction2_fresh_instance(
    advice: &Rv32imMainRecursionFPrimeAdvice,
) -> Result<Rv32imMainRecursionConstruction2FreshInstance, SimpleKernelError> {
    let shape = build_rv32im_main_recursion_construction2_f_prime_ccs_shape(core::slice::from_ref(advice))?;
    if advice.chunk_count_in() > 0 {
        return Err(SimpleKernelError::Bridge(
            "RV32IM native Construction-2 fresh instance builder for an inductive F' step still requires the prior-step output u_i = (c_i, x_i) to be threaded explicitly; use the explicit input-threaded builder"
                .into(),
        ));
    }
    build_rv32im_main_recursion_construction2_default_fresh_instance(
        advice.verifier_key_fs(),
        build_rv32im_main_recursion_construction2_default_full_width_from_ccs_shape(
            &build_rv32im_main_recursion_construction2_f_prime_ccs_shape(core::slice::from_ref(advice))?,
        )?,
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV32IM native Construction-2 base-case fresh instance build failed for canonical u_perp (shape digest {:?}): {err}",
            shape.expected_digest(),
        ))
    })
}
