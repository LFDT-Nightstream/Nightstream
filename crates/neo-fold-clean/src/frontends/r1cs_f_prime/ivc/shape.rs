//! Shared Stage 2 field-R1CS shape helpers.
//!
//! Owns the application and semantic-state wiring reused by the Nebula
//! frontend. It is not a Stage 1 production relation.
//!
//! Does not own: paper semantics, selective low-norm lowering, or permission to
//! remove any emitted row.
//!
//! | Stage path | Mathematical obligation | Rust owner |
//! |---|---|---|
//! | `fprime.{base,recursive}.finalize.application` | Derive the application semantic digest | `enforce_semantic_digests` |
//! | `fprime.{base,recursive}.finalize.semantic_links` | Bind application semantics to the transition output | `bind_semantic_state` |

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::R1csIvcError;
use crate::engine::r1cs_circuit::{enforce_poseidon2_hash, Lc, R1csBuilder, Var};
use crate::frontends::f_prime::recursive_plan::{
    semantic_state_app_public_header, semantic_state_field_header, RecursiveStepImagePlan,
};
use crate::paper::construction2::SemanticStateMode;
use crate::paper::digest::{digest32_as_fields, StateXOutDigestMode};
use crate::paper::f_prime::digest_circuit::alloc_constant;
use crate::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use crate::paper::f_prime::r1cs::FPrimeStepOutput;

#[derive(Clone, Copy)]
pub(crate) struct SemanticValues {
    pub input: Option<[F; 4]>,
    pub output: Option<[F; 4]>,
}

pub(crate) fn semantic_values(plan: &RecursiveStepImagePlan, assignment: &[F]) -> Result<SemanticValues, R1csIvcError> {
    let Some(state) = plan.state_x_out.as_ref() else {
        return Ok(SemanticValues {
            input: None,
            output: None,
        });
    };
    let input = (!state.semantic_state_in_var_indices.is_empty())
        .then(|| semantic_state_digest_for_assignment(assignment, &state.semantic_state_in_var_indices));
    let output = if !state.semantic_state_out_var_indices.is_empty() {
        Some(semantic_state_digest_for_assignment(
            assignment,
            &state.semantic_state_out_var_indices,
        ))
    } else if !state.app_public_input_var_indices.is_empty() || !state.app_public_input_bit_var_indices.is_empty() {
        let preimage = app_public_semantic_preimage_for_assignment(plan, assignment)?;
        Some(encode_poseidon_trace(&preimage).digest_native)
    } else {
        None
    };
    Ok(SemanticValues { input, output })
}

fn semantic_state_digest_for_assignment(assignment: &[F], indices: &[usize]) -> [F; 4] {
    let values = indices
        .iter()
        .map(|&index| assignment[index])
        .collect::<Vec<_>>();
    encode_poseidon_trace(&crate::frontends::f_prime::recursive_plan::build_semantic_state_preimage_fields(&values))
        .digest_native
}

fn app_public_semantic_preimage_for_assignment(
    plan: &RecursiveStepImagePlan,
    assignment: &[F],
) -> Result<Vec<F>, R1csIvcError> {
    let Some(state) = plan.state_x_out.as_ref() else {
        return Ok(Vec::new());
    };
    let mut preimage = semantic_state_app_public_header(
        state.app_public_input_var_indices.len(),
        state.app_public_input_bit_var_indices.len(),
    );
    preimage.extend(
        state
            .app_public_input_var_indices
            .iter()
            .map(|&index| assignment[index]),
    );
    for chunk in state.app_public_input_bit_var_indices.chunks(64) {
        let mut packed = 0u64;
        for (bit, &index) in chunk.iter().enumerate() {
            let value = assignment[index];
            if value == F::ZERO {
                continue;
            }
            if value == F::ONE {
                packed |= 1 << bit;
                continue;
            }
            return Err(R1csIvcError::PackedPublicInputNotBit { index, value });
        }
        preimage.push(F::from_u64(packed));
    }
    Ok(preimage)
}

pub(crate) fn digest_mode(plan: &RecursiveStepImagePlan) -> StateXOutDigestMode {
    let mode = super::super::semantic_state_mode_for_plan(plan);
    match mode {
        SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
        SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
    }
}

pub(crate) struct SemanticWires {
    input: Option<[Var; 4]>,
    output: Option<[Var; 4]>,
}

pub(crate) fn enforce_semantic_digests(
    builder: &mut R1csBuilder,
    plan: &RecursiveStepImagePlan,
    assignment: &[F],
    app_vars: &[Var],
) -> Result<SemanticWires, R1csIvcError> {
    let Some(state) = plan.state_x_out.as_ref() else {
        return Ok(SemanticWires {
            input: None,
            output: None,
        });
    };
    let input = (!state.semantic_state_in_var_indices.is_empty()).then(|| {
        semantic_field_digest_wires(
            builder,
            state
                .semantic_state_in_var_indices
                .iter()
                .map(|&index| app_vars[index]),
        )
    });
    let output = if !state.semantic_state_out_var_indices.is_empty() {
        Some(semantic_field_digest_wires(
            builder,
            state
                .semantic_state_out_var_indices
                .iter()
                .map(|&index| app_vars[index]),
        ))
    } else if !state.app_public_input_var_indices.is_empty() || !state.app_public_input_bit_var_indices.is_empty() {
        let mut values: Vec<Var> = state
            .app_public_input_var_indices
            .iter()
            .map(|&index| app_vars[index])
            .collect();
        for chunk in state.app_public_input_bit_var_indices.chunks(64) {
            let mut packed_value = 0u64;
            let mut packed_lc = Lc::zero();
            let mut coefficient = F::ONE;
            for (bit, &index) in chunk.iter().enumerate() {
                if assignment[index] == F::ONE {
                    packed_value |= 1u64 << bit;
                }
                packed_lc.add_term(app_vars[index], coefficient);
                coefficient += coefficient;
            }
            let packed = builder.alloc(F::from_u64(packed_value));
            builder.enforce_eq(&Lc::from_var(packed), &packed_lc);
            values.push(packed);
        }
        Some(semantic_app_public_digest_wires(
            builder,
            state.app_public_input_var_indices.len(),
            state.app_public_input_bit_var_indices.len(),
            values,
        ))
    } else {
        None
    };
    Ok(SemanticWires { input, output })
}

fn semantic_field_digest_wires(builder: &mut R1csBuilder, values: impl IntoIterator<Item = Var>) -> [Var; 4] {
    let values: Vec<Var> = values.into_iter().collect();
    semantic_digest_wires(builder, semantic_state_field_header(values.len()), values)
}

fn semantic_app_public_digest_wires(
    builder: &mut R1csBuilder,
    field_count: usize,
    bit_count: usize,
    values: impl IntoIterator<Item = Var>,
) -> [Var; 4] {
    semantic_digest_wires(
        builder,
        semantic_state_app_public_header(field_count, bit_count),
        values,
    )
}

fn semantic_digest_wires(builder: &mut R1csBuilder, header: Vec<F>, values: impl IntoIterator<Item = Var>) -> [Var; 4] {
    let mut preimage: Vec<Var> = header
        .into_iter()
        .map(|value| alloc_constant(builder, value))
        .collect();
    preimage.extend(values);
    enforce_poseidon2_hash(builder, &preimage)
}

pub(crate) fn bind_semantic_state(
    builder: &mut R1csBuilder,
    plan: &RecursiveStepImagePlan,
    output: &FPrimeStepOutput,
    semantic: SemanticWires,
    base: bool,
) {
    if let Some(input) = semantic.input {
        bind_digest(builder, &output.state_in.semantic_state_digest, &input);
    }
    if let Some(out) = semantic.output {
        bind_digest(builder, &output.state_out.semantic_state_digest, &out);
    }
    if base {
        if output.state_in.nebula.is_some() {
            // The Nebula base circuit binds this state lane to the raw
            // initial semantic digest inside its verifier-owned program
            // binding, then carries the recomputed binding digest.
            return;
        }
        let anchor = plan
            .state_x_out
            .as_ref()
            .and_then(|state| state.initial_semantic_state_digest_anchor)
            .unwrap_or_else(crate::paper::digest::empty_semantic_state_digest);
        let anchor = digest32_as_fields(anchor);
        for lane in 0..4 {
            builder.enforce_eq(
                &Lc::from_var(output.state_in.semantic_state_digest[lane]),
                &Lc::from_const(anchor[lane]),
            );
        }
    }
}

fn bind_digest(builder: &mut R1csBuilder, left: &[Var; 4], right: &[Var; 4]) {
    for lane in 0..4 {
        builder.enforce_eq(&Lc::from_var(left[lane]), &Lc::from_var(right[lane]));
    }
}

pub(crate) fn pin_app_constant(plan: &RecursiveStepImagePlan) -> bool {
    let Some(state) = plan.state_x_out.as_ref() else {
        return plan.app_private_var_widths.iter().any(|&width| width < 64);
    };
    // One semantic role can use z[0] as an ordinary value. Zero roles or
    // the same lane on both transition sides select the conventional
    // constant-one role, which must be constrained directly.
    let mut zero_semantic_roles = usize::from(state.semantic_state_in_var_indices.contains(&0))
        + usize::from(state.semantic_state_out_var_indices.contains(&0));
    if state.semantic_state_out_var_indices.is_empty()
        && (state.app_public_input_var_indices.contains(&0) || state.app_public_input_bit_var_indices.contains(&0))
    {
        zero_semantic_roles += 1;
    }
    plan.app_private_var_widths.iter().any(|&width| width < 64)
        || !state.app_public_input_bit_var_indices.is_empty()
        || (state.initial_semantic_state_digest_anchor.is_some() && zero_semantic_roles != 1)
}
