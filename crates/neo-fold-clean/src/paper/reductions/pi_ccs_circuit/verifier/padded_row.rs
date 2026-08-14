//! One-joint recursive verifier for `PaddedRowIdentity`.
//!
//! This module mirrors the selected native transcript and Section 7.3
//! terminal equation. The only rectangular specialization is the virtual
//! identity matrix `[I; 0]` on one zero-padded row cube.
//!
//! Owns: the selected transcript schedule, joint dimensions, round chain,
//! and paper terminal equality.
//!
//! Does not own: claim allocation, digest encoding, native proving, or
//! matrix evaluation.
//!
//! Emits constraints: one fixed-width SumCheck and its terminal relation.
//!
//! | Phase | Constraint family |
//! | --- | --- |
//! | prefix | public statement, alpha, and gamma binding |
//! | rounds | fixed-width one-joint SumCheck |
//! | terminal | CCS, norm, and carried-evaluation equality |

use super::*;

use crate::engine::r1cs_circuit::field_ext::{alloc_klc, enforce_k_mul, klc_add, KLc};
use crate::engine::r1cs_circuit::sumcheck::{enforce_eq_k, enforce_sumcheck_round, gamma_powers};

const PUBLIC_INPUT_TAG: u64 = 40;
const PROTOCOL_VERSION: u64 = 2;
const STATEMENT_TAG: u64 = 41;
const ALPHA_TAG: u64 = 42;
const GAMMA_TAG: u64 = 43;
const ROUND_TAG: u64 = 45;
const ROUND_CHALLENGE_TAG: u64 = 46;
const COMPACT_BINDING_TAG: u64 = 47;

type Dims = neo_reductions::engines::pi_ccs_joint::JointDims;

pub(super) fn enforce(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    cfg: &PiCcsVerifierConfig<'_>,
    msg: &PiCcsVerifierMessages<'_>,
    matrix_digest_wires: Option<[Var; 4]>,
) -> Result<PiCcsVerifierResult, Error> {
    builder.begin_encoding_stage(stage::ROOT);
    let dims = dimensions(cfg, msg.fresh.len(), msg.running.len())?;
    validate_messages(cfg, msg, dims)?;

    builder.begin_encoding_stage(stage::ALLOCATIONS);
    let allocation_start = builder.rows();
    let fresh_wires = msg
        .fresh
        .iter()
        .map(|claim| alloc_fresh_wires(builder, claim))
        .collect::<Vec<_>>();
    let running_wires = msg
        .running
        .iter()
        .map(|claim| alloc_ce_wires(builder, claim))
        .collect::<Result<Vec<_>, _>>()?;
    let output_wires = msg
        .outputs
        .iter()
        .map(|claim| alloc_ce_wires(builder, claim))
        .collect::<Result<Vec<_>, _>>()?;
    let running_parent_authority_wires = msg
        .running_parent_authority
        .map(|claim| alloc_ce_wires(builder, claim))
        .transpose()?;
    builder.record_row_family(stage::ALLOCATIONS, allocation_start);

    builder.begin_encoding_stage(stage::CANONICALITY);
    let canonicality_start = builder.rows();
    enforce_running_point_consistency(builder, &running_wires)?;
    enforce_claim_canonicality(builder, &running_wires, "running")?;
    enforce_claim_canonicality(builder, &output_wires, "output")?;
    if let Some(parent) = &running_parent_authority_wires {
        enforce_claim_canonicality(builder, std::slice::from_ref(parent), "running parent")?;
    }
    builder.record_row_family(stage::CANONICALITY, canonicality_start);

    builder.begin_encoding_stage(stage::BINDING);
    let binding_start = builder.rows();
    let fresh_digests = fresh_wires
        .iter()
        .map(|fresh| {
            enforce_ccs_claim_digest(
                builder,
                fresh.c_d,
                fresh.c_kappa,
                &fresh.c_data,
                &fresh.x,
                fresh.m_in,
                fresh.adv.as_ref(),
            )
        })
        .collect::<Vec<_>>();
    let running_parent_digest = running_parent_authority_wires
        .as_ref()
        .map(|parent| enforce_accumulator_ce_claim_digest(builder, &accumulator_digest_inputs(parent)))
        .transpose()?;
    let instance_digest = enforce_pi_ccs_instance_digest_parent_authority(
        builder,
        &fresh_digests,
        running_wires.len(),
        running_parent_digest,
    );
    let running_acc_digest = if running_wires.is_empty() {
        AccumulatorHandle::empty()
            .digest_fields()
            .map(|value| alloc_constant_var(builder, value))
    } else {
        let parent = running_parent_authority_wires
            .as_ref()
            .expect("nonempty running family was validated to have a parent");
        let parent_inputs = accumulator_digest_inputs(parent);
        let child_inputs = running_wires
            .iter()
            .map(accumulator_digest_inputs)
            .collect::<Vec<_>>();
        enforce_strict_radix_accumulator_family_digest(builder, cfg.params.b(), &parent_inputs, &child_inputs)?
    };
    builder.record_row_family(stage::BINDING, binding_start);

    builder.begin_encoding_stage(stage::PREFIX);
    let prefix_start = builder.rows();
    let matrix_digest = matrix_digest_wires.unwrap_or_else(|| {
        cfg.matrix_digest
            .map(|value| alloc_constant_var(builder, value))
    });
    absorb_public(
        builder,
        transcript,
        cfg,
        dims,
        matrix_digest,
        instance_digest,
        fresh_wires.len(),
        running_wires.len(),
        running_acc_digest,
    );
    absorb_statement(builder, transcript, cfg, dims, fresh_wires.len(), running_wires.len());
    builder.record_row_family(stage::PREFIX, prefix_start);

    builder.begin_encoding_stage(stage::CHALLENGES);
    let challenge_start = builder.rows();
    let alpha = (0..dims.variables)
        .map(|index| squeeze(builder, transcript, ALPHA_TAG, Some(index)))
        .collect::<Vec<_>>();
    let gamma = squeeze(builder, transcript, GAMMA_TAG, None);
    let powers = gamma_powers(
        builder,
        gamma,
        gamma_power_count(msg.fresh.len(), msg.running.len(), dims),
    );
    let initial = initial_claim(builder, &powers, msg.fresh.len(), &running_wires, dims);
    builder.record_row_family(stage::CHALLENGES, challenge_start);

    builder.begin_encoding_stage(stage::SUMCHECK);
    let sumcheck_start = builder.rows();
    let mut claim = initial;
    let mut point = Vec::with_capacity(dims.variables);
    for (round_index, round) in msg.sumcheck_rounds.iter().enumerate() {
        let coefficients = alloc_k_vec(builder, round);
        let mut fields = Vec::with_capacity(3 + 2 * coefficients.len());
        push_const(builder, &mut fields, ROUND_TAG);
        push_const(builder, &mut fields, round_index as u64);
        push_const(builder, &mut fields, coefficients.len() as u64);
        push_k_vars(&mut fields, &coefficients);
        transcript.append_fields_unframed_vars(builder, &fields);
        let challenge = squeeze(builder, transcript, ROUND_CHALLENGE_TAG, Some(round_index));
        claim = enforce_sumcheck_round(builder, &coefficients, challenge, claim);
        point.push(challenge);
    }
    builder.record_row_family(stage::SUMCHECK, sumcheck_start);

    builder.begin_encoding_stage(stage::TERMINAL);
    let terminal_start = builder.rows();
    bind_outputs(builder, &fresh_wires, &running_wires, &output_wires, &point, dims)?;
    let expected = terminal(
        builder,
        cfg,
        &powers,
        &alpha,
        &point,
        msg.fresh.len(),
        &running_wires,
        &output_wires,
        dims,
    )?;
    enforce_kvar_eq(builder, claim, expected);
    builder.record_row_family(stage::TERMINAL, terminal_start);

    builder.begin_encoding_stage(stage::OUTPUT_TRANSCRIPT);
    let output_transcript_start = builder.rows();
    let fold_digest = transcript.digest_fields(builder);
    enforce_output_fold_digest_matches_header(builder, &output_wires, fold_digest);
    builder.record_row_family(stage::OUTPUT_TRANSCRIPT, output_transcript_start);

    builder.begin_encoding_stage(stage::OUTPUT_DIGEST);
    let output_digest_start = builder.rows();
    let output_digest_inputs = output_wires
        .iter()
        .map(|output| PiCcsOutputMessageDigestInputs { y_ring: &output.y_ring })
        .collect::<Vec<_>>();
    let output_digest_wires = enforce_pi_ccs_outputs_digest(
        builder,
        PiCcsOutputMessageProfile::new(output_wires.len(), dims.matrix_count),
        &output_digest_inputs,
    )?;
    builder.begin_encoding_stage(stage::OUTPUT_MESSAGE_CLAIM);
    for (wire, value) in output_digest_wires.digest.iter().zip(msg.outputs_digest) {
        let claimed = builder.alloc(value);
        enforce_var_eq(builder, *wire, claimed);
    }
    builder.record_row_family(stage::OUTPUT_DIGEST, output_digest_start);

    Ok(PiCcsVerifierResult {
        r_prime: point,
        outputs: output_wires.into_iter().map(public_wires).collect(),
        output_claims_digest: output_digest_wires.digest,
        output_message_preimage: output_digest_wires.preimage,
        fresh_x: fresh_wires.iter().map(|claim| claim.x.clone()).collect(),
        fresh_adv: fresh_wires.iter().map(|claim| claim.adv.clone()).collect(),
        running_c_data: running_wires
            .iter()
            .map(|claim| claim.c_data.clone())
            .collect(),
        running: running_wires.into_iter().map(public_wires).collect(),
        running_parent_authority: running_parent_authority_wires.map(public_wires),
        running_acc_digest,
    })
}

fn dimensions(cfg: &PiCcsVerifierConfig<'_>, fresh_count: usize, running_count: usize) -> Result<Dims, Error> {
    neo_reductions::engines::pi_ccs_joint::build_joint_dims_for_shape(
        cfg.params.inner(),
        cfg.structure.n(),
        cfg.structure.m(),
        cfg.structure.t(),
        cfg.structure.max_degree(),
        fresh_count,
        running_count,
    )
    .map_err(|error| Error::Shape(error.to_string()))
}

fn validate_messages(cfg: &PiCcsVerifierConfig<'_>, msg: &PiCcsVerifierMessages<'_>, dims: Dims) -> Result<(), Error> {
    if msg.fresh.is_empty() {
        return Err(Error::Shape("PaddedRowIdentity requires a fresh source".into()));
    }
    match (msg.running.is_empty(), msg.running_parent_authority) {
        (true, None) | (false, Some(_)) => {}
        (true, Some(_)) => {
            return Err(Error::Shape(
                "empty running accumulator carries a parent authority".into(),
            ));
        }
        (false, None) => {
            return Err(Error::Shape(
                "nonempty running accumulator is missing its parent authority".into(),
            ));
        }
    }
    if msg.outputs.len() != msg.fresh.len() + msg.running.len() {
        return Err(Error::Shape("one-joint output source count mismatch".into()));
    }
    if msg.sumcheck_rounds.len() != dims.variables
        || msg
            .sumcheck_rounds
            .iter()
            .any(|round| round.len() != dims.degree + 1)
    {
        return Err(Error::Shape("one-joint SumCheck message shape mismatch".into()));
    }
    for (index, claim) in msg.fresh.iter().enumerate() {
        validate_fresh_shape(cfg, index, claim)?;
        if claim.m_in % D != 0 {
            return Err(Error::Shape(format!("fresh[{index}] is not a whole-ring paper claim")));
        }
    }
    for (label, claims) in [("running", msg.running), ("output", msg.outputs)] {
        for (index, claim) in claims.iter().enumerate() {
            validate_selected_ce(cfg, &format!("{label}[{index}]"), claim, dims)?;
        }
    }
    if let Some(parent) = msg.running_parent_authority {
        validate_selected_ce(cfg, "running_parent_authority", parent, dims)?;
    }
    Ok(())
}

fn validate_selected_ce(cfg: &PiCcsVerifierConfig<'_>, label: &str, claim: &CeClaim, dims: Dims) -> Result<(), Error> {
    let d_pad = D.next_power_of_two();
    if claim.c.d != D
        || claim.c.kappa != cfg.params.kappa() as usize
        || claim.c.data.len() != D * cfg.params.kappa() as usize
        || claim.m_in > cfg.structure.m()
        || claim.m_in % D != 0
        || claim.X.rows() != D
        || claim.X.cols() != crate::paper::relations::superneo_public_x_cols(claim.m_in)
        || claim.r.len() != dims.variables
        || claim.y_ring.len() != dims.matrix_count
        || claim.ct.len() != dims.matrix_count
    {
        return Err(Error::Shape(format!("{label} does not have the selected CE shape")));
    }
    if claim.y_ring.iter().any(|row| row.len() != d_pad) {
        return Err(Error::Shape(format!(
            "{label} has a noncanonical ring-evaluation width"
        )));
    }
    Ok(())
}

fn enforce_claim_canonicality(builder: &mut R1csBuilder, claims: &[CeClaimWires], label: &str) -> Result<(), Error> {
    for (index, claim) in claims.iter().enumerate() {
        enforce_ct_from_y_ring(builder, &format!("{label}[{index}]"), claim)?;
        enforce_y_ring_padding_zero(builder, claim);
    }
    Ok(())
}

fn enforce_running_point_consistency(builder: &mut R1csBuilder, running: &[CeClaimWires]) -> Result<(), Error> {
    let Some(first) = running.first() else {
        return Ok(());
    };
    for (index, claim) in running.iter().enumerate().skip(1) {
        if claim.r.len() != first.r.len() {
            return Err(Error::Shape(format!("running[{index}] point length mismatch")));
        }
        for (&left, &right) in first.r.iter().zip(&claim.r) {
            enforce_kvar_eq(builder, left, right);
        }
    }
    Ok(())
}

fn absorb_public(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    cfg: &PiCcsVerifierConfig<'_>,
    dims: Dims,
    matrix_digest: [Var; 4],
    instance_digest: [Var; 4],
    fresh_count: usize,
    running_count: usize,
    running_acc_digest: [Var; 4],
) {
    let mut fields = Vec::new();
    for value in [
        PUBLIC_INPUT_TAG,
        PROTOCOL_VERSION,
        dims.variables as u64,
        fresh_count as u64,
        running_count as u64,
        dims.matrix_count as u64,
        D as u64,
        dims.assignment_width as u64,
        dims.row_count as u64,
        dims.degree as u64,
        cfg.structure.n() as u64,
        cfg.structure.m() as u64,
    ] {
        push_const(builder, &mut fields, value);
    }
    fields.extend(matrix_digest);
    push_const(builder, &mut fields, COMPACT_BINDING_TAG);
    fields.extend(instance_digest);
    push_const(builder, &mut fields, running_count as u64);
    push_const(builder, &mut fields, 1);
    fields.extend(running_acc_digest);
    transcript.append_fields_unframed_vars(builder, &fields);
}

fn absorb_statement(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    cfg: &PiCcsVerifierConfig<'_>,
    dims: Dims,
    fresh_count: usize,
    running_count: usize,
) {
    let mut fields = Vec::new();
    for value in [
        STATEMENT_TAG,
        dims.variables as u64,
        fresh_count as u64,
        running_count as u64,
        dims.matrix_count as u64,
        D as u64,
        cfg.structure.max_degree() as u64,
        cfg.structure.polynomial().terms().len() as u64,
    ] {
        push_const(builder, &mut fields, value);
    }
    for term in cfg.structure.polynomial().terms() {
        fields.push(alloc_constant_var(builder, term.coeff));
        fields.push(alloc_constant_var(builder, F::ZERO));
        fields.push(alloc_constant_var(builder, F::ZERO));
        for &exponent in &term.exps {
            push_const(builder, &mut fields, exponent as u64);
        }
    }
    push_const(builder, &mut fields, COMPACT_BINDING_TAG);
    transcript.append_fields_unframed_vars(builder, &fields);
}

fn squeeze(builder: &mut R1csBuilder, transcript: &mut TranscriptGadget, label: u64, index: Option<usize>) -> KVar {
    let mut fields = Vec::with_capacity(2);
    push_const(builder, &mut fields, label);
    if let Some(index) = index {
        push_const(builder, &mut fields, index as u64);
    }
    transcript.append_fields_unframed_vars(builder, &fields);
    let limbs = transcript.challenge_fields_raw(builder, 2);
    KVar::new(limbs[0], limbs[1])
}

fn gamma_power_count(fresh_count: usize, running_count: usize, dims: Dims) -> usize {
    let total = fresh_count + running_count;
    let norm_last = fresh_count + total - 1;
    let carried_last = if running_count == 0 {
        0
    } else {
        2 * fresh_count + running_count + running_count * dims.matrix_count * D - 1
    };
    norm_last.max(carried_last) + 1
}

fn initial_claim(
    builder: &mut R1csBuilder,
    powers: &[KVar],
    fresh_count: usize,
    running: &[CeClaimWires],
    dims: Dims,
) -> KVar {
    let mut sum = KLc::zero();
    for (running_index, claim) in running.iter().enumerate() {
        for matrix in 0..dims.matrix_count {
            for coefficient in 0..D {
                let exponent = carried_exponent(
                    fresh_count,
                    running.len(),
                    dims.matrix_count,
                    running_index,
                    matrix,
                    coefficient,
                );
                let term = enforce_k_mul(
                    builder,
                    &KLc::from_var(powers[exponent]),
                    &KLc::from_var(claim.y_ring[matrix][coefficient]),
                );
                sum.c0.add_term(term.c0, F::ONE);
                sum.c1.add_term(term.c1, F::ONE);
            }
        }
    }
    alloc_klc(builder, &sum)
}

#[allow(clippy::too_many_arguments)]
fn terminal(
    builder: &mut R1csBuilder,
    cfg: &PiCcsVerifierConfig<'_>,
    powers: &[KVar],
    alpha: &[KVar],
    point: &[KVar],
    fresh_count: usize,
    running: &[CeClaimWires],
    outputs: &[CeClaimWires],
    dims: Dims,
) -> Result<KVar, Error> {
    let mut fresh_sum = KLc::zero();
    for (source, output) in outputs.iter().take(fresh_count).enumerate() {
        let values = output.ct[1..].to_vec();
        let residual = sparse_poly_eval(builder, cfg.structure.polynomial(), &values)?;
        let weighted = enforce_k_mul(builder, &KLc::from_var(powers[source]), &KLc::from_var(residual));
        fresh_sum.c0.add_term(weighted.c0, F::ONE);
        fresh_sum.c1.add_term(weighted.c1, F::ONE);
    }
    let fresh_sum = alloc_klc(builder, &fresh_sum);

    let mut norm_sum = KLc::zero();
    for (source, output) in outputs.iter().enumerate() {
        let norm = range_product(builder, output.ct[0], cfg.params.b());
        let weighted = enforce_k_mul(
            builder,
            &KLc::from_var(powers[fresh_count + source]),
            &KLc::from_var(norm),
        );
        norm_sum.c0.add_term(weighted.c0, F::ONE);
        norm_sum.c1.add_term(weighted.c1, F::ONE);
    }
    let norm_sum = alloc_klc(builder, &norm_sum);
    let residual = alloc_klc(builder, &klc_add(&KLc::from_var(fresh_sum), &KLc::from_var(norm_sum)));
    let eq_alpha = enforce_eq_k(builder, point, alpha);
    let left = enforce_k_mul(builder, &KLc::from_var(eq_alpha), &KLc::from_var(residual));

    let right = if let Some(first) = running.first() {
        let mut carried = KLc::zero();
        for (running_index, output) in outputs.iter().skip(fresh_count).enumerate() {
            for matrix in 0..dims.matrix_count {
                for coefficient in 0..D {
                    let exponent = carried_exponent(
                        fresh_count,
                        running.len(),
                        dims.matrix_count,
                        running_index,
                        matrix,
                        coefficient,
                    );
                    let weighted = enforce_k_mul(
                        builder,
                        &KLc::from_var(powers[exponent]),
                        &KLc::from_var(output.y_ring[matrix][coefficient]),
                    );
                    carried.c0.add_term(weighted.c0, F::ONE);
                    carried.c1.add_term(weighted.c1, F::ONE);
                }
            }
        }
        let carried = alloc_klc(builder, &carried);
        let eq_prior = enforce_eq_k(builder, point, &first.r);
        enforce_k_mul(builder, &KLc::from_var(eq_prior), &KLc::from_var(carried))
    } else {
        alloc_klc(builder, &KLc::zero())
    };
    Ok(alloc_klc(
        builder,
        &klc_add(&KLc::from_var(left), &KLc::from_var(right)),
    ))
}

fn sparse_poly_eval(builder: &mut R1csBuilder, polynomial: &SparsePoly<F>, values: &[KVar]) -> Result<KVar, Error> {
    if values.len() != polynomial.arity() {
        return Err(Error::Shape("terminal polynomial arity mismatch".into()));
    }
    let mut sum = KLc::zero();
    for term in polynomial.terms() {
        let mut value = alloc_klc(builder, &KLc::from_base_const(term.coeff));
        for (&base, &exponent) in values.iter().zip(&term.exps) {
            if exponent == 0 {
                continue;
            }
            let mut power = base;
            for _ in 1..exponent {
                power = enforce_k_mul(builder, &KLc::from_var(power), &KLc::from_var(base));
            }
            value = enforce_k_mul(builder, &KLc::from_var(value), &KLc::from_var(power));
        }
        sum.c0.add_term(value.c0, F::ONE);
        sum.c1.add_term(value.c1, F::ONE);
    }
    Ok(alloc_klc(builder, &sum))
}

fn range_product(builder: &mut R1csBuilder, value: KVar, base: u32) -> KVar {
    let mut product = alloc_klc(builder, &KLc::from_base_const(F::ONE));
    for integer in -((base as i64) - 1)..=((base as i64) - 1) {
        let mut factor = KLc::from_var(value);
        factor.c0.constant -= F::from_i64(integer);
        product = enforce_k_mul(builder, &KLc::from_var(product), &factor);
    }
    product
}

fn carried_exponent(
    fresh_count: usize,
    running_count: usize,
    matrix_count: usize,
    running: usize,
    matrix: usize,
    coefficient: usize,
) -> usize {
    2 * fresh_count + running_count + running + running_count * matrix + running_count * matrix_count * coefficient
}

fn bind_outputs(
    builder: &mut R1csBuilder,
    fresh: &[CcsClaimWires],
    running: &[CeClaimWires],
    outputs: &[CeClaimWires],
    point: &[KVar],
    dims: Dims,
) -> Result<(), Error> {
    for (index, output) in outputs.iter().enumerate() {
        if output.r.len() != point.len() || output.y_ring.len() != dims.matrix_count {
            return Err(Error::Shape(format!("output[{index}] has the wrong one-joint shape")));
        }
        for (&actual, &expected) in output.r.iter().zip(point) {
            enforce_kvar_eq(builder, actual, expected);
        }
        if index < fresh.len() {
            let input = &fresh[index];
            bind_metadata_and_commitment(builder, output, input.c_d_var, input.c_kappa_var, &input.c_data)?;
            if output.m_in != input.m_in
                || output.x_rows != D
                || output.x_cols != crate::paper::relations::superneo_public_x_cols(input.m_in)
            {
                return Err(Error::Shape(format!("fresh output[{index}] public shape mismatch")));
            }
            enforce_var_eq(builder, output.m_in_var, input.m_in_var);
            for coordinate in 0..input.m_in {
                let row = coordinate % D;
                let column = coordinate / D;
                enforce_var_eq(builder, output.x[row * output.x_cols + column], input.x[coordinate]);
            }
        } else {
            let input = &running[index - fresh.len()];
            bind_metadata_and_commitment(builder, output, input.c_d_var, input.c_kappa_var, &input.c_data)?;
            if output.m_in != input.m_in || output.x_rows != input.x_rows || output.x_cols != input.x_cols {
                return Err(Error::Shape(format!("running output[{index}] public shape mismatch")));
            }
            enforce_var_eq(builder, output.m_in_var, input.m_in_var);
            for (&actual, &expected) in output.x.iter().zip(&input.x) {
                enforce_var_eq(builder, actual, expected);
            }
        }
    }
    Ok(())
}

fn bind_metadata_and_commitment(
    builder: &mut R1csBuilder,
    output: &CeClaimWires,
    c_d: Var,
    c_kappa: Var,
    data: &[Var],
) -> Result<(), Error> {
    if output.c_data.len() != data.len() {
        return Err(Error::Shape("output commitment length mismatch".into()));
    }
    enforce_var_eq(builder, output.c_d_var, c_d);
    enforce_var_eq(builder, output.c_kappa_var, c_kappa);
    for (&actual, &expected) in output.c_data.iter().zip(data) {
        enforce_var_eq(builder, actual, expected);
    }
    Ok(())
}

fn public_wires(claim: CeClaimWires) -> PiCcsOutputWires {
    PiCcsOutputWires {
        c_d: claim.c_d,
        c_d_var: claim.c_d_var,
        c_kappa: claim.c_kappa,
        c_kappa_var: claim.c_kappa_var,
        c_data: claim.c_data,
        adv: claim.adv,
        x: claim.x,
        x_rows: claim.x_rows,
        x_rows_var: claim.x_rows_var,
        x_cols: claim.x_cols,
        x_cols_var: claim.x_cols_var,
        m_in: claim.m_in,
        m_in_var: claim.m_in_var,
        r: claim.r,
        y_ring: claim.y_ring,
        ct: claim.ct,
        fold_digest_fields: claim.fold_digest_fields,
    }
}

fn push_const(builder: &mut R1csBuilder, fields: &mut Vec<Var>, value: u64) {
    fields.push(alloc_constant_var(builder, F::from_u64(value)));
}

fn push_k_vars(fields: &mut Vec<Var>, values: &[KVar]) {
    for value in values {
        fields.push(value.c0);
        fields.push(value.c1);
    }
}
