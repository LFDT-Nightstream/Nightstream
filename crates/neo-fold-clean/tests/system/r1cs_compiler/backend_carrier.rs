//! CUDA-carrier integration checks for the R1CS F' compiler.

use super::*;

#[test]
fn r1cs_compiler_backend_verified_prior_fold_flag_is_consumed_once() {
    let r1cs = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00E3).expect("preprocess");

    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");
    let compiled_base = chain
        .append_assignment(assignment_one_product(3, 7))
        .expect("base append");
    let mut ctx = chain.context().clone();
    let prev_audit = chain.audit().expect("audit after base").clone();
    let placeholder = r1cs_f_prime::build_instance(&prep, &compiled_base.encoded).expect("placeholder instance");
    let derived =
        neo_fold_clean::lifecycle::extend(&prep.prep, prev_audit.clone(), vec![placeholder]).expect("prepared fold");

    let (pre_running, latest) = match &prev_audit.proof.state.proof {
        ProofState::Active { running, latest } => (
            running.materialize().expect("pre-running materialization"),
            latest.clone(),
        ),
        other => panic!("expected active pre-state, got {other:?}"),
    };
    let proof = match &derived.steps.last().expect("derived step").fold {
        FoldProof::Recursive(proof) => proof
            .materialize()
            .expect("recursive NIFS proof materialization"),
        FoldProof::NoFold => panic!("recursive extend must emit a fold proof"),
    };
    let post_running = match &derived.proof.state.proof {
        ProofState::Active { running, .. } => running.materialize().expect("post-running materialization"),
        other => panic!("expected active post-state, got {other:?}"),
    };
    ctx.fold_for_step = Some(R1csFoldForStep {
        pre_running,
        latest,
        proof,
        post_summary: None,
        post_running,
    });
    ctx.fold_for_step_needs_native_verify = false;

    compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_one_product(3, 7),
        },
    )
    .expect("backend-verified recursive compile");

    assert!(
        ctx.fold_for_step.is_none(),
        "recursive compile must consume fold_for_step"
    );
    assert!(
        ctx.fold_for_step_needs_native_verify,
        "backend-verified skip flag must reset after one compile"
    );
}

#[test]
fn r1cs_compiler_accepts_backend_post_summary_without_full_running_surface() {
    let r1cs = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x71C5_00E4).expect("preprocess");

    let mut chain = R1csChainBuilder::new(&prep).expect("start builder");
    let compiled_base = chain
        .append_assignment(assignment_one_product(3, 7))
        .expect("base append");
    let mut ctx = chain.context().clone();
    let prev_audit = chain.audit().expect("audit after base").clone();
    let placeholder = r1cs_f_prime::build_instance(&prep, &compiled_base.encoded).expect("placeholder instance");
    let derived =
        neo_fold_clean::lifecycle::extend(&prep.prep, prev_audit.clone(), vec![placeholder]).expect("prepared fold");

    let post_running = match &derived.proof.state.proof {
        ProofState::Active { running, .. } => running.materialize().expect("post-running materialization"),
        other => panic!("expected active post-state, got {other:?}"),
    };
    ctx.fold_for_step = None;
    ctx.fold_summary_for_step =
        Some(FPrimeFoldPostSummary::from_running(&post_running, ctx.public_input_len).expect("post summary"));
    ctx.fold_for_step_needs_native_verify = false;

    compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_one_product(3, 7),
        },
    )
    .expect("backend-summary recursive compile");

    assert!(
        ctx.fold_summary_for_step.is_none(),
        "recursive compile must consume fold_summary_for_step"
    );
}
