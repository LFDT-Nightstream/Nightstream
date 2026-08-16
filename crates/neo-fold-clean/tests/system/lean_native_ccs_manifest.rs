//! Fail-closed tests for the Lean native-CCS manifest boundary.
//!
//! The fixture is small. It tests the exact wire contract and selector
//! semantics. It is not a production F′ cost measurement.

use std::collections::BTreeSet;

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_fold_clean::frontends::direct_ccs::ajtai;
use neo_fold_clean::frontends::r1cs_f_prime::lean_manifest::{ColumnId, PhysicalOwner, TypedOwner, GOLDILOCKS_MODULUS};
use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::{
    audit_combined_terminal_context_guards, audit_terminal_context_guards, audit_terminal_statement_guards,
    compile_combined_terminal_r1cs, compile_combined_terminal_r1cs_statement, compile_terminal_r1cs,
    compile_terminal_r1cs_statement, compile_terminal_r1cs_statement_with_nebula_lanes,
    compile_terminal_r1cs_with_nebula_lanes, TerminalR1csInput, TerminalR1csStatement, TerminalRunningStatement,
    TerminalSpartanEngine, TerminalSpartanStatement, TERMINAL_CONTEXT_GUARD_NAMES, TERMINAL_PROOF_GUARD_NAMES,
    TERMINAL_R1CS_FAMILY_NAMES, TERMINAL_STATEMENT_GUARD_NAMES,
};
use neo_fold_clean::paper::construction2::{self, LatestInstance, ProofState, RunningInstance, State};
use neo_fold_clean::paper::digest::{
    digest_fields_as_digest32, f_prime_chunk_public_digest_for_uniform_shape, initial_boundary_digest,
    state_x_out_digest_with_mode, AccumulatorHandle, StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::r1cs::{encode_f_prime_superneo_public_input, F_PRIME_PUBLIC_INPUT_LEN};
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::relations::{
    superneo_has_canonical_x_shape, superneo_public_x_cols, CcsClaim, CcsInstance, CeClaim, LaneRanges, LaneScheme,
    WitnessMat,
};
use neo_fold_clean::{
    finish_combined_with_spartan, finish_uncompressed, finish_with_spartan, prove, verify_combined_spartan,
    verify_spartan, verify_uncompressed, LeanNativeCcsManifest, LeanNativeCcsPreprocessing, LeanNebulaCombinedManifest,
    LeanNebulaCombinedPreprocessing, TerminalR1csError, Uncompressed,
};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
use serde_json::json;
use wip_spartan::spartan::{RepeatedR1CSSNARK, R1CSSNARK};

#[path = "../support/lean_manifest_fixture.rs"]
mod lean_manifest_fixture;
use lean_manifest_fixture::{
    combined_lifecycle_manifest, combined_manifest, combined_public_suffix_manifest, extension_combined_manifest,
    instruction_owner, lifecycle_manifest, parse, parse_combined, valid_manifest,
};

const TEST_AJTAI_SEED: u64 = 0x5445_524d_494e_414c;

fn test_ajtai_seed() -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&TEST_AJTAI_SEED.to_le_bytes());
    seed
}

fn zero_superneo_public_x(m_in: usize) -> Mat<F> {
    let x = Mat::zero(D, superneo_public_x_cols(m_in), F::ZERO);
    assert!(
        superneo_has_canonical_x_shape(&x, m_in),
        "terminal fixture public input must contain complete degree-D ring elements"
    );
    x
}

fn field_value(column: &ColumnId, active: bool, valid: bool) -> F {
    match &column.owner {
        PhysicalOwner::BranchActivation { selected, .. } => {
            if *selected == !active {
                F::ONE
            } else {
                F::ZERO
            }
        }
        PhysicalOwner::Typed {
            owner: TypedOwner::Input { slot: 1 },
        } if column.coordinate_index == 0 => F::ONE,
        PhysicalOwner::Typed {
            owner: TypedOwner::Input { slot: 1 },
        } if column.coordinate_index == 1 => F::ONE,
        PhysicalOwner::Typed {
            owner: TypedOwner::Input { slot: 2 },
        } if column.coordinate_index == 0 => {
            if valid {
                F::ONE
            } else {
                F::ZERO
            }
        }
        _ => F::ONE,
    }
}

fn direct_terminal_fixture(
    manifest: &LeanNativeCcsManifest,
) -> (neo_ajtai::AjtaiSModule, Vec<CeClaim>, Vec<WitnessMat>, CcsInstance) {
    let step = manifest
        .emit_phi81_step(|column| Some(field_value(column, true, true)))
        .expect("honest Phi81 Step");
    assert!(step.is_satisfied());
    let params = Params::goldilocks_paper_b2();
    let log = ajtai::setup_seeded(&params, step.structure(), TEST_AJTAI_SEED);
    let fresh = CcsInstance::from_low_norm_assignment(
        &params,
        &log,
        step.structure(),
        step.assignment(),
        manifest.public_carrier_width(),
    )
    .expect("honest fresh instance");
    let zero_witness = Mat::zero(D, step.structure().m / D, F::ZERO);
    let joint_row_variables = step
        .structure()
        .n
        .max(step.structure().m)
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    let zero_claim = CeClaim {
        c: Commitment::zeros(D, manifest.terminal_r1cs().verifier_rows()),
        X: zero_superneo_public_x(manifest.public_carrier_width()),
        r: vec![K::ZERO; joint_row_variables],
        y_ring: vec![vec![K::ZERO; D.next_power_of_two()]; step.structure().t() + 1],
        ct: vec![K::ZERO; step.structure().t() + 1],
        m_in: manifest.public_carrier_width(),
        fold_digest: [0; 32],
        adv: None,
    };
    (
        log,
        vec![zero_claim; manifest.running_claim_count()],
        vec![zero_witness; manifest.running_claim_count()],
        fresh,
    )
}

fn direct_terminal_lane_scheme(manifest: &LeanNativeCcsManifest) -> LaneScheme {
    LaneScheme::from_seeds(
        manifest.terminal_r1cs().verifier_rows(),
        LaneRanges {
            ops: 0..1,
            is: 1..2,
            fs: 2..3,
        },
        [0xA1; 32],
        [0xB2; 32],
    )
    .expect("terminal lane scheme")
}

fn direct_combined_terminal_fixture(
    manifest: &LeanNebulaCombinedManifest,
    nebula_private: &[F],
) -> (neo_ajtai::AjtaiSModule, Vec<CeClaim>, Vec<WitnessMat>, CcsInstance) {
    let mut public = vec![F::ZERO; manifest.public_carrier_width()];
    public[0] = F::ONE;
    let emission = manifest
        .emit(&public, |_| Some(F::ZERO), nebula_private)
        .expect("honest combined emission");
    assert!(emission.is_satisfied());
    let params = Params::goldilocks_paper_b2();
    let log = ajtai::setup_seeded(&params, emission.structure(), TEST_AJTAI_SEED);
    let fresh = CcsInstance::from_low_norm_assignment(
        &params,
        &log,
        emission.structure(),
        emission.assignment(),
        manifest.public_carrier_width(),
    )
    .expect("honest combined fresh instance");
    let zero_witness = Mat::zero(D, emission.structure().m / D, F::ZERO);
    let joint_row_variables = emission
        .structure()
        .n
        .max(emission.structure().m)
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    let zero_claim = CeClaim {
        c: Commitment::zeros(D, manifest.terminal_r1cs().verifier_rows()),
        X: zero_superneo_public_x(manifest.public_carrier_width()),
        r: vec![K::ZERO; joint_row_variables],
        y_ring: vec![vec![K::ZERO; D.next_power_of_two()]; emission.structure().t() + 1],
        ct: vec![K::ZERO; emission.structure().t() + 1],
        m_in: manifest.public_carrier_width(),
        fold_digest: [0; 32],
        adv: None,
    };
    (log, vec![zero_claim; 14], vec![zero_witness; 14], fresh)
}

fn terminal_lifecycle_fixture(manifest: &LeanNativeCcsManifest) -> (neo_fold_clean::Preprocessing, Uncompressed) {
    let probe = manifest
        .emit_phi81_step(|_| Some(F::ZERO))
        .expect("terminal relation probe");
    let params = neo_fold_clean::config::ccs_params(
        probe.structure().n,
        probe.structure().m,
        probe.structure().t(),
        probe.structure().max_degree(),
    )
    .expect("shape-derived native CCS parameters");
    let log = ajtai::setup_seeded(&params, probe.structure(), TEST_AJTAI_SEED);
    let prep = neo_fold_clean::lifecycle::preprocess_with_test_log(
        params.clone(),
        probe.structure().clone(),
        log.clone(),
        Some(manifest.public_carrier_width()),
    )
    .expect("terminal preprocessing");

    let zero_witness = Mat::zero(D, probe.structure().m / D, F::ZERO);
    let joint_row_variables = probe
        .structure()
        .n
        .max(probe.structure().m)
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    let zero_claim = CeClaim {
        c: Commitment::zeros(D, manifest.terminal_r1cs().verifier_rows()),
        X: zero_superneo_public_x(manifest.public_carrier_width()),
        r: vec![K::ZERO; joint_row_variables],
        y_ring: vec![vec![K::ZERO; D.next_power_of_two()]; probe.structure().t() + 1],
        ct: vec![K::ZERO; probe.structure().t() + 1],
        m_in: manifest.public_carrier_width(),
        fold_digest: [0; 32],
        adv: None,
    };
    let running_claims = vec![zero_claim.clone(); manifest.running_claim_count()];
    let running_witnesses = vec![zero_witness; manifest.running_claim_count()];
    let acc_digest = AccumulatorHandle::from_running_parts(2, &running_claims, Some(&zero_claim)).digest();
    let z_0 = initial_boundary_digest(prep.structure_digest(), prep.public_input_len);

    let placeholder = CcsClaim {
        c: Commitment::zeros(D, manifest.terminal_r1cs().verifier_rows()),
        x: vec![F::ZERO; manifest.public_carrier_width()],
        m_in: manifest.public_carrier_width(),
        adv: None,
    };
    let false_activation = manifest
        .step_program()
        .receipts
        .iter()
        .flat_map(|receipt| &receipt.allocations)
        .find(|allocation| {
            matches!(
                allocation.id.owner,
                PhysicalOwner::BranchActivation { selected: false, .. }
            )
        })
        .expect("false activation")
        .id
        .clone();
    let row = manifest
        .step_program()
        .receipts
        .iter()
        .flat_map(|receipt| &receipt.rows)
        .next()
        .expect("one selected row");
    let row_indices = [
        probe.column_index(&row.a[0].column).expect("A column"),
        probe.column_index(&row.b[0].column).expect("B column"),
        probe.column_index(&row.c[0].column).expect("C column"),
    ];
    let activation_index = probe
        .column_index(&false_activation)
        .expect("activation column");

    let mut selected = None;
    for step_count in 1..=256u64 {
        let boundary = digest_fields_as_digest32(f_prime_chunk_public_digest_for_uniform_shape(
            step_count - 1,
            1,
            placeholder.c.d,
            placeholder.c.kappa,
            placeholder.m_in,
        ));
        let digest = state_x_out_digest_with_mode(
            StateXOutDigestMode::Stateless,
            prep.vk.digest(),
            prep.pi_ccs_header_bundle(),
            prep.structure_digest(),
            step_count,
            step_count,
            z_0,
            boundary,
            neo_fold_clean::paper::construction2::TRIVIAL_PC,
            acc_digest,
            acc_digest,
            boundary,
            None,
        );
        let carrier = encode_f_prime_superneo_public_input(neo_fold_clean::paper::digest::digest32_as_fields(digest));
        let row_ok = carrier[activation_index] == F::ZERO
            || carrier[row_indices[0]] * carrier[row_indices[1]] == carrier[row_indices[2]];
        if row_ok {
            selected = Some((step_count, boundary, carrier));
            break;
        }
    }
    let (step_count, boundary, assignment) = selected.expect("bounded terminal-link fixture");
    let emission = manifest
        .emit_phi81_step(|column| probe.column_index(column).map(|index| assignment[index]))
        .expect("linked terminal relation");
    assert!(emission.is_satisfied());
    let fresh = CcsInstance::from_low_norm_assignment(
        &params,
        &log,
        emission.structure(),
        emission.assignment(),
        manifest.public_carrier_width(),
    )
    .expect("linked fresh instance");
    assert_eq!(
        boundary,
        digest_fields_as_digest32(f_prime_chunk_public_digest_for_uniform_shape(
            step_count - 1,
            1,
            fresh.claim.c.d,
            fresh.claim.c.kappa,
            fresh.claim.m_in,
        ))
    );

    let running = RunningInstance::new(running_claims, running_witnesses, Some(zero_claim));
    let state = State {
        chunk_count: step_count,
        step_count,
        z_0,
        z_i: boundary,
        pc: neo_fold_clean::paper::construction2::TRIVIAL_PC,
        initial_semantic_state_digest: prep.initial_semantic_state_digest(),
        semantic_state_digest: acc_digest,
        acc_digest,
        public_trace: boundary,
        proof: ProofState::active(running, LatestInstance::from_instances(vec![fresh])),
        nebula: None,
    };
    (
        prep,
        Uncompressed {
            state,
            final_fold: None,
        },
    )
}

#[test]
fn emits_the_exact_four_matrix_native_selector() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let emission = manifest
        .emit_step(|column| Some(field_value(column, true, true)))
        .expect("emit native Step");

    assert_eq!(manifest.matrix_count(), 4);
    assert_eq!(manifest.polynomial_degree(), 3);
    assert_eq!(emission.structure().t(), 4);
    assert_eq!(emission.structure().max_degree(), 3);
    assert_eq!(emission.structure().m, 270);
    assert_eq!(emission.structure().n, 1);
    assert!(emission.is_satisfied());
}

#[test]
fn preprocessing_and_instance_are_derived_from_the_manifest() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let setup = LeanNativeCcsPreprocessing::new(manifest).expect("manifest-owned native preprocessing");

    assert!(setup.preprocessing().enforces_terminal_induction());
    assert_eq!(
        setup.preprocessing().public_input_len,
        Some(setup.manifest().public_carrier_width())
    );
    assert_eq!(
        setup.preprocessing().log.seeded_params(),
        Some((
            setup.manifest().terminal_r1cs().verifier_rows(),
            setup.manifest().ajtai_setup_seed(),
        ))
    );

    let instance = setup
        .build_instance(|column| Some(field_value(column, true, true)))
        .expect("satisfying manifest assignment");
    assert_eq!(instance.claim.m_in, setup.manifest().public_carrier_width());
}

#[test]
#[ignore = "runs one complete native F-prime fold and the 34,294-row Spartan/WHIR terminal proof"]
fn manifest_owned_lifecycle_proves_and_verifies() {
    let manifest = parse(&lifecycle_manifest()).expect("valid native lifecycle manifest");
    let setup = LeanNativeCcsPreprocessing::new(manifest).expect("manifest-owned native preprocessing");
    let prep = setup.preprocessing();

    let base = prove(prep, Vec::<Vec<CcsInstance>>::new()).expect("base lifecycle state");
    let mut post_state = base.proof.state.clone();
    let boundary = digest_fields_as_digest32(f_prime_chunk_public_digest_for_uniform_shape(
        0,
        1,
        D,
        prep.params.kappa() as usize,
        setup.manifest().public_carrier_width(),
    ));
    post_state.chunk_count = 1;
    post_state.step_count = 1;
    post_state.z_i = boundary;
    post_state.public_trace = boundary;
    let default_running = RunningInstance::canonical_zero(
        &prep.params,
        prep.structure(),
        setup.manifest().public_carrier_width(),
        neo_fold_clean::paper::construction2::LaneCommitmentMode::Plain,
    )
    .expect("paper default running accumulator");
    post_state.acc_digest =
        AccumulatorHandle::from_running_parts(2, &default_running.claims, default_running.parent_authority.as_ref())
            .digest();
    post_state.semantic_state_digest = post_state.acc_digest;
    let state = &post_state;
    let mode = match prep.semantic_state_mode() {
        construction2::SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
        construction2::SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
    };
    let expected = state_x_out_digest_with_mode(
        mode,
        prep.vk.digest(),
        prep.pi_ccs_header_bundle(),
        prep.structure_digest(),
        state.chunk_count,
        state.step_count,
        state.z_0,
        state.z_i,
        state.pc,
        state.semantic_state_digest,
        state.acc_digest,
        state.public_trace,
        None,
    );
    let carrier = encode_f_prime_superneo_public_input(neo_fold_clean::paper::digest::digest32_as_fields(expected));

    let baseline = setup
        .manifest()
        .emit_phi81_step(|column| Some(field_value(column, true, true)))
        .expect("baseline manifest assignment");
    let mut assignment = baseline.assignment().to_vec();
    assignment[..carrier.len()].copy_from_slice(&carrier);
    let instance = setup
        .build_instance(|column| baseline.column_index(column).map(|index| assignment[index]))
        .expect("base-linked manifest instance");
    assert_eq!(instance.claim.x, carrier);
    let audit = prove(prep, [vec![instance]]).expect("one native F-prime fold");
    assert_eq!(audit.proof.state.acc_digest, post_state.acc_digest);
    assert_eq!(
        audit.proof.state.semantic_state_digest,
        post_state.semantic_state_digest
    );
    let terminal = finish_uncompressed(prep, audit).expect("plain HyperNova running/latest terminal state");
    verify_uncompressed(prep, &terminal).expect("uncompressed terminal verification");

    let (statement, proof) =
        finish_with_spartan(prep, setup.manifest(), terminal).expect("terminal Spartan/WHIR proof");
    let public_image = statement.public_image().clone();
    verify_spartan(prep, setup.manifest(), &public_image, &statement, &proof)
        .expect("terminal Spartan/WHIR verification");
}

#[test]
#[ignore = "runs one complete combined F-prime fold and the 58,595-row Spartan/WHIR terminal proof"]
fn combined_manifest_owned_lifecycle_proves_and_verifies() {
    let manifest = parse_combined(&combined_lifecycle_manifest()).expect("valid combined lifecycle manifest");
    let setup = LeanNebulaCombinedPreprocessing::new(manifest).expect("manifest-owned combined preprocessing");
    let prep = setup.preprocessing();

    let base = prove(prep, Vec::<Vec<CcsInstance>>::new()).expect("base lifecycle state");
    let mut post_state = base.proof.state.clone();
    let boundary = digest_fields_as_digest32(f_prime_chunk_public_digest_for_uniform_shape(
        0,
        1,
        D,
        prep.params.kappa() as usize,
        setup.manifest().public_carrier_width(),
    ));
    post_state.chunk_count = 1;
    post_state.step_count = 1;
    post_state.z_i = boundary;
    post_state.public_trace = boundary;
    let default_running = RunningInstance::canonical_zero(
        &prep.params,
        prep.structure(),
        setup.manifest().public_carrier_width(),
        neo_fold_clean::paper::construction2::LaneCommitmentMode::Plain,
    )
    .expect("paper default running accumulator");
    post_state.acc_digest =
        AccumulatorHandle::from_running_parts(2, &default_running.claims, default_running.parent_authority.as_ref())
            .digest();
    post_state.semantic_state_digest = post_state.acc_digest;
    let mode = match prep.semantic_state_mode() {
        construction2::SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
        construction2::SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
    };
    let expected = state_x_out_digest_with_mode(
        mode,
        prep.vk.digest(),
        prep.pi_ccs_header_bundle(),
        prep.structure_digest(),
        post_state.chunk_count,
        post_state.step_count,
        post_state.z_0,
        post_state.z_i,
        post_state.pc,
        post_state.semantic_state_digest,
        post_state.acc_digest,
        post_state.public_trace,
        None,
    );
    let public = encode_f_prime_superneo_public_input(neo_fold_clean::paper::digest::digest32_as_fields(expected));
    assert_eq!(public.len(), setup.manifest().public_carrier_width());

    let baseline = setup
        .manifest()
        .core()
        .emit_phi81_step(|column| Some(field_value(column, true, true)))
        .expect("baseline native assignment");
    let instance = setup
        .build_instance(
            &public,
            |column| {
                baseline.column_index(column).map(|index| {
                    if index < F_PRIME_PUBLIC_INPUT_LEN {
                        public[index]
                    } else {
                        baseline.assignment()[index]
                    }
                })
            },
            &[F::ZERO],
        )
        .expect("combined linked instance");
    assert_eq!(instance.claim.x, public);

    let audit = prove(prep, [vec![instance]]).expect("one combined F-prime fold");
    assert_eq!(audit.proof.state.acc_digest, post_state.acc_digest);
    assert_eq!(
        audit.proof.state.semantic_state_digest,
        post_state.semantic_state_digest
    );
    let terminal = finish_uncompressed(prep, audit).expect("plain HyperNova running/latest terminal state");
    verify_uncompressed(prep, &terminal).expect("uncompressed combined terminal verification");

    let (statement, proof) =
        finish_combined_with_spartan(&setup, terminal).expect("combined terminal Spartan/WHIR proof");
    let public_image = statement.public_image().clone();
    verify_combined_spartan(&setup, &public_image, &statement, &proof)
        .expect("combined terminal Spartan/WHIR verification");
}

#[test]
fn active_selector_enforces_and_inactive_selector_disables() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let invalid_active = manifest
        .emit_step(|column| Some(field_value(column, true, false)))
        .expect("emit active native Step");
    assert!(!invalid_active.is_satisfied());

    let invalid_inactive = manifest
        .emit_step(|column| Some(field_value(column, false, false)))
        .expect("emit inactive native Step");
    assert!(invalid_inactive.is_satisfied());
}

#[test]
fn native_step_has_no_residual_row_or_column() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let emission = manifest
        .emit_step(|column| Some(field_value(column, true, true)))
        .expect("emit native Step");

    assert_eq!(manifest.step_cost().recurring_rows(), 1);
    assert_eq!(manifest.step_cost().auxiliary_columns(), 242);
    assert_eq!(emission.auxiliary_columns().len(), 242);
    assert_eq!(emission.structure().n, 1);
    assert_eq!(manifest.terminal_r1cs().logical_width(), 270);
    assert_eq!(manifest.terminal_r1cs().recursive_rows(), 1);
    assert_eq!(manifest.terminal_r1cs().cost().recurring_rows(), 34_292);
}

#[test]
fn rejects_polynomial_or_matrix_shape_drift() {
    let mut matrix_count = valid_manifest();
    matrix_count["step_program"]["matrix_count"] = json!(5);
    assert!(parse(&matrix_count).unwrap_err().contains("matrix_count"));

    let mut degree = valid_manifest();
    degree["step_program"]["polynomial_degree"] = json!(4);
    assert!(parse(&degree).unwrap_err().contains("polynomial_degree"));

    let mut exponent = valid_manifest();
    exponent["step_program"]["polynomial"][0]["exponents"][3] = json!(0);
    assert!(parse(&exponent).unwrap_err().contains("polynomial"));
}

#[test]
fn rejects_selector_substitution_and_wrong_selected_owner() {
    let mut selector = valid_manifest();
    let one = selector["step_program"]["one"].clone();
    let receipts = selector["step_program"]["receipts"].as_array_mut().unwrap();
    let target = receipts.len() - 1;
    receipts[target]["selector"] = one;
    assert!(parse(&selector).unwrap_err().contains("selected receipts"));

    let mut owner = valid_manifest();
    let receipts = owner["step_program"]["receipts"].as_array_mut().unwrap();
    let target = receipts.len() - 1;
    let wrong_owner = instruction_owner(&["rest"]);
    receipts[target]["owner"] = wrong_owner.clone();
    receipts[target]["rows"][0]["id"]["owner"] = wrong_owner.clone();
    for allocation in receipts[target]["allocations"].as_array_mut().unwrap() {
        allocation["id"]["owner"] = wrong_owner.clone();
    }
    assert!(parse(&owner)
        .unwrap_err()
        .contains("only native-selected receipt"));
}

#[test]
fn rejects_cost_drift_unknown_fields_and_noncanonical_coefficients() {
    let mut cost = valid_manifest();
    cost["step_cost"]["recurring_rows"] = json!(2);
    assert!(parse(&cost).unwrap_err().contains("step_cost"));

    let mut unknown = valid_manifest();
    unknown["rust_authority"] = json!(true);
    assert!(parse(&unknown).unwrap_err().contains("unknown field"));

    let mut coefficient = valid_manifest();
    let receipts = coefficient["step_program"]["receipts"]
        .as_array_mut()
        .unwrap();
    let target = receipts.len() - 1;
    receipts[target]["rows"][0]["a"][0]["coefficient"] = json!(GOLDILOCKS_MODULUS);
    assert!(parse(&coefficient)
        .unwrap_err()
        .contains("canonical Goldilocks residue"));
}

#[test]
fn rejects_invalid_or_mismatched_ajtai_setup() {
    let mut algorithm = valid_manifest();
    algorithm["ajtai_setup"]["algorithm"] = json!("different_sampler");
    assert!(parse(&algorithm).is_err());

    let mut short_seed = valid_manifest();
    short_seed["ajtai_setup"]["seed"] = json!(vec![0u8; 31]);
    assert!(parse(&short_seed).is_err());

    let mut no_fuel = valid_manifest();
    no_fuel["ajtai_setup"]["rejection_fuel"] = json!(0);
    assert!(parse(&no_fuel).unwrap_err().contains("rejection_fuel"));

    let mut different_seed = valid_manifest();
    different_seed["ajtai_setup"]["seed"][0] = json!(test_ajtai_seed()[0] ^ 1);
    let manifest = parse(&different_seed).expect("structurally valid alternate setup");
    let (log, running_claims, running_witnesses, fresh) = direct_terminal_fixture(&manifest);
    assert!(matches!(
        compile_terminal_r1cs(
            &manifest,
            &log,
            TerminalR1csInput {
                running_claims: &running_claims,
                running_witnesses: &running_witnesses,
                fresh: &fresh,
            },
        ),
        Err(TerminalR1csError::SetupMismatch)
    ));
}

#[test]
fn rejects_terminal_r1cs_shape_or_cost_drift() {
    let mut width = valid_manifest();
    width["terminal_r1cs"]["logical_width"] = json!(269);
    assert!(parse(&width).unwrap_err().contains("logical_width"));

    let mut rows = valid_manifest();
    rows["terminal_r1cs"]["recursive_rows"] = json!(2);
    assert!(parse(&rows).unwrap_err().contains("recursive_rows"));

    let mut domain = valid_manifest();
    domain["terminal_r1cs"]["row_variables"] = json!(1);
    assert!(parse(&domain).unwrap_err().contains("least power-of-two"));

    let mut matrix = valid_manifest();
    matrix["terminal_r1cs"]["matrix_count"] = json!(3);
    assert!(parse(&matrix).unwrap_err().contains("matrix_count"));

    let mut public = valid_manifest();
    public["terminal_r1cs"]["public_ring_columns"] = json!(4);
    assert!(parse(&public).unwrap_err().contains("public_ring_columns"));

    let mut verifier = valid_manifest();
    verifier["terminal_r1cs"]["verifier_rows"] = json!(17);
    assert!(parse(&verifier).unwrap_err().contains("verifier_rows"));

    let mut cost = valid_manifest();
    cost["terminal_r1cs"]["cost"]["auxiliary_columns"] = json!(4_052);
    assert!(parse(&cost).unwrap_err().contains("terminal_r1cs.cost"));
}

#[test]
#[ignore = "materializes the complete 14-running terminal reference relation"]
fn terminal_r1cs_compiles_with_the_exact_lean_cost() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let (log, running_claims, running_witnesses, fresh) = direct_terminal_fixture(&manifest);
    let relation = compile_terminal_r1cs(
        &manifest,
        &log,
        TerminalR1csInput {
            running_claims: &running_claims,
            running_witnesses: &running_witnesses,
            fresh: &fresh,
        },
    )
    .expect("honest terminal R1CS");

    assert_eq!(relation.shape().num_constraints_unpadded(), 34_292);
    assert_eq!(relation.shape().num_rest_unpadded(), 8_101);
    assert_eq!(relation.shape().num_public(), 26_190);
    assert_eq!(relation.lean_public_columns(), 26_191);
}

#[test]
#[ignore = "proves the complete bounded terminal reference relation with WHIR"]
fn terminal_r1cs_proves_and_verifies_with_spartan_and_whir() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let (log, running_claims, running_witnesses, fresh) = direct_terminal_fixture(&manifest);
    let relation = compile_terminal_r1cs(
        &manifest,
        &log,
        TerminalR1csInput {
            running_claims: &running_claims,
            running_witnesses: &running_witnesses,
            fresh: &fresh,
        },
    )
    .expect("honest terminal R1CS");
    let (_, witness, public) = relation.into_parts();
    let statement = compile_terminal_r1cs_statement(
        &manifest,
        &log,
        TerminalR1csStatement {
            running_claims: &running_claims,
            fresh_claim: &fresh.claim,
        },
    )
    .expect("verifier terminal R1CS");
    assert_eq!(statement.public_values(), public);
    let (shape, verifier_public) = statement.into_parts();
    let (prover_key, verifier_key) =
        R1CSSNARK::<TerminalSpartanEngine>::setup_direct(shape).expect("direct Spartan setup");
    let proof = RepeatedR1CSSNARK::<TerminalSpartanEngine>::prove_direct(&prover_key, &witness, &public, true)
        .expect("direct Spartan proof");

    assert_eq!(proof.verify(&verifier_key).expect("WHIR verification"), verifier_public);
}

#[test]
fn terminal_nebula_lane_openings_bind_the_same_witness() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let (log, mut running_claims, running_witnesses, mut fresh) = direct_terminal_fixture(&manifest);
    let lanes = direct_terminal_lane_scheme(&manifest);
    let zero_adv = lanes
        .commit(&running_witnesses[0])
        .expect("zero running lane commitments");
    for claim in &mut running_claims {
        claim.adv = Some(zero_adv.clone());
    }
    fresh.claim.adv = Some(
        lanes
            .commit(&fresh.witness.Z)
            .expect("fresh lane commitments"),
    );

    let relation = compile_terminal_r1cs_with_nebula_lanes(
        &manifest,
        &log,
        &lanes,
        TerminalR1csInput {
            running_claims: &running_claims,
            running_witnesses: &running_witnesses,
            fresh: &fresh,
        },
    )
    .expect("terminal relation with lane openings");
    let statement = compile_terminal_r1cs_statement_with_nebula_lanes(
        &manifest,
        &log,
        &lanes,
        TerminalR1csStatement {
            running_claims: &running_claims,
            fresh_claim: &fresh.claim,
        },
    )
    .expect("terminal lane-opening statement");

    let base = manifest.terminal_r1cs().cost();
    let claim_count = running_claims.len() + 1;
    let added = claim_count * 3 * D * manifest.terminal_r1cs().verifier_rows();
    assert_eq!(
        relation.shape().num_constraints_unpadded(),
        base.recurring_rows() + added
    );
    assert_eq!(relation.lean_public_columns(), base.public_columns() + added);
    assert_eq!(statement.shape(), relation.shape());
    assert_eq!(statement.public_values(), relation.public_values());
    assert!(relation
        .constraint_audit()
        .source()
        .is_satisfied(relation.constraint_audit().source().witness()));
    assert_eq!(
        relation
            .constraint_audit()
            .row_families()
            .iter()
            .map(|range| range.name)
            .collect::<BTreeSet<_>>(),
        TERMINAL_R1CS_FAMILY_NAMES.into_iter().collect()
    );
    let family_rows = |name: &str| {
        relation
            .constraint_audit()
            .row_families()
            .iter()
            .filter(|range| range.name == name)
            .map(|range| range.row_end - range.row_start)
            .sum::<usize>()
    };
    let commitment_rows_per_claim = 4 * D * manifest.terminal_r1cs().verifier_rows();
    assert_eq!(
        family_rows("terminal.running.commitment"),
        running_claims.len() * commitment_rows_per_claim
    );
    assert_eq!(family_rows("terminal.fresh.commitment"), commitment_rows_per_claim);

    fresh.claim.adv.as_mut().expect("fresh sidecar").ops.data[0] += F::ONE;
    assert!(matches!(
        compile_terminal_r1cs_with_nebula_lanes(
            &manifest,
            &log,
            &lanes,
            TerminalR1csInput {
                running_claims: &running_claims,
                running_witnesses: &running_witnesses,
                fresh: &fresh,
            },
        ),
        Err(TerminalR1csError::Unsatisfied(_))
    ));

    fresh.claim.adv = None;
    assert!(matches!(
        compile_terminal_r1cs_with_nebula_lanes(
            &manifest,
            &log,
            &lanes,
            TerminalR1csInput {
                running_claims: &running_claims,
                running_witnesses: &running_witnesses,
                fresh: &fresh,
            },
        ),
        Err(TerminalR1csError::Unsupported(_))
    ));
}

#[test]
fn terminal_lifecycle_rejects_preprocessing_without_recursive_induction() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let (prep, uncompressed) = terminal_lifecycle_fixture(&manifest);
    assert!(matches!(
        finish_with_spartan(&prep, &manifest, uncompressed),
        Err(TerminalR1csError::UncertifiedInduction)
    ));
}

#[test]
fn terminal_statement_guard_audit_covers_every_native_statement_check() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let (prep, uncompressed) = terminal_lifecycle_fixture(&manifest);
    let state = &uncompressed.state;
    let ProofState::Active { running, latest } = &state.proof else {
        panic!("terminal fixture must carry one running/latest pair");
    };
    let mode = match prep.semantic_state_mode() {
        construction2::SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
        construction2::SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
    };
    let x_out = state_x_out_digest_with_mode(
        mode,
        prep.vk.digest(),
        prep.pi_ccs_header_bundle(),
        prep.structure_digest(),
        state.chunk_count,
        state.step_count,
        state.z_0,
        state.z_i,
        state.pc,
        state.semantic_state_digest,
        state.acc_digest,
        state.public_trace,
        None,
    );
    let image = neo_fold_clean::PublicImage {
        vk_fs_digest: prep.vk.digest(),
        chunk_count: state.chunk_count,
        step_count: state.step_count,
        z_0: state.z_0,
        z_i: state.z_i,
        pc: state.pc,
        initial_semantic_state_digest: state.initial_semantic_state_digest,
        semantic_state_digest: state.semantic_state_digest,
        acc_digest: state.acc_digest,
        public_trace: state.public_trace,
        x_out: construction2::EncInst::from_digest(x_out),
    };
    let statement = TerminalSpartanStatement::new(
        image,
        TerminalRunningStatement::from_running(running),
        latest.instances[0].claim.clone(),
    );
    let audit =
        audit_terminal_statement_guards(&prep, &manifest, &statement).expect("valid terminal statement guard ledger");

    assert_eq!(audit.guard_names(), TERMINAL_STATEMENT_GUARD_NAMES);
}

#[test]
fn terminal_context_guard_audit_covers_every_native_context_check() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let setup = LeanNativeCcsPreprocessing::new(manifest).expect("manifest-owned native preprocessing");
    let audit = audit_terminal_context_guards(setup.preprocessing(), setup.manifest())
        .expect("valid terminal context guard ledger");

    assert_eq!(audit.guard_names(), TERMINAL_CONTEXT_GUARD_NAMES);
    let native_guards = TERMINAL_CONTEXT_GUARD_NAMES
        .into_iter()
        .chain(TERMINAL_STATEMENT_GUARD_NAMES)
        .chain(TERMINAL_PROOF_GUARD_NAMES)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        native_guards.len(),
        TERMINAL_CONTEXT_GUARD_NAMES.len() + TERMINAL_STATEMENT_GUARD_NAMES.len() + TERMINAL_PROOF_GUARD_NAMES.len()
    );
    assert!(TERMINAL_R1CS_FAMILY_NAMES
        .into_iter()
        .all(|family| !native_guards.contains(family)));
}

#[test]
fn combined_terminal_context_guard_audit_binds_manifest_owned_preprocessing() {
    let manifest = parse_combined(&combined_manifest()).expect("valid combined manifest");
    let setup = LeanNebulaCombinedPreprocessing::new(manifest).expect("manifest-owned combined preprocessing");
    let audit = audit_combined_terminal_context_guards(&setup).expect("valid combined context guard ledger");

    assert_eq!(audit.guard_names(), TERMINAL_CONTEXT_GUARD_NAMES);
    assert!(matches!(
        audit_terminal_context_guards(setup.preprocessing(), setup.manifest().core()),
        Err(TerminalR1csError::RelationMismatch)
    ));
}

#[test]
fn combined_manifest_emits_the_exact_nineteen_matrix_relation() {
    let manifest = parse_combined(&combined_manifest()).expect("valid combined manifest");
    let mut public = vec![F::ZERO; manifest.public_carrier_width()];
    public[0] = F::ONE;
    let private = vec![F::ZERO; manifest.nebula_private_width()];
    let emission = manifest
        .emit(&public, |_| Some(F::ZERO), &private)
        .expect("exact combined emission");

    assert_eq!(manifest.core().matrix_count(), 4);
    assert_eq!(manifest.matrix_count(), 19);
    assert_eq!(manifest.strict_degree_bound(), 5);
    assert_eq!(emission.structure().t(), 19);
    assert_eq!(emission.structure().max_degree(), 4);
    assert_eq!(emission.structure().n, 2);
    assert_eq!(emission.structure().m, 324);
    assert_eq!(emission.logical_width(), 284);
    assert_eq!(emission.public_width(), 270);
    assert!(emission.is_satisfied());
}

#[test]
fn combined_manifest_derives_the_application_public_suffix_layout() {
    let manifest = parse_combined(&combined_public_suffix_manifest()).expect("valid combined public-suffix manifest");

    assert_eq!(manifest.public_input_layout().suffix_len(), 1);
    assert_eq!(manifest.public_input_layout().total_len(), 270);
}

#[test]
fn combined_preprocessing_and_instance_are_manifest_owned() {
    let manifest = parse_combined(&combined_manifest()).expect("valid combined manifest");
    let setup = LeanNebulaCombinedPreprocessing::new(manifest).expect("combined preprocessing");
    let mut public = vec![F::ZERO; setup.manifest().public_carrier_width()];
    public[0] = F::ONE;
    let private = vec![F::ZERO; setup.manifest().nebula_private_width()];
    let instance = setup
        .build_instance(&public, |_| Some(F::ZERO), &private)
        .expect("combined instance");

    assert!(setup.preprocessing().enforces_terminal_induction());
    assert_eq!(setup.preprocessing().structure().t(), 19);
    assert_eq!(instance.claim.m_in, 270);
    assert_eq!(instance.claim.x, public);
}

#[test]
fn combined_terminal_r1cs_compiles_the_exact_lean_bit_lowering() {
    let manifest = parse_combined(&combined_manifest()).expect("valid combined manifest");
    let (log, running_claims, running_witnesses, fresh) = direct_combined_terminal_fixture(&manifest, &[F::ZERO]);
    let relation = compile_combined_terminal_r1cs(
        &manifest,
        &log,
        TerminalR1csInput {
            running_claims: &running_claims,
            running_witnesses: &running_witnesses,
            fresh: &fresh,
        },
    )
    .expect("honest combined terminal R1CS");
    let statement = compile_combined_terminal_r1cs_statement(
        &manifest,
        &log,
        TerminalR1csStatement {
            running_claims: &running_claims,
            fresh_claim: &fresh.claim,
        },
    )
    .expect("combined terminal statement");

    assert_eq!(relation.shape().num_constraints_unpadded(), 58_593);
    assert_eq!(relation.shape().num_rest_unpadded(), 9_721);
    assert_eq!(relation.shape().num_public(), 48_870);
    assert_eq!(relation.lean_public_columns(), 48_871);
    assert_eq!(statement.shape(), relation.shape());
    assert_eq!(statement.public_values(), relation.public_values());

    let audit = relation.constraint_audit();
    assert_eq!(audit.source().rows(), relation.shape().num_constraints_unpadded());
    assert_eq!(audit.source().cols(), 1 + 48_870 + 9_721);
    assert!(audit.source().is_satisfied(audit.source().witness()));
    assert_eq!(audit.source_public_columns(), 48_871);
    assert_eq!(audit.source_private_columns(), 9_721);
    assert_eq!(audit.spartan_rows(), relation.shape().num_constraints());
    assert_eq!(
        audit.spartan_columns(),
        relation.shape().num_variables() + 1 + relation.shape().num_public()
    );
    assert_eq!(audit.spartan_private_columns(), relation.shape().num_variables());
    assert_eq!(audit.source_to_spartan_column(0), Some(audit.spartan_private_columns()));
    assert_eq!(
        audit.source_to_spartan_column(1),
        Some(audit.spartan_private_columns() + 1)
    );
    assert_eq!(audit.source_to_spartan_column(audit.source_public_columns()), Some(0));
    assert_eq!(audit.source_to_spartan_column(audit.source().cols()), None);
    assert_eq!(
        audit
            .row_families()
            .iter()
            .map(|range| range.name)
            .collect::<BTreeSet<_>>(),
        TERMINAL_R1CS_FAMILY_NAMES.into_iter().collect()
    );
}

#[test]
fn combined_terminal_r1cs_compiles_the_exact_lean_extension_lowering() {
    let manifest = parse_combined(&extension_combined_manifest()).expect("valid extension manifest");
    let (log, running_claims, running_witnesses, fresh) = direct_combined_terminal_fixture(&manifest, &[F::ONE]);
    let relation = compile_combined_terminal_r1cs(
        &manifest,
        &log,
        TerminalR1csInput {
            running_claims: &running_claims,
            running_witnesses: &running_witnesses,
            fresh: &fresh,
        },
    )
    .expect("honest extension terminal R1CS");

    assert_eq!(relation.shape().num_constraints_unpadded(), 58_598);
    assert_eq!(relation.shape().num_rest_unpadded(), 9_726);
    assert_eq!(relation.shape().num_public(), 48_870);
    assert_eq!(relation.lean_public_columns(), 48_871);
}

#[test]
fn combined_manifest_rejects_relation_and_layout_drift() {
    let mut arity = combined_manifest();
    arity["relation"]["matrix_count"] = json!(4);
    assert!(parse_combined(&arity).unwrap_err().contains("matrix_count"));

    let mut polynomial = combined_manifest();
    polynomial["relation"]["polynomial"][0]["coefficient"] = json!(2);
    assert!(parse_combined(&polynomial)
        .unwrap_err()
        .contains("polynomial"));

    let mut position = combined_manifest();
    position["relation"]["application"]["rows"][0]["id"]["position"] = json!(1);
    assert!(parse_combined(&position).unwrap_err().contains("position"));

    let mut coefficient = combined_manifest();
    coefficient["relation"]["application"]["rows"][0]["images"]["bit"][0]["coefficient"] = json!(0);
    assert!(parse_combined(&coefficient)
        .unwrap_err()
        .contains("coefficient"));

    let mut layout = combined_manifest();
    layout["relation"]["layout"]["combined_logical_width"] = json!(285);
    assert!(parse_combined(&layout)
        .unwrap_err()
        .contains("combined_logical_width"));

    let mut terminal = combined_manifest();
    terminal["terminal_r1cs"]["cost"]["auxiliary_columns"] = json!(4_863);
    assert!(parse_combined(&terminal)
        .unwrap_err()
        .contains("terminal_r1cs.cost"));
}

#[test]
fn combined_emission_rejects_shared_and_witness_drift() {
    let manifest = parse_combined(&combined_manifest()).expect("valid combined manifest");
    let private = vec![F::ZERO; manifest.nebula_private_width()];

    let public = vec![F::ZERO; manifest.public_carrier_width()];
    assert!(manifest.emit(&public, |_| Some(F::ZERO), &private).is_err());

    let mut public = vec![F::ZERO; manifest.public_carrier_width()];
    public[0] = F::ONE;
    assert!(manifest.emit(&public, |_| Some(F::ZERO), &[]).is_err());
    assert!(manifest.emit(&public, |_| Some(F::ONE), &private).is_err());
}
