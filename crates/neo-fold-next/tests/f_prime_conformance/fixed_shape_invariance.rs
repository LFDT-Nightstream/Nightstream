//! HyperNova Construction 2 §6.3 fixed-shape F' discipline: the recursive-step
//! circuit shape (input/aux/constraint counts and constraint fingerprint) must
//! be invariant across step indices. Any step-dependent blow-up — constraint
//! count growing with `chunk_count_in`, auxiliary variables scaling with step
//! position, etc. — would mean the circuit is not actually fixed-shape and
//! would violate the paper-level guarantee that F' is a single compiled
//! circuit reused at every recursion step.

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsWitness, CeClaim};
use neo_fold_next::rv64im::audit::{
    audit_rv64im_main_recursion_step_spartan_fixed_shape_at_chunk_positions,
    debug_measure_rv64im_main_recursion_step_spartan_circuit_shape, Rv64imMainRecursionFPrimeBackendRelation,
    Rv64imMainRecursionStepSpartanShape,
};
use neo_fold_next::rv64im::debug_measure_rv64im_main_recursion_step_chunk_replay_fingerprint;
use neo_fold_next::rv64im::debug_measure_rv64im_main_relation_state_in_prefix_fingerprints;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

use super::support::{fast_structural_backend_relations, fast_structural_relations, fast_structural_spartan_shape};

fn perturb_ce_claim_values(claim: &mut CeClaim<Commitment, F, K>) {
    if let Some(first) = claim.c.data.first_mut() {
        *first += F::ONE;
    }
    if claim.X.rows() > 0 && claim.X.cols() > 0 {
        claim.X[(0, 0)] += F::ONE;
    }
    if let Some(first) = claim.r.first_mut() {
        *first += K::ONE;
    }
    if let Some(first) = claim.s_col.first_mut() {
        *first += K::ONE;
    }
    if let Some(row) = claim.y_ring.first_mut() {
        if let Some(first) = row.first_mut() {
            *first += K::ONE;
        }
    }
    if let Some(first) = claim.ct.first_mut() {
        *first += K::ONE;
    }
    if let Some(first) = claim.aux_openings.first_mut() {
        *first += K::ONE;
    }
    if let Some(first) = claim.y_zcol.first_mut() {
        *first += K::ONE;
    }
    if let Some(first) = claim.c_step_coords.first_mut() {
        *first += F::ONE;
    }
    claim.fold_digest[0] ^= 1;
}

fn perturb_ccs_claim_values(claim: &mut CcsClaim<Commitment, F>) {
    if let Some(first) = claim.c.data.first_mut() {
        *first += F::ONE;
    }
    if let Some(first) = claim.x.first_mut() {
        *first += F::ONE;
    }
}

fn perturb_ccs_witness_values(witness: &mut CcsWitness<F>) {
    if let Some(first) = witness.w.first_mut() {
        *first += F::ONE;
    }
    if witness.Z.rows() > 0 && witness.Z.cols() > 0 {
        witness.Z[(0, 0)] += F::ONE;
    }
}

fn perturb_state_in_r_values(relation: &mut Rv64imMainRecursionFPrimeBackendRelation) {
    for claim in &mut relation.payload.state_in_claims {
        if let Some(first) = claim.r.first_mut() {
            *first += K::ONE;
        }
    }
}

fn perturb_state_in_s_col_values(relation: &mut Rv64imMainRecursionFPrimeBackendRelation) {
    for claim in &mut relation.payload.state_in_claims {
        if let Some(first) = claim.s_col.first_mut() {
            *first += K::ONE;
        }
    }
}

fn perturb_state_in_y_ring_values(relation: &mut Rv64imMainRecursionFPrimeBackendRelation) {
    for claim in &mut relation.payload.state_in_claims {
        if let Some(row) = claim.y_ring.first_mut() {
            if let Some(first) = row.first_mut() {
                *first += K::ONE;
            }
        }
    }
}

fn perturb_backend_relation_values(relation: &mut Rv64imMainRecursionFPrimeBackendRelation) {
    for claim in &mut relation.payload.state_in_claims {
        perturb_ce_claim_values(claim);
    }
    for claim in &mut relation.payload.state_out_claims {
        perturb_ce_claim_values(claim);
    }
    for claim in &mut relation.payload.pi_ccs.ccs_outputs {
        perturb_ce_claim_values(claim);
    }
    perturb_ce_claim_values(&mut relation.payload.pi_rlc.parent);
    for child in &mut relation.payload.pi_dec.children {
        perturb_ce_claim_values(child);
    }
    for claim in &mut relation.payload.fresh_claims {
        perturb_ccs_claim_values(claim);
    }
    for witness in &mut relation.payload.fresh_witnesses {
        perturb_ccs_witness_values(witness);
    }
}

fn measure_family_perturbation(
    label: &str,
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    baseline_relation: &Rv64imMainRecursionFPrimeBackendRelation,
    mutate: impl FnOnce(&mut Rv64imMainRecursionFPrimeBackendRelation),
) -> String {
    let mut relation = baseline_relation.clone();
    mutate(&mut relation);
    let measured = debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(spartan_shape, &relation)
        .unwrap_or_else(|err| panic!("measure {label} perturbation: {err}"));
    println!("{label}: {}", measured.constraint_fingerprint);
    measured.constraint_fingerprint
}

fn print_state_in_prefix_fingerprints(label: &str, relation: &Rv64imMainRecursionFPrimeBackendRelation) {
    let measured = debug_measure_rv64im_main_relation_state_in_prefix_fingerprints(relation)
        .unwrap_or_else(|err| panic!("measure {label} state_in prefix fingerprints: {err}"));
    println!(
        "{label}.bind_me_input_digests_compute: {}",
        measured.bind_me_input_digests_compute
    );
    println!(
        "{label}.bind_me_input_digests_transcript: {}",
        measured.bind_me_input_digests_transcript
    );
    println!(
        "{label}.claimed_initial_sum_from_me_inputs: {}",
        measured.claimed_initial_sum_from_me_inputs
    );
}

fn run_state_in_prefix_breakdown_case(label: &str, mutate: impl FnOnce(&mut Rv64imMainRecursionFPrimeBackendRelation)) {
    let backend_relations = fast_structural_backend_relations();
    let baseline_relation = backend_relations
        .first()
        .expect("state_in prefix breakdown requires at least one recursive-step backend relation");
    print_state_in_prefix_fingerprints("baseline", baseline_relation);

    let mut mutated = baseline_relation.clone();
    mutate(&mut mutated);
    print_state_in_prefix_fingerprints(label, &mutated);
}

fn print_state_in_chunk_replay_fingerprint(label: &str, relation: &Rv64imMainRecursionFPrimeBackendRelation) {
    let measured = debug_measure_rv64im_main_recursion_step_chunk_replay_fingerprint(relation)
        .unwrap_or_else(|err| panic!("measure {label} chunk replay fingerprint: {err}"));
    println!("{label}.after_state_cover: {}", measured.after_state_cover);
    println!("{label}.after_chunk_meta: {}", measured.after_chunk_meta);
    println!("{label}.after_pi_ccs: {}", measured.after_pi_ccs);
    println!(
        "{label}.after_synthetic_relation_io: {}",
        measured.after_synthetic_relation_io
    );
    println!(
        "{label}.after_pi_rlc_parent_claim: {}",
        measured.after_pi_rlc_parent_claim
    );
    println!("{label}.after_pi_rlc_rhos: {}", measured.after_pi_rlc_rhos);
    println!("{label}.after_pi_rlc_rho_mats: {}", measured.after_pi_rlc_rho_mats);
    println!("{label}.after_pi_rlc_public: {}", measured.after_pi_rlc_public);
    println!("{label}.after_pi_rlc: {}", measured.after_pi_rlc);
    println!("{label}.after_chunk_body: {}", measured.after_chunk_body);
    println!("{label}.after_chunk_replay: {}", measured.after_chunk_replay);
}

fn run_state_in_chunk_replay_breakdown_case(
    label: &str,
    mutate: impl FnOnce(&mut Rv64imMainRecursionFPrimeBackendRelation),
) {
    let backend_relations = fast_structural_backend_relations();
    let baseline_relation = backend_relations
        .first()
        .expect("state_in chunk replay breakdown requires at least one recursive-step backend relation");
    print_state_in_chunk_replay_fingerprint("baseline", baseline_relation);

    let mut mutated = baseline_relation.clone();
    mutate(&mut mutated);
    print_state_in_chunk_replay_fingerprint(label, &mutated);
}

#[test]
fn f_prime_circuit_shape_is_n_invariant() {
    let measured =
        audit_rv64im_main_recursion_step_spartan_fixed_shape_at_chunk_positions(fast_structural_relations(), &[0, 1])
            .expect("measure recursive-step circuit shape across chunk positions");
    assert!(
        measured.len() >= 2,
        "N-invariance requires at least two measured chunk positions; fixture produced {}",
        measured.len()
    );
    let baseline = &measured[0].2;

    for (probe_index, (chunk_count_in, _, measured)) in measured.iter().enumerate().skip(1) {
        assert_eq!(
            measured.num_inputs, baseline.num_inputs,
            "probe {probe_index} (chunk_count_in={chunk_count_in}): HN Construction-2 F' must be fixed-shape, but num_inputs drifted from the baseline shape"
        );
        assert_eq!(
            measured.num_aux, baseline.num_aux,
            "probe {probe_index} (chunk_count_in={chunk_count_in}): HN Construction-2 F' must be fixed-shape, but num_aux drifted from the baseline shape"
        );
        assert_eq!(
            measured.num_constraints, baseline.num_constraints,
            "probe {probe_index} (chunk_count_in={chunk_count_in}): HN Construction-2 F' must be fixed-shape, but num_constraints drifted from the baseline shape"
        );
        assert_eq!(
            measured.constraint_fingerprint, baseline.constraint_fingerprint,
            "probe {probe_index} (chunk_count_in={chunk_count_in}): HN Construction-2 F' must be fixed-shape, but the constraint fingerprint drifted from the baseline shape"
        );
    }
}

#[test]
fn f_prime_circuit_shape_is_value_invariant() {
    let backend_relations = fast_structural_backend_relations();
    let spartan_shape = fast_structural_spartan_shape();
    let baseline_relation = backend_relations
        .first()
        .expect("value-invariance requires at least one recursive-step backend relation");
    let baseline = debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(spartan_shape, baseline_relation)
        .expect("measure baseline recursive-step circuit shape");

    let mut perturbed_relation = baseline_relation.clone();
    perturb_backend_relation_values(&mut perturbed_relation);

    let perturbed = debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(spartan_shape, &perturbed_relation)
        .expect("measure value-perturbed recursive-step circuit shape");

    assert_eq!(
        perturbed.num_inputs, baseline.num_inputs,
        "HN Construction-2 F' must be value-invariant, but num_inputs changed when only recursive-step payload values changed"
    );
    assert_eq!(
        perturbed.num_aux, baseline.num_aux,
        "HN Construction-2 F' must be value-invariant, but num_aux changed when only recursive-step payload values changed"
    );
    assert_eq!(
        perturbed.num_constraints, baseline.num_constraints,
        "HN Construction-2 F' must be value-invariant, but num_constraints changed when only recursive-step payload values changed"
    );
    assert_eq!(
        perturbed.constraint_fingerprint, baseline.constraint_fingerprint,
        "HN Construction-2 F' must be value-invariant, but the constraint fingerprint changed when only recursive-step payload values changed"
    );
}

#[test]
#[ignore = "manual Goal 2 diagnostic: isolate remaining value-dependent recursive-step fingerprint drift by payload family"]
fn f_prime_circuit_shape_value_invariant_family_breakdown() {
    let backend_relations = fast_structural_backend_relations();
    let spartan_shape = fast_structural_spartan_shape();
    let baseline_relation = backend_relations
        .first()
        .expect("value-invariance family breakdown requires at least one recursive-step backend relation");
    let baseline = debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(spartan_shape, baseline_relation)
        .expect("measure baseline recursive-step circuit shape");
    println!("baseline: {}", baseline.constraint_fingerprint);

    let families = [
        measure_family_perturbation("state_in_claims", spartan_shape, baseline_relation, |relation| {
            for claim in &mut relation.payload.state_in_claims {
                perturb_ce_claim_values(claim);
            }
        }),
        measure_family_perturbation("state_out_claims", spartan_shape, baseline_relation, |relation| {
            for claim in &mut relation.payload.state_out_claims {
                perturb_ce_claim_values(claim);
            }
        }),
        measure_family_perturbation("pi_ccs_outputs", spartan_shape, baseline_relation, |relation| {
            for claim in &mut relation.payload.pi_ccs.ccs_outputs {
                perturb_ce_claim_values(claim);
            }
        }),
        measure_family_perturbation("pi_rlc_parent", spartan_shape, baseline_relation, |relation| {
            perturb_ce_claim_values(&mut relation.payload.pi_rlc.parent);
        }),
        measure_family_perturbation("pi_dec_children", spartan_shape, baseline_relation, |relation| {
            for child in &mut relation.payload.pi_dec.children {
                perturb_ce_claim_values(child);
            }
        }),
        measure_family_perturbation("fresh_claims", spartan_shape, baseline_relation, |relation| {
            for claim in &mut relation.payload.fresh_claims {
                perturb_ccs_claim_values(claim);
            }
        }),
        measure_family_perturbation("fresh_witnesses", spartan_shape, baseline_relation, |relation| {
            for witness in &mut relation.payload.fresh_witnesses {
                perturb_ccs_witness_values(witness);
            }
        }),
    ];

    assert!(
        families
            .iter()
            .any(|fingerprint| fingerprint != &baseline.constraint_fingerprint),
        "family breakdown must expose at least one drifting payload family while Goal 2 is still open"
    );
}

#[test]
#[ignore = "manual Goal 2 diagnostic: isolate the remaining carried-claim state_in fingerprint drift by subfield"]
fn f_prime_circuit_shape_state_in_subfamily_breakdown() {
    let backend_relations = fast_structural_backend_relations();
    let spartan_shape = fast_structural_spartan_shape();
    let baseline_relation = backend_relations
        .first()
        .expect("state_in subfamily breakdown requires at least one recursive-step backend relation");
    let baseline = debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(spartan_shape, baseline_relation)
        .expect("measure baseline recursive-step circuit shape");
    println!("baseline: {}", baseline.constraint_fingerprint);

    let r_only = measure_family_perturbation("state_in_r_only", spartan_shape, baseline_relation, |relation| {
        perturb_state_in_r_values(relation);
    });
    let s_col_only = measure_family_perturbation("state_in_s_col_only", spartan_shape, baseline_relation, |relation| {
        perturb_state_in_s_col_values(relation);
    });
    let y_ring_only =
        measure_family_perturbation("state_in_y_ring_only", spartan_shape, baseline_relation, |relation| {
            perturb_state_in_y_ring_values(relation);
        });

    assert!(
        [r_only, s_col_only, y_ring_only]
            .iter()
            .any(|fingerprint| fingerprint != &baseline.constraint_fingerprint),
        "state_in subfamily breakdown must expose at least one drifting carried-claim subfield while Goal 2 is still open"
    );
}

#[test]
#[ignore = "manual Goal 2 diagnostic: isolate whether state_in drift originates in ME-input digest binding or FE initial-sum"]
fn f_prime_circuit_shape_state_in_prefix_breakdown() {
    run_state_in_prefix_breakdown_case("state_in_r_only", perturb_state_in_r_values);
    run_state_in_prefix_breakdown_case("state_in_s_col_only", perturb_state_in_s_col_values);
}

#[test]
#[ignore = "manual Goal 2 diagnostic: isolate whether state_in r drift starts in the carried-claim prefix"]
fn f_prime_circuit_shape_state_in_prefix_breakdown_r_only() {
    run_state_in_prefix_breakdown_case("state_in_r_only", perturb_state_in_r_values);
}

#[test]
#[ignore = "manual Goal 2 diagnostic: isolate whether state_in s_col drift starts in the carried-claim prefix"]
fn f_prime_circuit_shape_state_in_prefix_breakdown_s_col_only() {
    run_state_in_prefix_breakdown_case("state_in_s_col_only", perturb_state_in_s_col_values);
}

#[test]
#[ignore = "manual Goal 2 diagnostic: isolate whether state_in r drift is already present across the chunk replay bridge"]
fn f_prime_circuit_shape_state_in_chunk_replay_breakdown_r_only() {
    run_state_in_chunk_replay_breakdown_case("state_in_r_only", perturb_state_in_r_values);
}

#[test]
#[ignore = "manual Goal 2 diagnostic: isolate whether state_in s_col drift is already present across the chunk replay bridge"]
fn f_prime_circuit_shape_state_in_chunk_replay_breakdown_s_col_only() {
    run_state_in_chunk_replay_breakdown_case("state_in_s_col_only", perturb_state_in_s_col_values);
}
