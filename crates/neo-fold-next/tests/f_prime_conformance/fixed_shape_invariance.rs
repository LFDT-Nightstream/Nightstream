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
    audit_rv64im_main_recursion_step_spartan_fixed_shape_across_chain,
    debug_check_rv64im_main_recursion_step_spartan_inactive_side_lane_constraints,
    debug_measure_rv64im_main_recursion_step_chunk_replay_aux_counts,
    debug_measure_rv64im_main_recursion_step_pi_ccs_aux_counts,
    debug_measure_rv64im_main_recursion_step_pi_ccs_fingerprint,
    debug_measure_rv64im_main_recursion_step_spartan_circuit_shape,
    debug_measure_rv64im_main_recursion_step_stage_aux_counts,
    debug_profile_rv64im_main_recursion_step_chunk_replay_stages, Rv64imCeClaimDigestShape,
    Rv64imMainRecursionFPrimeBackendRelation, Rv64imMainRecursionStepSpartanShape,
};
use neo_fold_next::rv64im::debug_measure_rv64im_main_recursion_step_chunk_replay_fingerprint;
use neo_fold_next::rv64im::debug_measure_rv64im_main_relation_state_in_prefix_fingerprints;
use neo_fold_next::rv64im::{
    build_rv64im_main_recursion_construction2_canonical_shape,
    build_rv64im_main_recursion_verifier_key_fs_for_step_cap, Rv64imMainRecursionPhiSide,
};
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

use super::fixed_shape_invariance_support::{
    five_step_cap_backend_relations, five_step_cap_spartan_shape, single_step_backend_relations,
    single_step_spartan_shape, two_step_backend_relations, two_step_relations, two_step_spartan_shape,
};
use super::support::{fast_structural_backend_relations, fast_structural_spartan_shape};

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
        "{label}.after_live_state_in_claim_alloc: {}",
        measured.after_live_state_in_claim_alloc
    );
    println!(
        "{label}.after_live_state_in_claim_alloc_aux: {}",
        measured.after_live_state_in_claim_alloc_aux
    );
    for (claim_index, fingerprint) in measured.per_claim_compute.iter().enumerate() {
        println!("{label}.per_claim_compute[{claim_index}]: {fingerprint}");
    }
    println!(
        "{label}.bind_me_input_digests_compute: {}",
        measured.bind_me_input_digests_compute
    );
    println!(
        "{label}.bind_me_input_digests_compute_aux: {}",
        measured.bind_me_input_digests_compute_aux
    );
    println!(
        "{label}.bind_me_input_digests_transcript: {}",
        measured.bind_me_input_digests_transcript
    );
    println!(
        "{label}.bind_me_input_digests_transcript_aux: {}",
        measured.bind_me_input_digests_transcript_aux
    );
    println!(
        "{label}.claimed_initial_sum_from_me_inputs: {}",
        measured.claimed_initial_sum_from_me_inputs
    );
    println!(
        "{label}.claimed_initial_sum_from_me_inputs_aux: {}",
        measured.claimed_initial_sum_from_me_inputs_aux
    );
    println!("{label}.fe_sumcheck_initial: {}", measured.fe_sumcheck_initial);
    println!("{label}.fe_sumcheck_initial_aux: {}", measured.fe_sumcheck_initial_aux);
    println!("{label}.fe_sumcheck: {}", measured.fe_sumcheck);
    println!("{label}.fe_sumcheck_aux: {}", measured.fe_sumcheck_aux);
    println!("{label}.nc_sumcheck_initial: {}", measured.nc_sumcheck_initial);
    println!("{label}.nc_sumcheck_initial_aux: {}", measured.nc_sumcheck_initial_aux);
    println!("{label}.nc_sumcheck: {}", measured.nc_sumcheck);
    println!("{label}.nc_sumcheck_aux: {}", measured.nc_sumcheck_aux);
    println!("{label}.relation_digest: {}", measured.relation_digest);
    println!("{label}.relation_digest_aux: {}", measured.relation_digest_aux);
    println!("{label}.ccs_outputs_and_binding: {}", measured.ccs_outputs_and_binding);
    println!(
        "{label}.ccs_outputs_and_binding_aux: {}",
        measured.ccs_outputs_and_binding_aux
    );
    println!("{label}.terminal_identities: {}", measured.terminal_identities);
    println!("{label}.terminal_identities_aux: {}", measured.terminal_identities_aux);
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

fn assert_shape_matches_canonical_contract(
    label: &str,
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    first: &Rv64imMainRecursionFPrimeBackendRelation,
) {
    let step_cap = first
        .f_prime_advice
        .verifier_key_fs()
        .step_cap()
        .unwrap_or_else(|err| panic!("{label}: derive recursive-step verifier-key step_cap: {err}"));
    let vk_fs = build_rv64im_main_recursion_verifier_key_fs_for_step_cap(step_cap)
        .unwrap_or_else(|err| panic!("{label}: build canonical recursive-step verifier-key FS: {err}"));
    let canonical_shape =
        build_rv64im_main_recursion_construction2_canonical_shape(&vk_fs, &Rv64imMainRecursionPhiSide::zero())
            .expect("build canonical Construction-2 fixed shape");
    let canonical_spartan_shape = Rv64imMainRecursionStepSpartanShape {
        cover_shape: canonical_shape.step_cover_shape,
        claim_cover: canonical_shape.claim_cover,
        rlc_zero_commit_suffix_len: spartan_shape.rlc_zero_commit_suffix_len,
        initial_transcript_in: spartan_shape.initial_transcript_in,
    };

    assert_eq!(
        spartan_shape.claim_cover, canonical_spartan_shape.claim_cover,
        "{label}: recursive-step claim-cover drifted from the canonical Construction-2 fixed shape"
    );
    assert!(
        spartan_shape
            .cover_shape
            .canonical_recursive_step_shape_equal(&canonical_spartan_shape.cover_shape),
        "{label}: recursive-step fixed-shape cover fields drifted from the canonical Construction-2 fixed shape"
    );
    assert!(
        spartan_shape
            .cover_shape
            .canonical_recursive_step_shape_equal(&first.payload.step_shape),
        "{label}: live first-step recursive-step shape fields drifted from the canonical fixed-shape cover"
    );
    assert!(
        canonical_spartan_shape
            .claim_cover
            .matches_payload(&first.payload),
        "{label}: canonical fixed-shape claim-cover no longer matches the live first-step payload surface"
    );
}

fn assert_inactive_side_lane_surface_is_zero(label: &str, relation: &Rv64imMainRecursionFPrimeBackendRelation) {
    assert_eq!(
        relation.f_prime_advice.side_witness().claim_count(),
        0,
        "{label}: inactive side-lane witness width drifted away from zero"
    );
    assert!(
        relation.payload.phi_side_commitment_words().is_empty(),
        "{label}: inactive side-lane commitment surface drifted away from zero"
    );
    debug_check_rv64im_main_recursion_step_spartan_inactive_side_lane_constraints(relation)
        .unwrap_or_else(|err| panic!("{label}: inactive side-lane zero-width subcircuit no longer holds: {err}"));
}

#[test]
fn f_prime_circuit_shape_is_n_invariant() {
    let (first, last) = audit_rv64im_main_recursion_step_spartan_fixed_shape_across_chain(two_step_relations())
        .expect("measure recursive-step fixed shape across an honest two-step chain");

    assert_eq!(
        last.num_inputs, first.num_inputs,
        "HN Construction-2 F' must keep a fixed input count across the honest recursive-step chain"
    );
    assert_eq!(
        last.num_aux, first.num_aux,
        "HN Construction-2 F' must keep a fixed aux count across the honest recursive-step chain"
    );
    assert_eq!(
        last.num_constraints, first.num_constraints,
        "HN Construction-2 F' must keep a fixed constraint count across the honest recursive-step chain"
    );
    assert_eq!(
        last.constraint_fingerprint, first.constraint_fingerprint,
        "HN Construction-2 F' must keep a fixed constraint fingerprint across the honest recursive-step chain"
    );
}

#[test]
#[ignore = "manual diagnostic: compare first-vs-last recursive-step aux counts across the honest two-step chain"]
fn f_prime_two_step_chain_stage_aux_breakdown() {
    let spartan_shape = two_step_spartan_shape();
    let backend_relations = two_step_backend_relations();
    let first = backend_relations
        .first()
        .expect("two-step chain aux breakdown requires a first backend relation");
    let last = backend_relations
        .last()
        .expect("two-step chain aux breakdown requires a last backend relation");

    let first_counts = debug_measure_rv64im_main_recursion_step_stage_aux_counts(spartan_shape, first)
        .expect("measure first recursive-step aux counts");
    let last_counts = debug_measure_rv64im_main_recursion_step_stage_aux_counts(spartan_shape, last)
        .expect("measure last recursive-step aux counts");

    println!("first={first_counts:#?}");
    println!("last={last_counts:#?}");
}

#[test]
#[ignore = "manual diagnostic: compare first-vs-last chunk-replay aux counts across the honest two-step chain"]
fn f_prime_two_step_chain_chunk_replay_aux_breakdown() {
    let backend_relations = two_step_backend_relations();
    let first = backend_relations
        .first()
        .expect("two-step chain chunk-replay breakdown requires a first backend relation");
    let last = backend_relations
        .last()
        .expect("two-step chain chunk-replay breakdown requires a last backend relation");

    let first_counts = debug_measure_rv64im_main_recursion_step_chunk_replay_aux_counts(first)
        .expect("measure first chunk-replay aux counts");
    let last_counts = debug_measure_rv64im_main_recursion_step_chunk_replay_aux_counts(last)
        .expect("measure last chunk-replay aux counts");

    println!(
        "first_surface=(chunk_steps={}, fresh_claims={}, ccs_outputs={}, child_claims={})",
        first.payload.effective_fresh_claim_count(),
        first.payload.fresh_claims.len(),
        first.payload.pi_ccs.ccs_outputs.len(),
        first.payload.pi_dec.children.len(),
    );
    println!(
        "last_surface=(chunk_steps={}, fresh_claims={}, ccs_outputs={}, child_claims={})",
        last.payload.effective_fresh_claim_count(),
        last.payload.fresh_claims.len(),
        last.payload.pi_ccs.ccs_outputs.len(),
        last.payload.pi_dec.children.len(),
    );
    println!("first={first_counts:#?}");
    println!("last={last_counts:#?}");
}

#[test]
#[ignore = "manual diagnostic: compare first-vs-last chunk-replay fingerprints across the honest two-step chain"]
fn f_prime_two_step_chain_chunk_replay_fingerprint_breakdown() {
    let backend_relations = two_step_backend_relations();
    let first = backend_relations
        .first()
        .expect("two-step chain chunk-replay fingerprint breakdown requires a first backend relation");
    let last = backend_relations
        .last()
        .expect("two-step chain chunk-replay fingerprint breakdown requires a last backend relation");

    print_state_in_chunk_replay_fingerprint("first", first);
    print_state_in_chunk_replay_fingerprint("last", last);
}

#[test]
#[ignore = "manual diagnostic: compare first-vs-last Pi_CCS fingerprints across the honest two-step chain"]
fn f_prime_two_step_chain_pi_ccs_fingerprint_breakdown() {
    let backend_relations = two_step_backend_relations();
    let first = backend_relations
        .first()
        .expect("two-step chain Pi_CCS fingerprint breakdown requires a first backend relation");
    let last = backend_relations
        .last()
        .expect("two-step chain Pi_CCS fingerprint breakdown requires a last backend relation");

    let first_fingerprints =
        debug_measure_rv64im_main_recursion_step_pi_ccs_fingerprint(first).expect("measure first Pi_CCS fingerprints");
    let last_fingerprints =
        debug_measure_rv64im_main_recursion_step_pi_ccs_fingerprint(last).expect("measure last Pi_CCS fingerprints");

    println!("first={first_fingerprints:#?}");
    println!("last={last_fingerprints:#?}");
}

#[test]
fn f_prime_circuit_shape_is_value_invariant() {
    let baseline_relation = single_step_backend_relations()
        .first()
        .expect("value-invariance requires a baseline honest recursive-step backend relation");
    let comparison_relation = two_step_backend_relations()
        .first()
        .expect("value-invariance requires a comparison honest recursive-step backend relation");
    let baseline =
        debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(single_step_spartan_shape(), baseline_relation)
            .expect("measure baseline honest recursive-step circuit shape");
    let comparison =
        debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(two_step_spartan_shape(), comparison_relation)
            .expect("measure comparison honest recursive-step circuit shape");

    assert_eq!(
        comparison.num_inputs, baseline.num_inputs,
        "HN Construction-2 F' must be value-invariant across honest traces in the same step-cap family, but num_inputs changed"
    );
    assert_eq!(
        comparison.num_aux, baseline.num_aux,
        "HN Construction-2 F' must be value-invariant across honest traces in the same step-cap family, but num_aux changed"
    );
    assert_eq!(
        comparison.num_constraints, baseline.num_constraints,
        "HN Construction-2 F' must be value-invariant across honest traces in the same step-cap family, but num_constraints changed"
    );
    assert_eq!(
        comparison.constraint_fingerprint, baseline.constraint_fingerprint,
        "HN Construction-2 F' must be value-invariant across honest traces in the same step-cap family, but the constraint fingerprint changed"
    );
}

#[test]
fn f_prime_fast_fixture_shape_builder_matches_canonical_contract() {
    let first = fast_structural_backend_relations()
        .first()
        .expect("fast-fixture canonical contract requires a backend relation");
    assert_shape_matches_canonical_contract("fast fixture", fast_structural_spartan_shape(), first);
}

#[test]
fn f_prime_single_step_shape_builder_matches_canonical_contract() {
    let first = single_step_backend_relations()
        .first()
        .expect("single-step canonical contract requires a backend relation");
    assert_shape_matches_canonical_contract("single-step fixture", single_step_spartan_shape(), first);
}

#[test]
fn f_prime_two_step_shape_builder_matches_canonical_contract() {
    let first = two_step_backend_relations()
        .first()
        .expect("two-step canonical contract requires a backend relation");
    assert_shape_matches_canonical_contract("two-step fixture", two_step_spartan_shape(), first);
}

#[test]
#[ignore = "RowsPerChunk(step_cap>1) fixed-shape terminal-padding audit is parked while the safe recursion path remains RowsPerChunk(1)"]
fn f_prime_five_step_cap_shape_builder_matches_canonical_contract() {
    let first = five_step_cap_backend_relations()
        .first()
        .expect("five-step-cap canonical contract requires a backend relation");
    assert_shape_matches_canonical_contract("five-step-cap fixture", five_step_cap_spartan_shape(), first);
}

#[test]
fn f_prime_fast_fixture_inactive_side_lane_surface_is_zero() {
    let fast_relation = fast_structural_backend_relations()
        .first()
        .expect("fast-fixture inactive side-lane check requires a backend relation");
    assert_inactive_side_lane_surface_is_zero("fast fixture", fast_relation);
}

#[test]
fn f_prime_single_step_inactive_side_lane_surface_is_zero() {
    let first = single_step_backend_relations()
        .first()
        .expect("single-step inactive side-lane check requires a backend relation");
    assert_inactive_side_lane_surface_is_zero("single-step fixture", first);
}

#[test]
fn f_prime_two_step_inactive_side_lane_surface_is_zero() {
    let first = two_step_backend_relations()
        .first()
        .expect("two-step inactive side-lane check requires a backend relation");
    assert_inactive_side_lane_surface_is_zero("two-step fixture", first);
}

#[test]
#[ignore = "RowsPerChunk(step_cap>1) fixed-shape terminal-padding audit is parked while the safe recursion path remains RowsPerChunk(1)"]
fn f_prime_five_step_cap_terminal_padding_preserves_fixed_shape() {
    let backend_relations = five_step_cap_backend_relations();
    assert!(
        backend_relations.len() >= 2,
        "five-step-cap fixture must expose at least one full-width chunk and one short terminal chunk"
    );
    let spartan_shape = five_step_cap_spartan_shape();
    let full = backend_relations
        .first()
        .expect("five-step-cap fixture must expose a first full-width chunk");
    let terminal = backend_relations
        .last()
        .expect("five-step-cap fixture must expose a final short terminal chunk");

    assert!(
        !full.payload.step_shape.terminal_step,
        "first five-step-cap backend relation must remain non-terminal"
    );
    assert!(
        terminal.payload.step_shape.terminal_step,
        "second five-step-cap backend relation must be terminal"
    );
    assert_eq!(
        full.f_prime_advice
            .verifier_key_fs()
            .step_cap()
            .expect("full step_cap"),
        5,
        "five-step-cap backend family must freeze step_cap=5"
    );
    assert_eq!(
        terminal
            .f_prime_advice
            .verifier_key_fs()
            .step_cap()
            .expect("terminal step_cap"),
        5,
        "short terminal chunk must stay inside the frozen five-step-cap family"
    );
    assert_eq!(
        full.payload.effective_fresh_claim_count(),
        5,
        "full-width five-step-cap relation must replay five effective fresh claims"
    );
    assert_eq!(
        full.payload.padded_fresh_claim_count(),
        5,
        "full-width five-step-cap relation must preserve the canonical padded fresh-claim width"
    );
    assert!(
        terminal.payload.effective_fresh_claim_count() < 5,
        "final five-step-cap relation must expose a short terminal effective fresh-claim count"
    );
    assert_eq!(
        terminal.payload.padded_fresh_claim_count(),
        5,
        "short terminal five-step-cap relation must preserve the canonical padded fresh-claim width"
    );
    assert_eq!(
        spartan_shape.cover_shape.fresh_claim_count, 5,
        "five-step-cap recursive-step cover must freeze the padded fresh-claim width at the chosen family cap"
    );
    assert!(
        spartan_shape
            .cover_shape
            .covers_recursive_step_shape(&full.payload.step_shape),
        "full-width five-step-cap relation drifted outside the canonical recursive-step cover"
    );
    assert!(
        spartan_shape
            .cover_shape
            .covers_recursive_step_shape(&terminal.payload.step_shape),
        "short terminal five-step-cap relation drifted outside the canonical recursive-step cover"
    );

    let full_measured = debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(spartan_shape, full)
        .expect("measure five-step-cap full-width recursive-step shape");
    let terminal_measured = debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(spartan_shape, terminal)
        .expect("measure five-step-cap short-terminal recursive-step shape");

    assert_eq!(
        full_measured.num_inputs, terminal_measured.num_inputs,
        "five-step-cap terminal padding must preserve recursive-step input count"
    );
    assert_eq!(
        full_measured.num_aux, terminal_measured.num_aux,
        "five-step-cap terminal padding must preserve recursive-step aux count"
    );
    assert_eq!(
        full_measured.num_constraints, terminal_measured.num_constraints,
        "five-step-cap terminal padding must preserve recursive-step constraint count"
    );
    assert_eq!(
        full_measured.constraint_fingerprint, terminal_measured.constraint_fingerprint,
        "five-step-cap terminal padding must preserve the recursive-step constraint fingerprint"
    );
}

#[test]
#[ignore = "manual Goal 2 diagnostic: isolate terminal-vs-nonterminal drift before full recursive-step setup by comparing prefix and chunk-replay fingerprints only"]
fn f_prime_terminal_vs_nonterminal_prefix_and_chunk_breakdown() {
    let fast_relation = fast_structural_backend_relations()
        .first()
        .expect("terminal/nonterminal prefix breakdown requires a fast structural backend relation");
    let single_step_relation = single_step_backend_relations()
        .first()
        .expect("terminal/nonterminal prefix breakdown requires a single-step backend relation");

    print_state_in_prefix_fingerprints("terminal_fast_live", fast_relation);
    print_state_in_prefix_fingerprints("nonterminal_single_step_live", single_step_relation);

    print_state_in_chunk_replay_fingerprint("terminal_fast_live", fast_relation);
    print_state_in_chunk_replay_fingerprint("nonterminal_single_step_live", single_step_relation);
}

#[test]
#[ignore = "manual Goal 2 diagnostic: compare terminal and non-terminal carried state-in claim surfaces without synthesizing the recursive-step circuit"]
fn f_prime_terminal_vs_nonterminal_state_in_claim_surface_breakdown() {
    let fast_relation = fast_structural_backend_relations()
        .first()
        .expect("terminal/nonterminal claim breakdown requires a fast structural backend relation");
    let single_step_relation = single_step_backend_relations()
        .first()
        .expect("terminal/nonterminal claim breakdown requires a single-step backend relation");

    assert_eq!(
        fast_relation.payload.state_in_claims.len(),
        single_step_relation.payload.state_in_claims.len(),
        "terminal and non-terminal fixtures must expose the same padded state-in claim count before claim-surface comparison"
    );

    for (claim_index, (terminal_claim, nonterminal_claim)) in fast_relation
        .payload
        .state_in_claims
        .iter()
        .zip(single_step_relation.payload.state_in_claims.iter())
        .enumerate()
    {
        let terminal_shape = Rv64imCeClaimDigestShape::from_claim(terminal_claim);
        let nonterminal_shape = Rv64imCeClaimDigestShape::from_claim(nonterminal_claim);
        if terminal_shape != nonterminal_shape
            || terminal_claim.m_in != nonterminal_claim.m_in
            || terminal_claim.u_offset != nonterminal_claim.u_offset
            || terminal_claim.u_len != nonterminal_claim.u_len
        {
            println!("claim_index={claim_index}");
            println!("terminal_shape={terminal_shape:?}");
            println!("nonterminal_shape={nonterminal_shape:?}");
            println!(
                "terminal_meta=(m_in={}, u_offset={}, u_len={})",
                terminal_claim.m_in, terminal_claim.u_offset, terminal_claim.u_len
            );
            println!(
                "nonterminal_meta=(m_in={}, u_offset={}, u_len={})",
                nonterminal_claim.m_in, nonterminal_claim.u_offset, nonterminal_claim.u_len
            );
        }
    }
}

#[test]
#[ignore = "manual Goal 2 diagnostic: compare recursive-step aux counts by top-level stage across full-width and short-terminal five-step-cap relations"]
fn f_prime_five_step_cap_terminal_padding_stage_aux_breakdown() {
    let spartan_shape = five_step_cap_spartan_shape();
    let backend_relations = five_step_cap_backend_relations();
    let full = backend_relations
        .iter()
        .find(|relation| !relation.payload.step_shape.terminal_step)
        .expect("five-step-cap aux breakdown requires a non-terminal full-width relation");
    let terminal = backend_relations
        .iter()
        .rev()
        .find(|relation| relation.payload.step_shape.terminal_step)
        .expect("five-step-cap aux breakdown requires a short terminal relation");

    let full_counts = debug_measure_rv64im_main_recursion_step_stage_aux_counts(spartan_shape, full)
        .expect("measure full-width recursive-step aux counts");
    let terminal_counts = debug_measure_rv64im_main_recursion_step_stage_aux_counts(spartan_shape, terminal)
        .expect("measure short-terminal recursive-step aux counts");

    println!("full={full_counts:#?}");
    println!("terminal={terminal_counts:#?}");
}

#[test]
#[ignore = "manual Goal 2 diagnostic: compare chunk-replay aux counts by checkpoint across full-width and short-terminal five-step-cap relations"]
fn f_prime_five_step_cap_terminal_padding_chunk_replay_aux_breakdown() {
    let backend_relations = five_step_cap_backend_relations();
    let full = backend_relations
        .iter()
        .find(|relation| !relation.payload.step_shape.terminal_step)
        .expect("five-step-cap chunk-replay aux breakdown requires a non-terminal full-width relation");
    let terminal = backend_relations
        .iter()
        .rev()
        .find(|relation| relation.payload.step_shape.terminal_step)
        .expect("five-step-cap chunk-replay aux breakdown requires a short terminal relation");

    let full_counts = debug_measure_rv64im_main_recursion_step_chunk_replay_aux_counts(full)
        .expect("measure full-width chunk-replay aux counts");
    let terminal_counts = debug_measure_rv64im_main_recursion_step_chunk_replay_aux_counts(terminal)
        .expect("measure short-terminal chunk-replay aux counts");

    println!(
        "full_surface=(fresh_claims={}, ccs_outputs={}, effective_fresh={}, padded_fresh={})",
        full.payload.fresh_claims.len(),
        full.payload.pi_ccs.ccs_outputs.len(),
        full.payload.effective_fresh_claim_count(),
        full.payload.padded_fresh_claim_count(),
    );
    println!(
        "terminal_surface=(fresh_claims={}, ccs_outputs={}, effective_fresh={}, padded_fresh={})",
        terminal.payload.fresh_claims.len(),
        terminal.payload.pi_ccs.ccs_outputs.len(),
        terminal.payload.effective_fresh_claim_count(),
        terminal.payload.padded_fresh_claim_count(),
    );
    println!("full={full_counts:#?}");
    println!("terminal={terminal_counts:#?}");
}

#[test]
#[ignore = "manual Goal 2 diagnostic: compare pi_ccs aux counts by sub-stage across full-width and short-terminal five-step-cap relations"]
fn f_prime_five_step_cap_terminal_padding_pi_ccs_aux_breakdown() {
    let backend_relations = five_step_cap_backend_relations();
    let full = backend_relations
        .iter()
        .find(|relation| !relation.payload.step_shape.terminal_step)
        .expect("five-step-cap pi_ccs aux breakdown requires a non-terminal full-width relation");
    let terminal = backend_relations
        .iter()
        .rev()
        .find(|relation| relation.payload.step_shape.terminal_step)
        .expect("five-step-cap pi_ccs aux breakdown requires a short terminal relation");

    let full_counts =
        debug_measure_rv64im_main_recursion_step_pi_ccs_aux_counts(full).expect("measure full-width pi_ccs aux counts");
    let terminal_counts = debug_measure_rv64im_main_recursion_step_pi_ccs_aux_counts(terminal)
        .expect("measure short-terminal pi_ccs aux counts");

    println!("full={full_counts:#?}");
    println!("terminal={terminal_counts:#?}");
}

#[test]
#[ignore = "manual Goal 2 diagnostic: compare pi_ccs fingerprints by sub-stage across full-width and short-terminal five-step-cap relations"]
fn f_prime_five_step_cap_terminal_padding_pi_ccs_fingerprint_breakdown() {
    let backend_relations = five_step_cap_backend_relations();
    let full = backend_relations
        .iter()
        .find(|relation| !relation.payload.step_shape.terminal_step)
        .expect("five-step-cap pi_ccs fingerprint breakdown requires a non-terminal full-width relation");
    let terminal = backend_relations
        .iter()
        .rev()
        .find(|relation| relation.payload.step_shape.terminal_step)
        .expect("five-step-cap pi_ccs fingerprint breakdown requires a short terminal relation");

    let full_fingerprints = debug_measure_rv64im_main_recursion_step_pi_ccs_fingerprint(full)
        .expect("measure full-width pi_ccs fingerprints");
    let terminal_fingerprints = debug_measure_rv64im_main_recursion_step_pi_ccs_fingerprint(terminal)
        .expect("measure short-terminal pi_ccs fingerprints");

    println!("full={full_fingerprints:#?}");
    println!("terminal={terminal_fingerprints:#?}");
}

#[test]
#[ignore = "manual Goal 2 diagnostic: compare chunk-replay fingerprints across full-width and short-terminal five-step-cap relations"]
fn f_prime_five_step_cap_terminal_padding_chunk_replay_fingerprint_breakdown() {
    let backend_relations = five_step_cap_backend_relations();
    let full = backend_relations
        .iter()
        .find(|relation| !relation.payload.step_shape.terminal_step)
        .expect("five-step-cap fingerprint breakdown requires a non-terminal full-width relation");
    let terminal = backend_relations
        .iter()
        .rev()
        .find(|relation| relation.payload.step_shape.terminal_step)
        .expect("five-step-cap fingerprint breakdown requires a short terminal relation");

    print_state_in_chunk_replay_fingerprint("full", full);
    print_state_in_chunk_replay_fingerprint("terminal", terminal);
}

#[test]
#[ignore = "manual Goal 2 diagnostic: compare pi_ccs sub-stage fingerprints across full-width and short-terminal five-step-cap relations"]
fn f_prime_five_step_cap_terminal_padding_prefix_breakdown() {
    let backend_relations = five_step_cap_backend_relations();
    let full = backend_relations
        .iter()
        .find(|relation| !relation.payload.step_shape.terminal_step)
        .expect("five-step-cap prefix breakdown requires a non-terminal full-width relation");
    let terminal = backend_relations
        .iter()
        .rev()
        .find(|relation| relation.payload.step_shape.terminal_step)
        .expect("five-step-cap prefix breakdown requires a short terminal relation");

    print_state_in_prefix_fingerprints("full", full);
    print_state_in_prefix_fingerprints("terminal", terminal);
}

#[test]
#[ignore = "manual Goal 2 diagnostic: compare five-step-cap padded fresh claims and CCS outputs across full-width and short-terminal relations"]
fn f_prime_five_step_cap_terminal_padding_payload_surface_breakdown() {
    let backend_relations = five_step_cap_backend_relations();
    let full = backend_relations
        .iter()
        .find(|relation| !relation.payload.step_shape.terminal_step)
        .expect("five-step-cap payload surface breakdown requires a non-terminal full-width relation");
    let terminal = backend_relations
        .iter()
        .rev()
        .find(|relation| relation.payload.step_shape.terminal_step)
        .expect("five-step-cap payload surface breakdown requires a short terminal relation");

    for (idx, (full_claim, terminal_claim)) in full
        .payload
        .fresh_claims
        .iter()
        .zip(terminal.payload.fresh_claims.iter())
        .enumerate()
    {
        println!(
            "fresh_claim[{idx}] full=(c_data={}, x_len={}, m_in={}) terminal=(c_data={}, x_len={}, m_in={})",
            full_claim.c.data.len(),
            full_claim.x.len(),
            full_claim.m_in,
            terminal_claim.c.data.len(),
            terminal_claim.x.len(),
            terminal_claim.m_in,
        );
    }

    for (idx, (full_output, terminal_output)) in full
        .payload
        .pi_ccs
        .ccs_outputs
        .iter()
        .zip(terminal.payload.pi_ccs.ccs_outputs.iter())
        .enumerate()
        .take(full.payload.padded_fresh_claim_count())
    {
        let full_y_ring_total: usize = full_output.y_ring.iter().map(Vec::len).sum();
        let terminal_y_ring_total: usize = terminal_output.y_ring.iter().map(Vec::len).sum();
        println!(
            "ccs_output[{idx}] full=(x_rows={}, x_cols={}, y_ring_rows={}, y_ring_total={}, ct_len={}, aux_openings_len={}, y_zcol_len={}, c_step_coords_len={}) terminal=(x_rows={}, x_cols={}, y_ring_rows={}, y_ring_total={}, ct_len={}, aux_openings_len={}, y_zcol_len={}, c_step_coords_len={})",
            full_output.X.rows(),
            full_output.X.cols(),
            full_output.y_ring.len(),
            full_y_ring_total,
            full_output.ct.len(),
            full_output.aux_openings.len(),
            full_output.y_zcol.len(),
            full_output.c_step_coords.len(),
            terminal_output.X.rows(),
            terminal_output.X.cols(),
            terminal_output.y_ring.len(),
            terminal_y_ring_total,
            terminal_output.ct.len(),
            terminal_output.aux_openings.len(),
            terminal_output.y_zcol.len(),
            terminal_output.c_step_coords.len(),
        );
    }
}

#[test]
#[ignore = "manual Goal 2 diagnostic: profile padded recursive-step chunk replay stages for five-step-cap full-width and short-terminal relations"]
fn f_prime_five_step_cap_terminal_padding_padded_stage_profile() {
    let backend_relations = five_step_cap_backend_relations();
    let full = backend_relations
        .iter()
        .find(|relation| !relation.payload.step_shape.terminal_step)
        .expect("five-step-cap padded stage profile requires a non-terminal full-width relation");
    let terminal = backend_relations
        .iter()
        .rev()
        .find(|relation| relation.payload.step_shape.terminal_step)
        .expect("five-step-cap padded stage profile requires a short terminal relation");

    println!("full_profile_start");
    debug_profile_rv64im_main_recursion_step_chunk_replay_stages(full).expect("profile full-width padded chunk replay");
    println!("terminal_profile_start");
    debug_profile_rv64im_main_recursion_step_chunk_replay_stages(terminal)
        .expect("profile short-terminal padded chunk replay");
}

#[test]
fn f_prime_shape_only_setup_skeleton_matches_live_first_step_shape_contract() {
    let first = fast_structural_backend_relations()
        .first()
        .expect("shape-only/live structural equivalence requires one backend relation");
    assert_shape_matches_canonical_contract("shape-only setup skeleton", fast_structural_spartan_shape(), first);
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
