#[path = "../support/mod.rs"]
mod support;

use neo_ajtai::{get_global_pp_for_dims, precompute_rot_columns, PP};
use neo_ccs::{traits::SModuleHomomorphism, CcsStructure, Mat, SparsePoly};
use neo_fold_clean::paper::construction2::ProofState;
use neo_math::ring::Rq as RqEl;
use neo_math::KExtensions;
use neo_math::{D, F};
use neo_reductions::common::{compute_v1_1_evaluations_from_z_and_r, project_x_from_witness_mat};
use p3_field::{Field, PrimeCharacteristicRing};

fn toy_instance_with_x_value(prep: &neo_fold_clean::Preprocessing, x: neo_math::F) -> neo_fold_clean::CcsInstance {
    let mut z = vec![neo_math::F::ZERO; prep.structure().m];
    z[0] = x;
    neo_fold_clean::CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &z, D)
        .expect("toy low-norm CCS instance with chosen public input")
}

fn wide_kernel_preprocessing() -> neo_fold_clean::Preprocessing {
    let cols = neo_fold_clean::config::KAPPA as usize + 1;
    let vars = D * cols;
    let structure = CcsStructure::new(vec![Mat::zero(1, vars, F::ZERO)], SparsePoly::new(1, vec![])).expect("wide CCS");
    let params = neo_fold_clean::config::ccs_params(structure.n, structure.m, 1, 1).expect("wide-kernel params");
    support::install_ajtai_module(&params, &structure);
    neo_fold_clean::preprocess(params, structure, Some(D)).expect("wide-kernel preprocessing")
}

fn ajtai_row_major_rows(pp: &PP<RqEl>) -> Vec<Vec<F>> {
    let d = pp.d;
    let cols = pp.m;
    let mut rows = Vec::with_capacity(d * pp.kappa);
    for commit_col in 0..pp.kappa {
        for commit_row in 0..d {
            let mut row = vec![F::ZERO; d * cols];
            for col in 0..cols {
                let mut rot_cols = vec![[F::ZERO; D]; D].into_boxed_slice();
                precompute_rot_columns(pp.m_rows[commit_col][col], &mut rot_cols);
                for input_row in 0..d {
                    row[input_row * cols + col] = rot_cols[input_row][commit_row];
                }
            }
            rows.push(row);
        }
    }
    rows
}

fn right_kernel_vector(mut rows: Vec<Vec<F>>) -> Vec<F> {
    let row_count = rows.len();
    let col_count = rows
        .first()
        .expect("kernel finder needs at least one row")
        .len();
    let mut pivot_cols = Vec::new();
    let mut pivot_row = 0usize;

    for col in 0..col_count {
        let Some(pivot) = (pivot_row..row_count).find(|&row| rows[row][col] != F::ZERO) else {
            continue;
        };
        rows.swap(pivot_row, pivot);
        let inv = rows[pivot_row][col].inverse();
        for entry in &mut rows[pivot_row][col..] {
            *entry *= inv;
        }
        let normalized = rows[pivot_row].clone();
        for row in (pivot_row + 1)..row_count {
            let factor = rows[row][col];
            if factor == F::ZERO {
                continue;
            }
            for j in col..col_count {
                rows[row][j] -= factor * normalized[j];
            }
        }
        pivot_cols.push(col);
        pivot_row += 1;
        if pivot_row == row_count {
            break;
        }
    }

    let free_col = (0..col_count)
        .find(|col| !pivot_cols.contains(col))
        .expect("wide Ajtai map must have a nontrivial right kernel");
    let mut vector = vec![F::ZERO; col_count];
    vector[free_col] = F::ONE;
    for (row, &pivot_col) in pivot_cols.iter().enumerate().rev() {
        let mut sum = F::ZERO;
        for col in (pivot_col + 1)..col_count {
            sum += rows[row][col] * vector[col];
        }
        vector[pivot_col] = -sum;
    }

    assert!(
        vector.iter().any(|&entry| entry != F::ZERO),
        "kernel vector must be nonzero"
    );
    vector
}

fn commitment_kernel_vector(prep: &neo_fold_clean::Preprocessing) -> Vec<F> {
    let pp = get_global_pp_for_dims(D, prep.structure().m.div_ceil(D)).expect("Ajtai PP for prep");
    let rows = ajtai_row_major_rows(&pp);
    let vector = right_kernel_vector(rows.clone());
    for row in rows {
        let dot = row
            .iter()
            .zip(&vector)
            .fold(F::ZERO, |acc, (&a, &b)| acc + a * b);
        assert_eq!(dot, F::ZERO, "computed vector must be in the Ajtai kernel");
    }
    vector
}

fn one_batch_proof(prep: &neo_fold_clean::Preprocessing, value: neo_math::F) -> neo_fold_clean::Uncompressed {
    let proof =
        neo_fold_clean::prove(prep, vec![vec![toy_instance_with_x_value(prep, value)]]).expect("one-batch proof");
    neo_fold_clean::finish_uncompressed(prep, proof).expect("finish one-batch proof")
}

fn final_running(proof: &neo_fold_clean::Uncompressed) -> neo_fold_clean::RunningInstance {
    match &proof.state.proof {
        ProofState::Active { running, latest } => {
            assert!(
                latest.instances.is_empty(),
                "red-team helper expects a finalized proof with empty latest"
            );
            running.clone()
        }
        ProofState::Initial => panic!("test helper requires a finalized Active proof"),
    }
}

fn final_running_mut(proof: &mut neo_fold_clean::Uncompressed) -> &mut neo_fold_clean::RunningInstance {
    match &mut proof.state.proof {
        ProofState::Active { running, latest } => {
            assert!(
                latest.instances.is_empty(),
                "red-team helper expects a finalized proof with empty latest"
            );
            running
        }
        ProofState::Initial => panic!("test helper requires a finalized Active proof"),
    }
}

fn recompute_active_running_acc_digest(proof: &neo_fold_clean::Uncompressed) -> [u8; 32] {
    let running = final_running(proof);
    if running.claims.is_empty() {
        neo_fold_clean::paper::digest::AccumulatorHandle::empty().digest()
    } else {
        let parent = running
            .parent_authority
            .as_ref()
            .expect("non-empty running must carry parent authority");
        neo_fold_clean::paper::digest::AccumulatorHandle::from_running_parts(2, &running.claims, Some(parent)).digest()
    }
}

fn running_acc_digest(running: &neo_fold_clean::RunningInstance) -> [u8; 32] {
    if running.claims.is_empty() {
        neo_fold_clean::paper::digest::AccumulatorHandle::empty().digest()
    } else {
        let parent = running
            .parent_authority
            .as_ref()
            .expect("non-empty running must carry parent authority");
        neo_fold_clean::paper::digest::AccumulatorHandle::from_running_parts(2, &running.claims, Some(parent)).digest()
    }
}

fn final_running_passes_witness_authority(prep: &neo_fold_clean::Preprocessing, proof: &neo_fold_clean::Uncompressed) {
    let running = final_running(proof);
    neo_fold_clean::lifecycle::validate_final_witness_authority(prep, &running)
        .expect("final running accumulator must remain locally valid under terminal CE authority");
}

#[test]
fn verify_uncompressed_audit_rejects_commitment_kernel_terminal_witness_forge() {
    let prep = wide_kernel_preprocessing();
    let instance = neo_fold_clean::CcsInstance::from_low_norm_assignment(
        &prep.params,
        &prep.log,
        prep.structure(),
        &vec![F::ZERO; prep.structure().m],
        D,
    )
    .expect("zero wide instance");
    let audit = neo_fold_clean::prove(&prep, vec![vec![instance]]).expect("prove wide instance");
    let mut finished = neo_fold_clean::finish_uncompressed_with_audit(&prep, audit).expect("finish wide audit");
    neo_fold_clean::verify_uncompressed_audit(&prep, &finished).expect("honest audit verifies");

    let delta = commitment_kernel_vector(&prep);
    {
        let running = final_running_mut(&mut finished.proof);
        let witness = running
            .witnesses
            .get_mut(0)
            .expect("test fixture must carry a terminal witness");
        let rows = witness.rows();
        let columns = witness.cols();
        let mut dense = witness.to_dense_vec();
        assert_eq!(
            dense.len(),
            delta.len(),
            "kernel vector must match packed witness length"
        );
        let before = prep.log.commit(witness);
        for (entry, delta) in dense.iter_mut().zip(delta) {
            *entry += delta;
        }
        *witness = Mat::from_row_major(rows, columns, dense);
        assert_eq!(
            prep.log.commit(witness),
            before,
            "test setup must mutate inside the verifier-owned Ajtai commitment kernel"
        );
    }

    let err = neo_fold_clean::verify_uncompressed_audit(&prep, &finished)
        .expect_err("audit verifier accepted a same-commitment forged terminal witness");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::FinalAccumulatorLowNormViolation { .. }
                | neo_fold_clean::Error::FinalAccumulatorPublicInputMismatch { .. }
                | neo_fold_clean::Error::FinalAccumulatorCeRelationViolation { .. }
        ),
        "commitment-kernel witness forge must be rejected by terminal authority, got {err:?}"
    );
}

/// Self-consistent final-running relabel attack on `Eval_A`.
///
/// Hacker model: mutate one genuine-matrix evaluation in the recorded
/// terminal running claim, then recompute `proof.state.acc_digest` from the
/// mutated running so a shallow "digest matches state" check passes. This
/// targets the HyperNova boundary where `U_i` must be bound as a full CE
/// claim, not merely through commitment data.
#[test]
fn verify_uncompressed_rejects_recorded_eval_a_tamper_even_after_redigest() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 81)]])
        .expect("one-batch uncompressed proof");
    let mut finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish");
    neo_fold_clean::verify_uncompressed(&prep, &finished).expect("honest proof verifies");

    let running = final_running_mut(&mut finished);
    assert!(
        !running.claims[0].eval_a.is_empty() && !running.claims[0].eval_a[0].is_empty(),
        "test setup requires Eval_A"
    );
    running.claims[0].eval_a[0][0] += neo_math::K::ONE;
    finished.state.acc_digest = recompute_active_running_acc_digest(&finished);

    let err = neo_fold_clean::verify_uncompressed(&prep, &finished)
        .expect_err("verify_uncompressed accepted a recorded Eval_A tamper after attacker re-digested state");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::PostStateMismatch
                | neo_fold_clean::Error::FinalAccumulatorCeRelationViolation { .. }
        ),
        "recorded Eval_A relabel must be caught by verifier-derived state binding or final CE authority, got {err:?}"
    );
}

/// Self-consistent final-running relabel attack on the CE evaluation point `r`.
///
/// This keeps the vector shape intact, changes only the point value, and then
/// re-digests the recorded running accumulator. If the verifier ever binds
/// only commitments, or treats the carried accumulator digest as authority,
/// this is the kind of mutation that can slip through while the state remains
/// locally self-consistent.
#[test]
fn verify_uncompressed_rejects_recorded_r_value_tamper_even_after_redigest() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 82)]])
        .expect("one-batch uncompressed proof");
    let mut finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish");
    neo_fold_clean::verify_uncompressed(&prep, &finished).expect("honest proof verifies");

    let running = final_running_mut(&mut finished);
    assert!(!running.claims[0].r.is_empty(), "test setup requires non-empty r");
    running.claims[0].r[0] += neo_math::K::ONE;
    finished.state.acc_digest = recompute_active_running_acc_digest(&finished);

    let err = neo_fold_clean::verify_uncompressed(&prep, &finished)
        .expect_err("verify_uncompressed accepted a recorded r-value tamper after attacker re-digested state");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::PostStateMismatch
                | neo_fold_clean::Error::FinalAccumulatorCeRelationViolation { .. }
        ),
        "recorded r relabel must be caught by verifier-derived state binding or final CE authority, got {err:?}"
    );
}

/// Direct terminal-authority relabel of the K-valued CE evaluation point.
///
/// This bypasses the normal proof entrypoint and mutates `claim.r[0]` in
/// place while keeping the vector length, commitment, X projection, witness,
/// and evaluation-family shapes unchanged. The witness is intentionally
/// non-zero so `M·Z(r)` depends on the point; a zero witness would make this
/// a vacuous test. Mutating only c1 catches verifiers that accidentally treat
/// the extension-field point as a base-field scalar.
#[test]
fn final_witness_authority_rejects_same_shape_r_c1_limb_relabel() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![toy_instance_with_x_value(&prep, neo_math::F::ONE)]])
        .expect("one-batch proof with non-zero terminal witness");
    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish");
    let mut running = final_running(&finished);

    assert!(
        running.witnesses[0]
            .to_dense_vec()
            .iter()
            .any(|&entry| entry != neo_math::F::ZERO),
        "test setup must carry a non-zero witness so r relabelling changes M·Z(r)"
    );
    assert!(!running.claims[0].r.is_empty(), "test setup requires non-empty r");
    let original = running.claims[0].r[0];
    running.claims[0].r[0] = original + neo_math::K::from_coeffs([neo_math::F::ZERO, neo_math::F::ONE]);
    assert_eq!(
        running.claims[0].r[0].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave the c0 limb unchanged"
    );
    assert_ne!(
        running.claims[0].r[0].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change only the c1 limb"
    );

    let err = neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &running)
        .expect_err("same-shape c1-only r relabel must violate the v1_1 evaluation relation");
    assert!(
        matches!(err, neo_fold_clean::Error::FinalAccumulatorCeRelationViolation { .. }),
        "expected FinalAccumulatorCeRelationViolation for same-shape r relabel, got {err:?}"
    );
}

/// Self-consistent final-accumulator substitution.
///
/// Hacker model: mutate the opened terminal witness `Z`, then recompute the
/// recorded CE claim from that new `Z` (`commit`, public projection,
/// `Eval_K`, and `Eval_A`) and re-digest the running accumulator. The final
/// accumulator is now locally valid under the SuperNeo CE relation, so a
/// verifier that only checks the terminal witness authority would accept it.
///
/// HyperNova still requires more: the terminal NIFS verifier must derive the
/// exact same post-fold running accumulator from the transcript. This test
/// proves `verify_uncompressed` rejects the locally-valid-but-wrong
/// accumulator rather than trusting the re-digested final state.
#[test]
fn verify_uncompressed_rejects_locally_valid_final_accumulator_substitution() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 83)]])
        .expect("one-batch uncompressed proof");
    let mut finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish");
    neo_fold_clean::verify_uncompressed(&prep, &finished).expect("honest proof verifies");

    let running = final_running_mut(&mut finished);
    let claim = running.claims.get_mut(0).expect("test setup: final claim");
    let witness = running
        .witnesses
        .get_mut(0)
        .expect("test setup: final witness");
    witness[(0, 0)] += neo_math::F::ONE;

    claim.c = prep.log.commit(witness);
    claim.X = project_x_from_witness_mat(witness, prep.structure().m, claim.m_in)
        .expect("mutated witness still has valid public projection shape");
    let ell_d = neo_math::D.next_power_of_two().trailing_zeros() as usize;
    let evaluations = compute_v1_1_evaluations_from_z_and_r(prep.structure(), witness, &claim.r, ell_d);
    claim.eval_k = evaluations.eval_k;
    claim.eval_a = evaluations.eval_a;
    finished.state.acc_digest = recompute_active_running_acc_digest(&finished);

    let err = neo_fold_clean::verify_uncompressed(&prep, &finished)
        .expect_err("verify_uncompressed accepted a locally valid final accumulator from the wrong terminal fold");
    assert!(
        matches!(err, neo_fold_clean::Error::PostStateMismatch),
        "locally valid final accumulator substitution must be rejected by terminal-fold state binding, got {err:?}"
    );
}

/// Terminal-latest relabel while the final accumulator still opens.
///
/// Hacker model: leave `proof.state.proof.running` and all terminal witnesses
/// untouched, but rewrite the public CCS claim that the terminal NIFS verifier
/// is supposed to fold. A verifier that checked only the final accumulator's
/// CE authority would accept, because that accumulator is still locally valid.
/// HyperNova Construction 2 requires more: terminal NIFS.V must replay from
/// the exact `(pre_final_running, latest)` snapshot.
#[test]
fn verify_uncompressed_rejects_terminal_latest_claim_relabel_even_though_final_accumulator_opens() {
    let prep = support::toy_preprocessing();
    let mut finished = one_batch_proof(&prep, neo_math::F::ONE);
    neo_fold_clean::verify_uncompressed(&prep, &finished).expect("honest proof verifies");
    final_running_passes_witness_authority(&prep, &finished);

    let final_fold = finished
        .final_fold
        .as_mut()
        .expect("one-batch proof must carry a terminal fold");
    let latest = final_fold
        .terminal_inputs
        .latest
        .instances
        .get_mut(0)
        .expect("terminal fold must carry a trailing latest claim");
    latest.claim.x[0] += neo_math::F::ONE;

    // The final accumulator is untouched and remains a locally valid CE
    // opening. Only terminal-fold replay should notice that the terminal
    // fold's public input snapshot no longer matches the proof transcript.
    final_running_passes_witness_authority(&prep, &finished);
    let err = neo_fold_clean::verify_uncompressed(&prep, &finished)
        .expect_err("verify_uncompressed accepted a relabelled terminal latest claim");
    assert!(
        !matches!(
            err,
            neo_fold_clean::Error::FinalAccumulatorWitnessCommitmentMismatch { .. }
                | neo_fold_clean::Error::FinalAccumulatorPublicInputMismatch { .. }
                | neo_fold_clean::Error::FinalAccumulatorCeRelationViolation { .. }
        ),
        "attack must be stopped by terminal NIFS replay, not by final CE authority; got {err:?}"
    );
}

/// Non-first terminal latest relabel in a multi-fresh terminal batch.
///
/// HyperNova's recursive link applies to every fresh instance in a chunk, not
/// just `fresh[0]`. This keeps the final accumulator untouched and mutates
/// only the second terminal latest public input; a verifier that accidentally
/// checks only the first terminal fresh claim would accept.
#[test]
fn verify_uncompressed_rejects_second_terminal_latest_claim_relabel_even_though_final_accumulator_opens() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(
        &prep,
        vec![vec![support::toy_instance(&prep, 21), support::toy_instance(&prep, 22)]],
    )
    .expect("one-batch proof with multi-fresh terminal latest");
    let mut finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish");
    neo_fold_clean::verify_uncompressed(&prep, &finished).expect("honest proof verifies");
    final_running_passes_witness_authority(&prep, &finished);

    let final_fold = finished
        .final_fold
        .as_mut()
        .expect("multi-fresh terminal batch must carry a terminal fold");
    let latest = final_fold
        .terminal_inputs
        .latest
        .instances
        .get_mut(1)
        .expect("terminal latest must carry a second fresh claim");
    latest.claim.x[0] += neo_math::F::ONE;

    final_running_passes_witness_authority(&prep, &finished);
    let err = neo_fold_clean::verify_uncompressed(&prep, &finished)
        .expect_err("verify_uncompressed accepted a relabelled non-first terminal latest claim");
    assert!(
        !matches!(
            err,
            neo_fold_clean::Error::FinalAccumulatorWitnessCommitmentMismatch { .. }
                | neo_fold_clean::Error::FinalAccumulatorPublicInputMismatch { .. }
                | neo_fold_clean::Error::FinalAccumulatorCeRelationViolation { .. }
        ),
        "attack must be stopped by terminal latest/NIFS replay, not by final CE authority; got {err:?}"
    );
}

/// Whole-terminal-proof replay from a different fold history.
///
/// This splices proof B's `FinalFoldProof` (including its terminal inputs and
/// transcript messages) onto proof A while keeping proof A's final accumulator
/// untouched and locally valid. It is stronger than mutating a field: every
/// piece of the terminal proof is internally consistent, just for the wrong
/// final state.
#[test]
fn verify_uncompressed_rejects_cross_history_terminal_fold_replay_while_final_accumulator_opens() {
    let prep = support::toy_preprocessing();
    let mut finished_a = one_batch_proof(&prep, neo_math::F::ZERO);
    let finished_b = one_batch_proof(&prep, neo_math::F::ONE);
    neo_fold_clean::verify_uncompressed(&prep, &finished_a).expect("honest proof A verifies");
    neo_fold_clean::verify_uncompressed(&prep, &finished_b).expect("honest proof B verifies");
    final_running_passes_witness_authority(&prep, &finished_a);

    finished_a.final_fold = finished_b.final_fold.clone();
    let forged_pre_final_acc_digest = running_acc_digest(
        &finished_a
            .final_fold
            .as_ref()
            .expect("spliced terminal fold is present")
            .terminal_inputs
            .pre_final_running,
    );
    // Avoid the stateless semantic fast-fail so this test reaches the
    // terminal-fold replay / post-state binding layer. A malicious proof
    // can relabel public state fields too; the final accumulator remains
    // locally valid either way.
    finished_a.state.semantic_state_digest = forged_pre_final_acc_digest;

    final_running_passes_witness_authority(&prep, &finished_a);
    let err = neo_fold_clean::verify_uncompressed(&prep, &finished_a)
        .expect_err("verify_uncompressed accepted a terminal fold proof replayed from a different history");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::PostStateMismatch | neo_fold_clean::Error::Construction2(_)
        ),
        "cross-history terminal proof replay must be rejected by terminal-fold replay/state binding, got {err:?}"
    );
}
