//! SHA-256 R1CS-F' batching experiments.
//!
//! This file keeps batching experiments separate from the main SHA
//! integration test so the statement shape is explicit. The experiment
//! here has two shapes:
//!
//! - ordered public batches: one app R1CS proves independent SHA
//!   preimages and exposes all digests as public inputs. This is not a
//!   hash-chain state machine.
//! - serial pairs: one app R1CS proves two SHA transitions by feeding
//!   the first digest directly into the second SHA gadget, then exposes
//!   `state_in` and `state_out` as public state bits. This is the
//!   paper-safe way to reduce F' append count for a serial SHA chain.

use std::time::{Duration, Instant};

use ::bellpepper::gadgets::boolean::{AllocatedBit, Boolean};
use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ccs::{CcsMatrix, CscMat};
use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::bellpepper::{synthesize_to_ccs, BellpepperCcs, BellpepperGoldilocks};
use neo_fold_clean::frontends::f_prime::image::{FPrimeImageLayout, NifsCeClaimShape, NifsPayloadShape};
use neo_fold_clean::frontends::f_prime::recursive_plan::{
    build_recursive_step_image_config, build_semantic_state_preimage_fields, AccumulatorPlanOptions,
    RecursiveStepImagePlan, StateXOutPlanOptions,
};
use neo_fold_clean::frontends::r1cs_f_prime::{self, R1csChainBuilder, SparseR1cs};
use neo_fold_clean::lifecycle;
use neo_fold_clean::paper::digest::digest_fields_as_digest32;
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_fold_clean::paper::params::Params;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use sha2::{Digest, Sha256};

const SHA256_AJTAI_SEED: u64 = 0x5348_4132_3556_5345;

#[derive(Clone, Debug)]
struct Sha256Circuit {
    preimage: Vec<u8>,
}

impl Circuit<BellpepperGoldilocks> for Sha256Circuit {
    fn synthesize<CS: ConstraintSystem<BellpepperGoldilocks>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let bit_values = ::bellpepper::gadgets::multipack::bytes_to_bits(&self.preimage)
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>();
        let preimage_bits = bit_values
            .into_iter()
            .enumerate()
            .map(|(idx, bit)| AllocatedBit::alloc(cs.namespace(|| format!("preimage_bit_{idx}")), bit))
            .map(|bit| bit.map(Boolean::from))
            .collect::<Result<Vec<_>, _>>()?;

        let hash_bits = ::bellpepper::gadgets::sha256::sha256(cs.namespace(|| "sha256"), &preimage_bits)?;
        for (bit_idx, bit) in hash_bits.iter().enumerate() {
            let value = bit
                .get_value()
                .ok_or(SynthesisError::AssignmentMissing)
                .map(|bit| {
                    if bit {
                        BellpepperGoldilocks::ONE
                    } else {
                        BellpepperGoldilocks::ZERO
                    }
                })?;
            let input = cs.alloc_input(|| format!("hash_out_bit_{bit_idx}"), || Ok(value))?;
            cs.enforce(
                || format!("hash_out_bit_match_{bit_idx}"),
                |_| bit.lc(CS::one(), BellpepperGoldilocks::ONE),
                |lc| lc + CS::one(),
                |lc| lc + input,
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Sha256SerialPackedStateCircuit {
    state_in: Vec<u8>,
    transitions: usize,
}

impl Circuit<BellpepperGoldilocks> for Sha256SerialPackedStateCircuit {
    fn synthesize<CS: ConstraintSystem<BellpepperGoldilocks>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        assert_eq!(self.state_in.len(), 32, "serial SHA state is a 32-byte digest");
        assert!(self.transitions > 0, "serial SHA circuit needs at least one transition");
        let state_in_bits = ::bellpepper::gadgets::multipack::bytes_to_bits(&self.state_in);
        let mut current = state_in_bits
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, bit)| {
                AllocatedBit::alloc(cs.namespace(|| format!("packed_state_in_private_bit_{idx}")), Some(bit))
            })
            .map(|bit| bit.map(Boolean::from))
            .collect::<Result<Vec<_>, _>>()?;

        enforce_public_lanes_from_bits(
            cs.namespace(|| "state_in_public_lanes"),
            "state_in_lane",
            &current,
            &state_lanes56_fields(&self.state_in),
        )?;

        for step in 0..self.transitions {
            current =
                ::bellpepper::gadgets::sha256::sha256(cs.namespace(|| format!("packed_sha256_step_{step}")), &current)?;
        }
        let state_out_bytes = sha_state_trace(&self.state_in, self.transitions)
            .pop()
            .expect("state trace includes final state");
        enforce_public_lanes_from_bits(
            cs.namespace(|| "state_out_public_lanes"),
            "state_out_lane",
            &current,
            &state_lanes56_fields(&state_out_bytes),
        )?;
        Ok(())
    }
}

#[test]
fn sha256_ordered_pair_r1cs_binds_both_public_digests() {
    let first = synthesize_to_ccs(Sha256Circuit {
        preimage: nth_preimage(0),
    })
    .expect("synthesize first SHA");
    let second = synthesize_to_ccs(Sha256Circuit {
        preimage: nth_preimage(1),
    })
    .expect("synthesize second SHA");
    let pair = combine_ordered_pair(&first, &second);

    assert_eq!(
        pair.assignment[..pair.r1cs.m_in],
        expected_pair_digest_bits(&nth_preimage(0), &nth_preimage(1)),
        "ordered-pair public input should be [1, digest_0_bits, digest_1_bits]"
    );
    pair.r1cs
        .is_satisfied_by(&pair.assignment)
        .expect("honest ordered-pair assignment satisfies");

    let mut tampered = pair.assignment.clone();
    tampered[1 + 256] = if tampered[1 + 256] == F::ZERO { F::ONE } else { F::ZERO };
    pair.r1cs
        .is_satisfied_by(&tampered)
        .expect_err("tampering the second digest's first public bit must reject");
}

#[test]
fn sha256_ordered_batch_r1cs_binds_middle_public_digest() {
    let artifacts: Vec<_> = (0..4)
        .map(|idx| {
            synthesize_to_ccs(Sha256Circuit {
                preimage: nth_preimage(idx),
            })
            .expect("synthesize SHA")
        })
        .collect();
    let batch = combine_ordered_batch(&artifacts);

    assert_eq!(
        batch.assignment[..batch.r1cs.m_in],
        expected_batch_digest_bits(&(0..4).map(nth_preimage).collect::<Vec<_>>()),
        "ordered batch public input should be [1, digest_0_bits, ..., digest_3_bits]"
    );
    batch
        .r1cs
        .is_satisfied_by(&batch.assignment)
        .expect("honest ordered-batch assignment satisfies");

    let mut tampered = batch.assignment.clone();
    let second_digest_first_bit = 1 + 256;
    tampered[second_digest_first_bit] = if tampered[second_digest_first_bit] == F::ZERO {
        F::ONE
    } else {
        F::ZERO
    };
    batch
        .r1cs
        .is_satisfied_by(&tampered)
        .expect_err("tampering a middle digest's public bit must reject");
}

#[test]
fn sha256_serial_pair_packed_state_r1cs_binds_state_in_and_state_out() {
    let state_in = initial_sha_state();
    let serial = synthesize_to_ccs(Sha256SerialPackedStateCircuit {
        state_in: state_in.clone(),
        transitions: 2,
    })
    .expect("synthesize packed-state serial SHA pair");

    assert_eq!(
        serial.assignment[..serial.sparse_r1cs.m_in],
        expected_serial_pair_state_lanes56(&state_in),
        "packed-state serial pair public input should be [1, state_in_56bit_lanes, state_out_56bit_lanes]"
    );
    serial
        .sparse_r1cs
        .is_satisfied_by(&serial.assignment)
        .expect("honest packed-state serial-pair assignment satisfies");

    let mut tampered_in = serial.assignment.clone();
    tampered_in[1] += F::ONE;
    serial
        .sparse_r1cs
        .is_satisfied_by(&tampered_in)
        .expect_err("tampering the packed public state_in limb must reject");

    let mut tampered_out = serial.assignment.clone();
    tampered_out[1 + STATE_LANES56] += F::ONE;
    serial
        .sparse_r1cs
        .is_satisfied_by(&tampered_out)
        .expect_err("tampering the packed public state_out limb must reject");
}

/// Width-policy analysis snapshot: histogram of inferred app-private var
/// widths for the production serial-quad SHA shape. `plan.limbs` (and with
/// it most of the F' image) is `Σ widths + 1`, so this is the measurement
/// study for any limb-policy change. Analysis only — no assertion beyond
/// shape sanity.
#[test]
#[ignore = "analysis snapshot; run manually with --ignored --nocapture"]
fn sha256_serial_quad_app_var_width_histogram_snapshot() {
    let states = sha_state_trace(&initial_sha_state(), 4);
    let serial = synthesize_to_ccs(Sha256SerialPackedStateCircuit {
        state_in: states[0].clone(),
        transitions: 4,
    })
    .expect("synthesize packed-state serial SHA quad");
    let shape = r1cs_f_prime::R1csShape::from(&serial.sparse_r1cs);
    let widths = shape.conservative_app_private_var_widths();
    assert_eq!(widths.len(), serial.sparse_r1cs.m, "one width per app var");

    let mut count_by_width = std::collections::BTreeMap::<usize, (usize, usize)>::new();
    for &w in &widths {
        let entry = count_by_width.entry(w).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += w;
    }
    let total_vars: usize = widths.len();
    let total_limbs: usize = widths.iter().sum();
    eprintln!("  app vars {total_vars}, total width limbs {total_limbs} (plan.limbs = total + 1)");
    eprintln!(
        "  {:>6} {:>10} {:>12} {:>8} {:>8}",
        "width", "vars", "limbs", "vars%", "limbs%"
    );
    for (w, (vars, limbs)) in &count_by_width {
        eprintln!(
            "  {:>6} {:>10} {:>12} {:>7.2}% {:>7.2}%",
            w,
            vars,
            limbs,
            *vars as f64 * 100.0 / total_vars as f64,
            *limbs as f64 * 100.0 / total_limbs as f64,
        );
    }

    // Dump the defining rows of the first few width-64 (inference-failed)
    // vars so the escape patterns are visible.
    let rows_of = |mat: &CcsMatrix<F>| -> std::collections::BTreeMap<usize, Vec<(usize, u64)>> {
        let mut map = std::collections::BTreeMap::<usize, Vec<(usize, u64)>>::new();
        if let CcsMatrix::Csc(csc) = mat {
            for col in 0..csc.ncols {
                for k in csc.col_ptr[col]..csc.col_ptr[col + 1] {
                    map.entry(csc.row_idx[k])
                        .or_default()
                        .push((col, csc.vals[k].as_canonical_u64()));
                }
            }
        }
        map
    };
    let (ra, rb, rc) = (
        rows_of(&serial.sparse_r1cs.a),
        rows_of(&serial.sparse_r1cs.b),
        rows_of(&serial.sparse_r1cs.c),
    );
    let signed = |v: u64| -> i128 {
        let p = F::ORDER_U64 as i128;
        if (v as i128) > p / 2 {
            v as i128 - p
        } else {
            v as i128
        }
    };
    let fmt_lc = |lc: Option<&Vec<(usize, u64)>>| -> String {
        lc.map(|terms| {
            terms
                .iter()
                .map(|&(var, coeff)| format!("{}*v{}", signed(coeff), var))
                .collect::<Vec<_>>()
                .join(" + ")
        })
        .unwrap_or_else(|| "0".into())
    };
    let mut dumped = 0;
    for (var, &w) in widths.iter().enumerate() {
        if w != POSEIDON2_GOLDILOCKS_BITS || dumped >= 2 || var % 2 == 1 {
            continue;
        }
        dumped += 1;
        eprintln!("  -- unproven var v{var}: rows mentioning it --");
        let mentions = |map: &std::collections::BTreeMap<usize, Vec<(usize, u64)>>| -> Vec<usize> {
            map.iter()
                .filter(|(_, terms)| terms.iter().any(|&(v, _)| v == var))
                .map(|(&row, _)| row)
                .collect()
        };
        let mut rows: Vec<usize> = mentions(&ra);
        rows.extend(mentions(&rb));
        rows.extend(mentions(&rc));
        eprintln!(
            "     (width of neighbours: v{} -> {}, v{} -> {})",
            var - 1,
            widths[var - 1],
            var + 1,
            widths[var + 1]
        );
        rows.sort_unstable();
        rows.dedup();
        for row in rows.into_iter().take(4) {
            eprintln!(
                "     row {row}: ({}) * ({}) = ({})",
                fmt_lc(ra.get(&row)),
                fmt_lc(rb.get(&row)),
                fmt_lc(rc.get(&row)),
            );
        }
    }
}

#[test]
fn sha256_serial_quad_packed_state_r1cs_binds_state_in_and_state_out() {
    let state_in = initial_sha_state();
    let serial = synthesize_to_ccs(Sha256SerialPackedStateCircuit {
        state_in: state_in.clone(),
        transitions: 4,
    })
    .expect("synthesize packed-state serial SHA quad");

    assert_eq!(
        serial.assignment[..serial.sparse_r1cs.m_in],
        expected_serial_state_lanes56(&state_in, 4),
        "packed-state serial quad public input should be [1, state_in_56bit_lanes, state_out_56bit_lanes]"
    );
    serial
        .sparse_r1cs
        .is_satisfied_by(&serial.assignment)
        .expect("honest packed-state serial-quad assignment satisfies");
}

#[test]
#[ignore = "production-core SHA batching perf snapshot; run manually to compare ordered pair batching"]
fn sha256_production_core_ordered_pair_batch_four_statements_perf_snapshot() {
    let synth_start = Instant::now();
    let artifacts: Vec<_> = (0..4)
        .map(|idx| {
            synthesize_to_ccs(Sha256Circuit {
                preimage: nth_preimage(idx),
            })
            .expect("synthesize SHA")
        })
        .collect();
    let synth = synth_start.elapsed();
    for artifact in artifacts.iter().skip(1) {
        assert_eq!(artifact.shape, artifacts[0].shape);
    }

    let single = time_single_sha_four_statements(&artifacts);
    let paired = time_ordered_pair_sha_four_statements(&artifacts);

    eprintln!();
    eprintln!("======================================================================");
    eprintln!("  SHA-256 ordered-pair batching perf snapshot");
    eprintln!("======================================================================");
    eprintln!();
    eprintln!("  Workload");
    eprintln!("    SHA statements                   4");
    eprintln!("    single F' chunks                 {}", single.folds);
    eprintln!("    ordered-pair F' chunks           {}", paired.folds);
    eprintln!("    Bellpepper synth (shared)     {:>8.3} ms", ms(synth));
    eprintln!();
    eprintln!("  Timing (ms)");
    eprintln!("    stage                 single chunks   ordered-pair chunks");
    eprintln!("    -------------------   -------------   -------------------");
    eprintln!(
        "    preprocess            {:>10.3}          {:>10.3}",
        ms(single.preprocess),
        ms(paired.preprocess)
    );
    eprintln!(
        "    append total          {:>10.3}          {:>10.3}",
        ms(single.append_total),
        ms(paired.append_total)
    );
    eprintln!(
        "    finish                {:>10.3}          {:>10.3}",
        ms(single.finish),
        ms(paired.finish)
    );
    eprintln!(
        "    verify                {:>10.3}          {:>10.3}",
        ms(single.verify),
        ms(paired.verify)
    );
    eprintln!(
        "    verify surface        {:>10}          {:>10}",
        single.verify_surface.as_str(),
        paired.verify_surface.as_str()
    );
    eprintln!(
        "    total                 {:>10.3}          {:>10.3}",
        ms(single.total),
        ms(paired.total)
    );
    eprintln!();
    eprintln!("  Throughput (SHA statements/s)");
    eprintln!("    single chunks         {:>10.2}", 4.0 / single.total.as_secs_f64());
    eprintln!("    ordered-pair chunks   {:>10.2}", 4.0 / paired.total.as_secs_f64());
    eprintln!("======================================================================");

    assert_eq!(single.final_public_digest_bits, expected_digest_bits(&nth_preimage(3)));
    assert_eq!(
        paired.final_public_digest_bits,
        expected_pair_digest_bits(&nth_preimage(2), &nth_preimage(3)),
        "ordered-pair final semantic state should expose the last pair's two SHA digests"
    );
}

#[test]
#[ignore = "production-core SHA batching curve; run manually to compare ordered batch sizes"]
fn sha256_production_core_ordered_batch_size_curve_four_statements_perf_snapshot() {
    let synth_start = Instant::now();
    let artifacts: Vec<_> = (0..4)
        .map(|idx| {
            synthesize_to_ccs(Sha256Circuit {
                preimage: nth_preimage(idx),
            })
            .expect("synthesize SHA")
        })
        .collect();
    let synth = synth_start.elapsed();

    let batch_1 = time_ordered_batch_sha_statements(&artifacts, 1);
    let batch_2 = time_ordered_batch_sha_statements(&artifacts, 2);
    let batch_4 = time_ordered_batch_sha_statements(&artifacts, 4);

    eprintln!();
    eprintln!("======================================================================");
    eprintln!("  SHA-256 ordered-batch size curve");
    eprintln!("======================================================================");
    eprintln!();
    eprintln!("  Workload");
    eprintln!("    SHA statements                   4");
    eprintln!("    Bellpepper synth (shared)     {:>8.3} ms", ms(synth));
    eprintln!();
    eprintln!("  Timing (ms)");
    eprintln!("    batch size        1 statement       2 statements      4 statements");
    eprintln!(
        "    chunks            {:>10}       {:>10}      {:>10}",
        batch_1.folds, batch_2.folds, batch_4.folds
    );
    eprintln!(
        "    preprocess        {:>10.3}       {:>10.3}      {:>10.3}",
        ms(batch_1.preprocess),
        ms(batch_2.preprocess),
        ms(batch_4.preprocess)
    );
    eprintln!(
        "    append total      {:>10.3}       {:>10.3}      {:>10.3}",
        ms(batch_1.append_total),
        ms(batch_2.append_total),
        ms(batch_4.append_total)
    );
    eprintln!(
        "    finish            {:>10.3}       {:>10.3}      {:>10.3}",
        ms(batch_1.finish),
        ms(batch_2.finish),
        ms(batch_4.finish)
    );
    eprintln!(
        "    verify            {:>10.3}       {:>10.3}      {:>10.3}",
        ms(batch_1.verify),
        ms(batch_2.verify),
        ms(batch_4.verify)
    );
    eprintln!(
        "    verify surface    {:>10}       {:>10}      {:>10}",
        batch_1.verify_surface.as_str(),
        batch_2.verify_surface.as_str(),
        batch_4.verify_surface.as_str()
    );
    eprintln!(
        "    total             {:>10.3}       {:>10.3}      {:>10.3}",
        ms(batch_1.total),
        ms(batch_2.total),
        ms(batch_4.total)
    );
    eprintln!();
    eprintln!("  Throughput (SHA statements/s)");
    eprintln!("    batch size 1      {:>10.2}", 4.0 / batch_1.total.as_secs_f64());
    eprintln!("    batch size 2      {:>10.2}", 4.0 / batch_2.total.as_secs_f64());
    eprintln!("    batch size 4      {:>10.2}", 4.0 / batch_4.total.as_secs_f64());
    eprintln!();
    eprintln!("  Shape");
    eprintln!("    batch size              1 statement       2 statements      4 statements");
    eprintln!(
        "    app constraints         {:>10}       {:>10}      {:>10}",
        batch_1.app_constraints, batch_2.app_constraints, batch_4.app_constraints
    );
    eprintln!(
        "    app vars                {:>10}       {:>10}      {:>10}",
        batch_1.app_vars, batch_2.app_vars, batch_4.app_vars
    );
    eprintln!(
        "    app public inputs       {:>10}       {:>10}      {:>10}",
        batch_1.app_public_inputs, batch_2.app_public_inputs, batch_4.app_public_inputs
    );
    eprintln!(
        "    plan limbs              {:>10}       {:>10}      {:>10}",
        batch_1.plan_limbs, batch_2.plan_limbs, batch_4.plan_limbs
    );
    eprintln!(
        "    F' rows n               {:>10}       {:>10}      {:>10}",
        batch_1.structure_n, batch_2.structure_n, batch_4.structure_n
    );
    eprintln!(
        "    F' vars m               {:>10}       {:>10}      {:>10}",
        batch_1.structure_m, batch_2.structure_m, batch_4.structure_m
    );
    eprintln!(
        "    F' matrices t           {:>10}       {:>10}      {:>10}",
        batch_1.structure_t, batch_2.structure_t, batch_4.structure_t
    );
    eprintln!("======================================================================");

    assert_eq!(batch_1.final_public_digest_bits, expected_digest_bits(&nth_preimage(3)));
    assert_eq!(
        batch_2.final_public_digest_bits,
        expected_pair_digest_bits(&nth_preimage(2), &nth_preimage(3))
    );
    assert_eq!(
        batch_4.final_public_digest_bits,
        expected_batch_digest_bits(&(0..4).map(nth_preimage).collect::<Vec<_>>())
    );
}

#[test]
#[ignore = "production-core packed-state serial SHA pair perf snapshot; run manually to compare sound two-transition chunks"]
fn sha256_production_core_serial_pair_packed_state_two_transitions_perf_snapshot() {
    let synth_start = Instant::now();
    let states = sha_state_trace(&initial_sha_state(), 2);
    let chunks = [synthesize_to_ccs(Sha256SerialPackedStateCircuit {
        state_in: states[0].clone(),
        transitions: 2,
    })
    .expect("synthesize packed serial SHA pair")];
    let synth = synth_start.elapsed();

    let snapshot = time_serial_sha_packed_state_transitions(&chunks, &states[0]);

    eprintln!();
    eprintln!("======================================================================");
    eprintln!("  SHA-256 packed-state serial-pair two-transition perf snapshot");
    eprintln!("======================================================================");
    eprintln!();
    eprintln!("  Workload");
    eprintln!("    SHA transitions                 2");
    eprintln!("    serial-pair F' chunks           {}", snapshot.folds);
    eprintln!(
        "    public state limbs/chunk        {} x 56-bit in + {} x 56-bit out",
        STATE_LANES56, STATE_LANES56
    );
    eprintln!("    Bellpepper synth (shared)     {:>8.3} ms", ms(synth));
    eprintln!();
    eprintln!("  Timing (ms)");
    eprintln!(
        "    setup: plan/structure         {:>10.3}",
        ms(snapshot.plan_structure)
    );
    eprintln!("    setup: structure builds       {:>10}", snapshot.plan_iterations);
    eprintln!("    setup: prepare cache          {:>10.3}", ms(snapshot.prepare_cache));
    eprintln!("    setup: preprocess             {:>10.3}", ms(snapshot.preprocess));
    eprintln!("    setup total                   {:>10.3}", ms(snapshot.setup_total()));
    eprintln!(
        "    prove with prepared key       {:>10.3}",
        ms(snapshot.prepared_total())
    );
    eprintln!("    append total                  {:>10.3}", ms(snapshot.append_total));
    print_append_times(&snapshot.append_times);
    eprintln!("    finish                        {:>10.3}", ms(snapshot.finish));
    eprintln!(
        "    verify ({})              {:>10.3}",
        snapshot.verify_surface.as_str(),
        ms(snapshot.verify)
    );
    eprintln!(
        "    online prove                  {:>10.3}",
        ms(snapshot.online_prove())
    );
    eprintln!(
        "    online prove+verify           {:>10.3}",
        ms(snapshot.online_total())
    );
    eprintln!("    setup+online total            {:>10.3}", ms(snapshot.total));
    eprintln!();
    eprintln!("  Shape");
    eprintln!("    app constraints               {:>10}", snapshot.app_constraints);
    eprintln!("    app vars                      {:>10}", snapshot.app_vars);
    eprintln!("    app public inputs             {:>10}", snapshot.app_public_inputs);
    eprintln!("    plan limbs                    {:>10}", snapshot.plan_limbs);
    eprintln!("    F' rows n                     {:>10}", snapshot.structure_n);
    eprintln!("    F' vars m                     {:>10}", snapshot.structure_m);
    eprintln!("    F' matrices t                 {:>10}", snapshot.structure_t);
    eprintln!("======================================================================");

    assert_eq!(
        snapshot.final_semantic_digest,
        serial_state_lanes56_semantic_digest(&states[2]),
        "packed serial-pair final semantic state should be SHA^2(initial)"
    );
}

#[test]
#[ignore = "production-core packed-state serial SHA pair perf snapshot; run manually to compare sound two-transition chunks"]
fn sha256_production_core_serial_pair_packed_state_four_transitions_perf_snapshot() {
    let synth_start = Instant::now();
    let states = sha_state_trace(&initial_sha_state(), 4);
    let chunks = [
        synthesize_to_ccs(Sha256SerialPackedStateCircuit {
            state_in: states[0].clone(),
            transitions: 2,
        })
        .expect("synthesize packed serial SHA pair 0"),
        synthesize_to_ccs(Sha256SerialPackedStateCircuit {
            state_in: states[2].clone(),
            transitions: 2,
        })
        .expect("synthesize packed serial SHA pair 1"),
    ];
    let synth = synth_start.elapsed();
    assert_eq!(chunks[0].shape, chunks[1].shape);

    let snapshot = time_serial_sha_packed_state_transitions(&chunks, &states[0]);

    eprintln!();
    eprintln!("======================================================================");
    eprintln!("  SHA-256 packed-state serial-pair perf snapshot");
    eprintln!("======================================================================");
    eprintln!();
    eprintln!("  Workload");
    eprintln!("    SHA transitions                 4");
    eprintln!("    serial-pair F' chunks           {}", snapshot.folds);
    eprintln!(
        "    public state limbs/chunk        {} x 56-bit in + {} x 56-bit out",
        STATE_LANES56, STATE_LANES56
    );
    eprintln!("    Bellpepper synth (shared)     {:>8.3} ms", ms(synth));
    eprintln!();
    eprintln!("  Timing (ms)");
    eprintln!(
        "    setup: plan/structure         {:>10.3}",
        ms(snapshot.plan_structure)
    );
    eprintln!("    setup: structure builds       {:>10}", snapshot.plan_iterations);
    eprintln!("    setup: prepare cache          {:>10.3}", ms(snapshot.prepare_cache));
    eprintln!("    setup: preprocess             {:>10.3}", ms(snapshot.preprocess));
    eprintln!("    setup total                   {:>10.3}", ms(snapshot.setup_total()));
    eprintln!(
        "    prove with prepared key       {:>10.3}",
        ms(snapshot.prepared_total())
    );
    eprintln!("    append total                  {:>10.3}", ms(snapshot.append_total));
    print_append_times(&snapshot.append_times);
    eprintln!("    finish                        {:>10.3}", ms(snapshot.finish));
    eprintln!(
        "    verify ({})              {:>10.3}",
        snapshot.verify_surface.as_str(),
        ms(snapshot.verify)
    );
    eprintln!(
        "    online prove                  {:>10.3}",
        ms(snapshot.online_prove())
    );
    eprintln!(
        "    online prove+verify           {:>10.3}",
        ms(snapshot.online_total())
    );
    eprintln!("    setup+online total            {:>10.3}", ms(snapshot.total));
    eprintln!();
    eprintln!("  Throughput");
    eprintln!(
        "    transitions/s                 {:>10.2}",
        4.0 / snapshot.total.as_secs_f64()
    );
    eprintln!(
        "    F' chunks/s                   {:>10.2}",
        snapshot.folds as f64 / snapshot.total.as_secs_f64()
    );
    eprintln!();
    eprintln!("  Shape");
    eprintln!("    app constraints               {:>10}", snapshot.app_constraints);
    eprintln!("    app vars                      {:>10}", snapshot.app_vars);
    eprintln!("    app public inputs             {:>10}", snapshot.app_public_inputs);
    eprintln!("    plan limbs                    {:>10}", snapshot.plan_limbs);
    eprintln!("    F' rows n                     {:>10}", snapshot.structure_n);
    eprintln!("    F' vars m                     {:>10}", snapshot.structure_m);
    eprintln!("    F' matrices t                 {:>10}", snapshot.structure_t);
    eprintln!("======================================================================");

    assert_eq!(
        snapshot.final_semantic_digest,
        serial_state_lanes56_semantic_digest(&states[4]),
        "packed serial-pair final semantic state should be SHA^4(initial)"
    );
}

#[test]
#[ignore = "production-core packed-state serial SHA quad perf snapshot; run manually to compare four transitions per F' append"]
fn sha256_production_core_serial_quad_packed_state_four_transitions_perf_snapshot() {
    let synth_start = Instant::now();
    let states = sha_state_trace(&initial_sha_state(), 4);
    let chunks = [synthesize_to_ccs(Sha256SerialPackedStateCircuit {
        state_in: states[0].clone(),
        transitions: 4,
    })
    .expect("synthesize packed serial SHA quad")];
    let synth = synth_start.elapsed();

    let snapshot = time_serial_sha_packed_state_transitions(&chunks, &states[0]);

    eprintln!();
    eprintln!("======================================================================");
    eprintln!("  SHA-256 packed-state serial-quad perf snapshot");
    eprintln!("======================================================================");
    eprintln!();
    eprintln!("  Workload");
    eprintln!("    SHA transitions                 4");
    eprintln!("    serial-quad F' chunks           {}", snapshot.folds);
    eprintln!(
        "    public state limbs/chunk        {} x 56-bit in + {} x 56-bit out",
        STATE_LANES56, STATE_LANES56
    );
    eprintln!("    Bellpepper synth (shared)     {:>8.3} ms", ms(synth));
    eprintln!();
    eprintln!("  Timing (ms)");
    eprintln!(
        "    setup: plan/structure         {:>10.3}",
        ms(snapshot.plan_structure)
    );
    eprintln!("    setup: structure builds       {:>10}", snapshot.plan_iterations);
    eprintln!("    setup: prepare cache          {:>10.3}", ms(snapshot.prepare_cache));
    eprintln!("    setup: preprocess             {:>10.3}", ms(snapshot.preprocess));
    eprintln!("    setup total                   {:>10.3}", ms(snapshot.setup_total()));
    eprintln!(
        "    prove with prepared key       {:>10.3}",
        ms(snapshot.prepared_total())
    );
    eprintln!("    append total                  {:>10.3}", ms(snapshot.append_total));
    print_append_times(&snapshot.append_times);
    eprintln!("    finish                        {:>10.3}", ms(snapshot.finish));
    eprintln!(
        "    verify ({})              {:>10.3}",
        snapshot.verify_surface.as_str(),
        ms(snapshot.verify)
    );
    eprintln!(
        "    online prove                  {:>10.3}",
        ms(snapshot.online_prove())
    );
    eprintln!(
        "    online prove+verify           {:>10.3}",
        ms(snapshot.online_total())
    );
    eprintln!("    setup+online total            {:>10.3}", ms(snapshot.total));
    eprintln!();
    eprintln!("  Throughput");
    eprintln!(
        "    transitions/s                 {:>10.2}",
        4.0 / snapshot.total.as_secs_f64()
    );
    eprintln!(
        "    F' chunks/s                   {:>10.2}",
        snapshot.folds as f64 / snapshot.total.as_secs_f64()
    );
    eprintln!();
    eprintln!("  Shape");
    eprintln!("    app constraints               {:>10}", snapshot.app_constraints);
    eprintln!("    app vars                      {:>10}", snapshot.app_vars);
    eprintln!("    app public inputs             {:>10}", snapshot.app_public_inputs);
    eprintln!("    plan limbs                    {:>10}", snapshot.plan_limbs);
    eprintln!("    F' rows n                     {:>10}", snapshot.structure_n);
    eprintln!("    F' vars m                     {:>10}", snapshot.structure_m);
    eprintln!("    F' matrices t                 {:>10}", snapshot.structure_t);
    eprintln!("======================================================================");

    assert_eq!(
        snapshot.final_semantic_digest,
        serial_state_lanes56_semantic_digest(&states[4]),
        "packed serial-quad final semantic state should be SHA^4(initial)"
    );
}

#[test]
#[ignore = "production-core prepared-key amortization snapshot; run manually to measure reusable verifier preprocessing for one anchored statement"]
fn sha256_production_core_serial_quad_prepared_key_amortization_snapshot() {
    const PROOFS: usize = 3;
    const TRANSITIONS: usize = 4;

    let base_state = initial_sha_state();
    // The R1CS-F' structure contains a verifier-owned base semantic-state
    // anchor. Reusing one prepared verifier key is therefore valid only for
    // the same anchored statement. Different initial SHA states require a
    // separately prepared structure, which is exactly what the soundness
    // checks should force.
    let initial_states = vec![base_state; PROOFS];

    let synth_start = Instant::now();
    let chunks = initial_states
        .iter()
        .map(|state| {
            synthesize_to_ccs(Sha256SerialPackedStateCircuit {
                state_in: state.clone(),
                transitions: TRANSITIONS,
            })
            .expect("synthesize packed serial SHA quad")
        })
        .collect::<Vec<_>>();
    let synth = synth_start.elapsed();
    for chunk in chunks.iter().skip(1) {
        assert_eq!(chunk.shape, chunks[0].shape);
        assert_eq!(chunk.sparse_r1cs.n, chunks[0].sparse_r1cs.n);
        assert_eq!(chunk.sparse_r1cs.m, chunks[0].sparse_r1cs.m);
        assert_eq!(chunk.sparse_r1cs.m_in, chunks[0].sparse_r1cs.m_in);
    }

    let setup_start = Instant::now();
    let (derived, plan_iterations) = sha256_packed_state_derived_structure_for_r1cs_with_params(
        &chunks[0].sparse_r1cs,
        &Params::production(),
        &initial_states[0],
    );
    let plan_structure = setup_start.elapsed();
    let plan_limbs = derived.plan().limbs;
    let structure_n = derived.structure().ccs.n;
    let structure_m = derived.structure().ccs.m;
    let structure_t = derived.structure().ccs.t();
    let params = Params::for_ccs_shape(structure_n, structure_t, derived.structure().ccs.max_degree())
        .expect("packed serial-quad production params");

    let start = Instant::now();
    let prepared = r1cs_f_prime::prepare_derived_structure(derived).expect("packed serial-quad prepare");
    let prepare_cache = start.elapsed();

    let start = Instant::now();
    let prep = r1cs_f_prime::preprocess_seeded_prepared_with_params(
        prepared,
        params,
        SHA256_AJTAI_SEED ^ 0x5154_4154_455f_0056,
    )
    .expect("packed serial-quad preprocess from prepared structure");
    let preprocess = start.elapsed();
    let setup_total = plan_structure + prepare_cache + preprocess;

    let mut online = Vec::with_capacity(PROOFS);
    for (state, chunk) in initial_states.iter().zip(chunks.iter()) {
        let snapshot = prove_serial_sha_packed_state_with_preprocessing(&prep, std::slice::from_ref(chunk), state);
        let expected_final = sha_state_trace(state, TRANSITIONS)
            .pop()
            .expect("state trace includes final state");
        assert_eq!(
            snapshot.final_semantic_digest,
            serial_state_lanes56_semantic_digest(&expected_final),
            "prepared-key proof final semantic state should be SHA^4(initial)"
        );
        online.push(snapshot);
    }

    let online_sum = online
        .iter()
        .fold(Duration::ZERO, |acc, snapshot| acc + snapshot.online_total());
    let prove_sum = online
        .iter()
        .fold(Duration::ZERO, |acc, snapshot| acc + snapshot.online_prove());
    let verify_sum = online
        .iter()
        .fold(Duration::ZERO, |acc, snapshot| acc + snapshot.verify);
    let amortized_setup_online = (setup_total + online_sum).as_secs_f64() / PROOFS as f64;

    eprintln!();
    eprintln!("======================================================================");
    eprintln!("  SHA-256 packed-state serial-quad prepared-key amortization snapshot");
    eprintln!("======================================================================");
    eprintln!();
    eprintln!("  Workload");
    eprintln!("    proofs with same anchor           {}", PROOFS);
    eprintln!("    SHA transitions/proof            {}", TRANSITIONS);
    eprintln!("    total SHA transitions            {}", PROOFS * TRANSITIONS);
    eprintln!("    serial-quad F' chunks/proof      1");
    eprintln!(
        "    public state limbs/chunk         {} x 56-bit in + {} x 56-bit out",
        STATE_LANES56, STATE_LANES56
    );
    eprintln!("    Bellpepper synth (all proofs) {:>10.3} ms", ms(synth));
    eprintln!();
    eprintln!("  One-time verifier-key setup (ms)");
    eprintln!("    setup: plan/structure         {:>10.3}", ms(plan_structure));
    eprintln!("    setup: structure builds       {:>10}", plan_iterations);
    eprintln!("    setup: prepare cache          {:>10.3}", ms(prepare_cache));
    eprintln!("    setup: preprocess             {:>10.3}", ms(preprocess));
    eprintln!("    setup total                   {:>10.3}", ms(setup_total));
    eprintln!();
    eprintln!("  Per-proof online timing (ms)");
    eprintln!("     idx    append     finish     verify     online   surface");
    for (idx, snapshot) in online.iter().enumerate() {
        eprintln!(
            "    {:>4}  {:>8.3}  {:>8.3}  {:>8.3}  {:>9.3}   {}",
            idx,
            ms(snapshot.append_total),
            ms(snapshot.finish),
            ms(snapshot.verify),
            ms(snapshot.online_total()),
            snapshot.verify_surface.as_str()
        );
    }
    eprintln!("  Totals / amortized");
    eprintln!("    online prove total           {:>10.3}", ms(prove_sum));
    eprintln!("    verify total                 {:>10.3}", ms(verify_sum));
    eprintln!("    online prove+verify total    {:>10.3}", ms(online_sum));
    eprintln!(
        "    amortized setup+online/proof {:>10.3}",
        amortized_setup_online * 1000.0
    );
    eprintln!();
    eprintln!("  Shape");
    eprintln!("    app constraints              {:>10}", chunks[0].sparse_r1cs.n);
    eprintln!("    app vars                     {:>10}", chunks[0].sparse_r1cs.m);
    eprintln!("    app public inputs            {:>10}", chunks[0].sparse_r1cs.m_in);
    eprintln!("    plan limbs                   {:>10}", plan_limbs);
    eprintln!("    F' rows n                    {:>10}", structure_n);
    eprintln!("    F' vars m                    {:>10}", structure_m);
    eprintln!("    F' matrices t                {:>10}", structure_t);
    eprintln!("======================================================================");
}

struct ShaBatchSnapshot {
    folds: usize,
    app_constraints: usize,
    app_vars: usize,
    app_public_inputs: usize,
    plan_limbs: usize,
    structure_n: usize,
    structure_m: usize,
    structure_t: usize,
    preprocess: Duration,
    append_total: Duration,
    finish: Duration,
    verify: Duration,
    verify_surface: VerifySurface,
    total: Duration,
    final_public_digest_bits: Vec<F>,
}

struct ShaSerialSnapshot {
    folds: usize,
    app_constraints: usize,
    app_vars: usize,
    app_public_inputs: usize,
    plan_limbs: usize,
    structure_n: usize,
    structure_m: usize,
    structure_t: usize,
    plan_structure: Duration,
    plan_iterations: usize,
    prepare_cache: Duration,
    preprocess: Duration,
    append_total: Duration,
    append_times: Vec<Duration>,
    finish: Duration,
    verify: Duration,
    verify_surface: VerifySurface,
    total: Duration,
    final_semantic_digest: [u8; 32],
}

impl ShaSerialSnapshot {
    fn setup_total(&self) -> Duration {
        self.plan_structure + self.prepare_cache + self.preprocess
    }

    fn prepared_total(&self) -> Duration {
        self.preprocess + self.online_total()
    }

    fn online_prove(&self) -> Duration {
        self.append_total + self.finish
    }

    fn online_total(&self) -> Duration {
        self.online_prove() + self.verify
    }
}

struct ShaSerialOnlineSnapshot {
    append_total: Duration,
    finish: Duration,
    verify: Duration,
    verify_surface: VerifySurface,
    final_semantic_digest: [u8; 32],
}

impl ShaSerialOnlineSnapshot {
    fn online_prove(&self) -> Duration {
        self.append_total + self.finish
    }

    fn online_total(&self) -> Duration {
        self.online_prove() + self.verify
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerifySurface {
    TerminalOnly,
    AuditReplay,
}

impl VerifySurface {
    fn as_str(self) -> &'static str {
        match self {
            VerifySurface::TerminalOnly => "terminal-only",
            VerifySurface::AuditReplay => "audit-replay",
        }
    }
}

fn verify_best_supported(
    prep: &r1cs_f_prime::R1csFPrimePreprocessing,
    audit: &neo_fold_clean::UncompressedAudit,
) -> VerifySurface {
    match lifecycle::verify_uncompressed(&prep.prep, &audit.proof) {
        Ok(()) => VerifySurface::TerminalOnly,
        Err(lifecycle::Error::FPrimeNonReplayUnsupported { .. }) => {
            lifecycle::verify_uncompressed_audit(&prep.prep, audit).expect("audit verifier accepts proof");
            VerifySurface::AuditReplay
        }
        Err(err) => panic!("verify_uncompressed rejected proof unexpectedly: {err:?}"),
    }
}

fn time_single_sha_four_statements(artifacts: &[BellpepperCcs]) -> ShaBatchSnapshot {
    time_ordered_batch_sha_statements(artifacts, 1)
}

fn time_ordered_pair_sha_four_statements(artifacts: &[BellpepperCcs]) -> ShaBatchSnapshot {
    time_ordered_batch_sha_statements(artifacts, 2)
}

fn time_ordered_batch_sha_statements(artifacts: &[BellpepperCcs], batch_size: usize) -> ShaBatchSnapshot {
    assert!(!artifacts.is_empty());
    assert!(batch_size > 0);
    assert_eq!(
        artifacts.len() % batch_size,
        0,
        "ordered-batch perf helper requires exact chunks"
    );
    let total_start = Instant::now();
    let batches: Vec<_> = artifacts
        .chunks(batch_size)
        .map(combine_ordered_batch)
        .collect();
    for batch in batches.iter().skip(1) {
        assert_eq!(batch.r1cs.n, batches[0].r1cs.n);
        assert_eq!(batch.r1cs.m, batches[0].r1cs.m);
        assert_eq!(batch.r1cs.m_in, batches[0].r1cs.m_in);
    }

    let (plan, structure_probe) = sha256_production_core_lifecycle_plan_for_r1cs(&batches[0].r1cs);
    let params = Params::for_ccs_shape(
        structure_probe.ccs.n,
        structure_probe.ccs.t(),
        structure_probe.ccs.max_degree(),
    )
    .expect("ordered-batch production params");

    let start = Instant::now();
    let prep = r1cs_f_prime::preprocess_sparse_seeded_with_params(&batches[0].r1cs, &plan, params, SHA256_AJTAI_SEED)
        .expect("ordered-batch preprocess");
    let preprocess = start.elapsed();

    let mut chain = R1csChainBuilder::new(&prep).expect("ordered-batch chain");
    let mut append_total = Duration::ZERO;
    for batch in &batches {
        let start = Instant::now();
        chain
            .append_assignment(batch.assignment.clone())
            .expect("ordered-batch append");
        append_total += start.elapsed();
    }

    let start = Instant::now();
    let audit = chain.finish_with_audit().expect("ordered-batch finish");
    let finish = start.elapsed();

    let start = Instant::now();
    let verify_surface = verify_best_supported(&prep, &audit);
    let verify = start.elapsed();

    let last_batch = batches.last().expect("non-empty batches");
    ShaBatchSnapshot {
        folds: batches.len(),
        app_constraints: batches[0].r1cs.n,
        app_vars: batches[0].r1cs.m,
        app_public_inputs: batches[0].r1cs.m_in,
        plan_limbs: plan.limbs,
        structure_n: structure_probe.ccs.n,
        structure_m: structure_probe.ccs.m,
        structure_t: structure_probe.ccs.t(),
        preprocess,
        append_total,
        finish,
        verify,
        verify_surface,
        total: total_start.elapsed(),
        final_public_digest_bits: last_batch.assignment[..last_batch.r1cs.m_in].to_vec(),
    }
}

fn time_serial_sha_packed_state_transitions(chunks: &[BellpepperCcs], initial_state: &[u8]) -> ShaSerialSnapshot {
    let total_start = Instant::now();
    for chunk in chunks.iter().skip(1) {
        assert_eq!(chunk.sparse_r1cs.n, chunks[0].sparse_r1cs.n);
        assert_eq!(chunk.sparse_r1cs.m, chunks[0].sparse_r1cs.m);
        assert_eq!(chunk.sparse_r1cs.m_in, chunks[0].sparse_r1cs.m_in);
    }

    let start = Instant::now();
    let (derived, plan_iterations) = sha256_packed_state_derived_structure_for_r1cs_with_params(
        &chunks[0].sparse_r1cs,
        &Params::production(),
        initial_state,
    );
    let plan_structure = start.elapsed();
    let plan_limbs = derived.plan().limbs;
    let structure_n = derived.structure().ccs.n;
    let structure_m = derived.structure().ccs.m;
    let structure_t = derived.structure().ccs.t();
    let params = Params::for_ccs_shape(structure_n, structure_t, derived.structure().ccs.max_degree())
        .expect("packed serial-pair production params");

    let start = Instant::now();
    let prepared = r1cs_f_prime::prepare_derived_structure(derived).expect("packed serial-pair prepare");
    let prepare_cache = start.elapsed();

    let start = Instant::now();
    let prep = r1cs_f_prime::preprocess_seeded_prepared_with_params(
        prepared,
        params,
        SHA256_AJTAI_SEED ^ 0x5154_4154_455f_0056,
    )
    .expect("packed serial-pair preprocess from prepared structure");
    let preprocess = start.elapsed();

    let mut chain = R1csChainBuilder::new(&prep).expect("packed serial-pair chain");
    let mut append_total = Duration::ZERO;
    let mut append_times = Vec::with_capacity(chunks.len());
    let mut final_semantic_digest = serial_state_lanes56_semantic_digest(initial_state);
    for chunk in chunks {
        let start = Instant::now();
        let compiled = chain
            .append_assignment(chunk.assignment.clone())
            .expect("packed serial-pair append");
        let append_time = start.elapsed();
        append_total += append_time;
        append_times.push(append_time);
        final_semantic_digest = digest_fields_as_digest32(compiled.semantic_state_digest_out);
    }

    let start = Instant::now();
    let audit = chain
        .finish_with_audit()
        .expect("packed serial-pair finish");
    let finish = start.elapsed();

    let start = Instant::now();
    let verify_surface = verify_best_supported(&prep, &audit);
    let verify = start.elapsed();

    ShaSerialSnapshot {
        folds: chunks.len(),
        app_constraints: chunks[0].sparse_r1cs.n,
        app_vars: chunks[0].sparse_r1cs.m,
        app_public_inputs: chunks[0].sparse_r1cs.m_in,
        plan_limbs,
        structure_n,
        structure_m,
        structure_t,
        plan_structure,
        plan_iterations,
        prepare_cache,
        preprocess,
        append_total,
        append_times,
        finish,
        verify,
        verify_surface,
        total: total_start.elapsed(),
        final_semantic_digest,
    }
}

fn prove_serial_sha_packed_state_with_preprocessing(
    prep: &r1cs_f_prime::R1csFPrimePreprocessing,
    chunks: &[BellpepperCcs],
    initial_state: &[u8],
) -> ShaSerialOnlineSnapshot {
    let mut chain = R1csChainBuilder::new(prep).expect("packed serial prepared-key chain");
    let mut append_total = Duration::ZERO;
    let mut final_semantic_digest = serial_state_lanes56_semantic_digest(initial_state);
    for chunk in chunks {
        let start = Instant::now();
        let compiled = chain
            .append_assignment(chunk.assignment.clone())
            .expect("packed serial prepared-key append");
        append_total += start.elapsed();
        final_semantic_digest = digest_fields_as_digest32(compiled.semantic_state_digest_out);
    }

    let start = Instant::now();
    let audit = chain
        .finish_with_audit()
        .expect("packed serial prepared-key finish");
    let finish = start.elapsed();

    let start = Instant::now();
    let verify_surface = verify_best_supported(prep, &audit);
    let verify = start.elapsed();

    ShaSerialOnlineSnapshot {
        append_total,
        finish,
        verify,
        verify_surface,
        final_semantic_digest,
    }
}

struct OrderedPairR1cs {
    r1cs: SparseR1cs,
    assignment: Vec<F>,
}

fn combine_ordered_pair(first: &BellpepperCcs, second: &BellpepperCcs) -> OrderedPairR1cs {
    combine_ordered_batch_refs(&[first, second])
}

fn combine_ordered_batch(artifacts: &[BellpepperCcs]) -> OrderedPairR1cs {
    let refs = artifacts.iter().collect::<Vec<_>>();
    combine_ordered_batch_refs(&refs)
}

fn combine_ordered_batch_refs(artifacts: &[&BellpepperCcs]) -> OrderedPairR1cs {
    assert!(!artifacts.is_empty());
    let first = artifacts[0];
    for artifact in artifacts.iter().skip(1) {
        assert_eq!(first.shape, artifact.shape);
    }
    let single_inputs = first.shape.inputs;
    let single_aux = first.shape.aux;
    let combined_inputs = 1 + artifacts.len() * (single_inputs - 1);
    let combined_aux = artifacts.len() * single_aux;
    let combined_m = combined_inputs + combined_aux;
    let combined_n = artifacts.len() * first.shape.constraints;

    let map_col = |artifact_idx: usize, col: usize| -> usize {
        if col == 0 {
            0
        } else if col < single_inputs {
            1 + artifact_idx * (single_inputs - 1) + (col - 1)
        } else {
            combined_inputs + artifact_idx * single_aux + (col - single_inputs)
        }
    };

    let a = combine_matrix_batch(
        artifacts,
        combined_n,
        combined_m,
        |artifact| &artifact.sparse_r1cs.a,
        &map_col,
    );
    let b = combine_matrix_batch(
        artifacts,
        combined_n,
        combined_m,
        |artifact| &artifact.sparse_r1cs.b,
        &map_col,
    );
    let c = combine_matrix_batch(
        artifacts,
        combined_n,
        combined_m,
        |artifact| &artifact.sparse_r1cs.c,
        &map_col,
    );
    let r1cs = SparseR1cs::new(a, b, c, combined_n, combined_m, combined_inputs).expect("combined pair R1CS shape");

    let mut assignment = Vec::with_capacity(combined_m);
    assignment.push(F::ONE);
    for artifact in artifacts {
        assignment.extend_from_slice(&artifact.assignment[1..single_inputs]);
    }
    for artifact in artifacts {
        assignment.extend_from_slice(&artifact.assignment[single_inputs..]);
    }
    assert_eq!(assignment.len(), combined_m);
    r1cs.is_satisfied_by(&assignment)
        .expect("combined ordered batch assignment satisfies R1CS");

    OrderedPairR1cs { r1cs, assignment }
}

fn combine_matrix_batch(
    artifacts: &[&BellpepperCcs],
    nrows: usize,
    ncols: usize,
    matrix: impl Fn(&BellpepperCcs) -> &CcsMatrix<F>,
    map_col: &impl Fn(usize, usize) -> usize,
) -> CcsMatrix<F> {
    let mut trips = Vec::new();
    let row_stride = artifacts[0].shape.constraints;
    for (artifact_idx, artifact) in artifacts.iter().enumerate() {
        push_remapped_trips(
            matrix(artifact),
            artifact_idx * row_stride,
            &|col| map_col(artifact_idx, col),
            &mut trips,
        );
    }
    CcsMatrix::Csc(CscMat::from_triplets(trips, nrows, ncols))
}

fn push_remapped_trips(
    matrix: &CcsMatrix<F>,
    row_offset: usize,
    map_col: &impl Fn(usize) -> usize,
    out: &mut Vec<(usize, usize, F)>,
) {
    match matrix {
        CcsMatrix::Identity { n } => {
            for idx in 0..*n {
                out.push((row_offset + idx, map_col(idx), F::ONE));
            }
        }
        CcsMatrix::Csc(csc) => {
            for col in 0..csc.ncols {
                for idx in csc.col_ptr[col]..csc.col_ptr[col + 1] {
                    out.push((row_offset + csc.row_idx[idx], map_col(col), csc.vals[idx]));
                }
            }
        }
    }
}

fn sha256_production_core_lifecycle_plan_for_r1cs(
    r1cs: &SparseR1cs,
) -> (
    RecursiveStepImagePlan,
    neo_fold_clean::frontends::f_prime::structure::FPrimeStructure,
) {
    sha256_lifecycle_plan_for_r1cs_with_params(r1cs, &Params::production())
}

fn sha256_lifecycle_plan_for_r1cs_with_params(
    r1cs: &SparseR1cs,
    params: &Params,
) -> (
    RecursiveStepImagePlan,
    neo_fold_clean::frontends::f_prime::structure::FPrimeStructure,
) {
    let shape = r1cs_f_prime::R1csShape::from(r1cs);
    let mut widths = shape.conservative_app_private_var_widths();
    for index in 0..shape.m_in() {
        widths[index] = 1;
    }
    let typed_bits: usize = widths.iter().sum();
    let c_data_entries = params.kappa() as usize * params.d() as usize;
    let child_count = params.k_rho() as u64;
    let mut r_len = challenge_len_for_domain(shape.n());
    let mut s_col_len = challenge_len_for_domain(typed_bits + 1);

    for _ in 0..8 {
        let mut plan =
            sha256_lifecycle_plan_with_ce_shape(shape.m(), shape.m_in(), c_data_entries, child_count, r_len, s_col_len);
        plan.limbs = typed_bits + 1;
        plan.app_private_var_widths = widths.clone();
        let layout = FPrimeImageLayout::new(build_recursive_step_image_config(&plan));
        let (structure, _) = r1cs_f_prime::build_r1cs_f_prime_structure(layout, &shape);
        let next_r_len = challenge_len_for_domain(structure.ccs.n);
        let next_s_col_len = challenge_len_for_domain(structure.ccs.m);
        if next_r_len == r_len && next_s_col_len == s_col_len {
            return (plan, structure);
        }
        r_len = next_r_len;
        s_col_len = next_s_col_len;
    }

    panic!("SHA ordered-pair R1CS-F' CE shape did not converge")
}

fn sha256_packed_state_derived_structure_for_r1cs_with_params(
    r1cs: &SparseR1cs,
    params: &Params,
    initial_state: &[u8],
) -> (r1cs_f_prime::R1csFPrimeDerivedStructure, usize) {
    let shape = r1cs_f_prime::R1csShape::from(r1cs);
    let widths = shape.conservative_app_private_var_widths();
    let typed_bits: usize = widths.iter().sum();
    let c_data_entries = params.kappa() as usize * params.d() as usize;
    let child_count = params.k_rho() as u64;
    let initial_anchor = serial_state_lanes56_semantic_digest(initial_state);
    let mut r_len = challenge_len_for_domain(shape.n());
    let mut s_col_len = challenge_len_for_domain(typed_bits + 1);

    for iteration in 1..=8 {
        let mut plan =
            sha256_lifecycle_plan_with_ce_shape(shape.m(), shape.m_in(), c_data_entries, child_count, r_len, s_col_len);
        let state_x_out = plan
            .state_x_out
            .as_mut()
            .expect("SHA lifecycle plan installs state_x_out");
        state_x_out.app_public_input_var_indices = (0..shape.m_in()).collect();
        state_x_out.app_public_input_bit_var_indices = Vec::new();
        state_x_out.semantic_state_in_var_indices = (1..=STATE_LANES56).collect();
        state_x_out.semantic_state_out_var_indices = ((1 + STATE_LANES56)..=(2 * STATE_LANES56)).collect();
        state_x_out.initial_semantic_state_digest_anchor = Some(initial_anchor);
        plan.limbs = typed_bits + 1;
        plan.app_private_var_widths = widths.clone();
        let derived = r1cs_f_prime::derive_sparse_preprocessing_structure(r1cs, &plan)
            .expect("derive packed-state SHA R1CS-F' structure");
        let next_r_len = challenge_len_for_domain(derived.structure().ccs.n);
        let next_s_col_len = challenge_len_for_domain(derived.structure().ccs.m);
        if next_r_len == r_len && next_s_col_len == s_col_len {
            return (derived, iteration);
        }
        r_len = next_r_len;
        s_col_len = next_s_col_len;
    }

    panic!("SHA packed-state serial R1CS-F' CE shape did not converge")
}

fn sha256_lifecycle_plan_with_ce_shape(
    m: usize,
    m_in: usize,
    c_data_entries: usize,
    child_count: u64,
    r_len: usize,
    s_col_len: usize,
) -> RecursiveStepImagePlan {
    let limbs = m * POSEIDON2_GOLDILOCKS_BITS + 1;
    let ce_shape = NifsCeClaimShape {
        c_data_entries,
        x_rows: 54,
        x_active_cols: 5,
        r_len,
        y_ring_inner_lens: vec![64; 8],
        y_zcol_len: 64,
        s_col_len,
    };
    let probe_plan = RecursiveStepImagePlan {
        limbs,
        app_private_var_widths: Vec::new(),
        boundary_bits: 4 * POSEIDON2_GOLDILOCKS_BITS,
        kmul_count: 0,
        ring_action_pair_count: 0,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        sponge_transcript_permutes: 0,
        nifs_payload_shapes: vec![NifsPayloadShape::CeClaim(ce_shape)],
        accumulator: Some(AccumulatorPlanOptions {
            ce_claim_payload_index: 0,
            c_data_entries,
            child_count,
            unified: true,
        }),
        state_x_out: None,
    };
    let probe_layout = FPrimeImageLayout::new(build_recursive_step_image_config(&probe_plan));
    let public_x_out_lane_bit_starts: [usize; 4] =
        std::array::from_fn(|i| probe_layout.boundary.offset + i * POSEIDON2_GOLDILOCKS_BITS);
    let mut plan = probe_plan;
    plan.state_x_out = Some(StateXOutPlanOptions {
        pc: 1,
        public_x_out_lane_bit_starts,
        app_public_input_var_indices: Vec::new(),
        app_public_input_bit_var_indices: (0..m_in).collect(),
        semantic_state_in_var_indices: Vec::new(),
        semantic_state_out_var_indices: Vec::new(),
        initial_semantic_state_digest_anchor: None,
    });
    plan
}

fn challenge_len_for_domain(size: usize) -> usize {
    size.next_power_of_two().max(2).trailing_zeros() as usize
}

fn nth_preimage(i: usize) -> Vec<u8> {
    vec![
        b'a' + (i % 26) as u8,
        b'a' + ((i / 26) % 26) as u8,
        b'a' + ((i / 676) % 26) as u8,
    ]
}

fn initial_sha_state() -> Vec<u8> {
    (0..32).map(|idx| idx as u8).collect()
}

fn sha_state_trace(initial: &[u8], transitions: usize) -> Vec<Vec<u8>> {
    let mut states = Vec::with_capacity(transitions + 1);
    states.push(initial.to_vec());
    for _ in 0..transitions {
        let next = Sha256::digest(states.last().expect("non-empty state trace")).to_vec();
        states.push(next);
    }
    states
}

fn expected_digest_bits(preimage: &[u8]) -> Vec<F> {
    let digest = Sha256::digest(preimage);
    let digest_bits = ::bellpepper::gadgets::multipack::bytes_to_bits(&digest);
    let mut out = Vec::with_capacity(1 + digest_bits.len());
    out.push(F::ONE);
    out.extend(
        digest_bits
            .into_iter()
            .map(|bit| if bit { F::ONE } else { F::ZERO }),
    );
    out
}

const STATE_LANES56: usize = 5;
const STATE_LIMB_BITS: usize = 56;

fn expected_serial_pair_state_lanes56(state_in: &[u8]) -> Vec<F> {
    expected_serial_state_lanes56(state_in, 2)
}

fn expected_serial_state_lanes56(state_in: &[u8], transitions: usize) -> Vec<F> {
    let states = sha_state_trace(state_in, transitions);
    let mut out = Vec::with_capacity(1 + 2 * STATE_LANES56);
    out.push(F::ONE);
    out.extend(state_lanes56_fields(&states[0]));
    out.extend(state_lanes56_fields(&states[transitions]));
    out
}

fn serial_state_lanes56_semantic_digest(state: &[u8]) -> [u8; 32] {
    digest_fields_as_digest32(
        encode_poseidon_trace(&build_semantic_state_preimage_fields(&state_lanes56_fields(state))).digest_native,
    )
}

fn state_lanes56_fields(state: &[u8]) -> Vec<F> {
    assert_eq!(state.len(), 32, "SHA state must be 32 bytes");
    ::bellpepper::gadgets::multipack::bytes_to_bits(state)
        .chunks(STATE_LIMB_BITS)
        .map(|chunk| {
            let mut value = 0u64;
            for (idx, bit) in chunk.iter().enumerate() {
                if *bit {
                    value |= 1u64 << idx;
                }
            }
            F::from_u64(value)
        })
        .collect()
}

fn enforce_public_lanes_from_bits<CS: ConstraintSystem<BellpepperGoldilocks>>(
    mut cs: CS,
    label: &str,
    bits: &[Boolean],
    lane_values: &[F],
) -> Result<(), SynthesisError> {
    assert_eq!(lane_values.len(), STATE_LANES56);
    for (lane_idx, value) in lane_values.iter().enumerate() {
        let input = cs.alloc_input(
            || format!("{label}_{lane_idx}"),
            || Ok(BellpepperGoldilocks::from(value.as_canonical_u64())),
        )?;
        let start = lane_idx * STATE_LIMB_BITS;
        let end = usize::min(start + STATE_LIMB_BITS, bits.len());
        let lane_bits = &bits[start..end];
        cs.enforce(
            || format!("{label}_{lane_idx}_matches_bits"),
            |lc| {
                let mut out = lc;
                let mut coeff = BellpepperGoldilocks::ONE;
                for bit in lane_bits {
                    out = out + &bit.lc(CS::one(), coeff);
                    coeff += coeff;
                }
                out
            },
            |lc| lc + CS::one(),
            |lc| lc + input,
        );
    }
    Ok(())
}

fn expected_pair_digest_bits(first: &[u8], second: &[u8]) -> Vec<F> {
    expected_batch_digest_bits(&[first.to_vec(), second.to_vec()])
}

fn expected_batch_digest_bits(preimages: &[Vec<u8>]) -> Vec<F> {
    let mut out = Vec::with_capacity(1 + preimages.len() * 256);
    out.push(F::ONE);
    for preimage in preimages {
        let digest_bits = expected_digest_bits(preimage);
        out.extend_from_slice(&digest_bits[1..]);
    }
    out
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn print_append_times(append_times: &[Duration]) {
    for (idx, append_time) in append_times.iter().enumerate() {
        eprintln!("      append[{idx}]                 {:>10.3}", ms(*append_time));
    }
}
