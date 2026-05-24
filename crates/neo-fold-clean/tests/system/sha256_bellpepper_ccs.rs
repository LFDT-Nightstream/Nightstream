//! Bellpepper SHA-256 → R1CS-F' → real IVC recursion via `R1csChainBuilder`.
//!
//! One broad test pins the load-bearing IVC contract: two distinct
//! SHA-256 proofs (different preimages, same R1CS shape) are chained
//! through R1CS-F' so step 1 embeds and verifies the NIFS fold authority
//! produced by step 0. Mirrors the Fibonacci recursive test
//! ([`fibonacci_chain_builder_appends_recursive_step_under_tiny_params`])
//! in shape and assertion depth; replaces the Fibonacci app-step
//! transition with one Bellpepper-synthesized SHA-256 per step.
//!
//! Runs under a test-only smaller `Params` profile (`kappa = 2,
//! m = 2^15, lambda = 40`) so prove + extend fit under the 5-minute
//! test cap. The Goldilocks ring + Π_RLC / Π_DEC algebraic identities
//! are unchanged; only the Ajtai-SIS security parameter is reduced,
//! which is the right knob for an algebraic-correctness fixture.
//!
//! Why R1CS-F', not direct-CCS: this is real IVC recursion — each
//! recursive step's encoded image embeds and verifies the prior NIFS
//! fold authority, threads chain state (`z_i`, `acc_digest`,
//! `public_trace`, counters), and commits to the per-step SHA digest
//! through `state_x_out`. Direct-CCS aggregates K independent CCS
//! claims into one accumulator without per-step state advance or
//! cross-step verification — a different protocol contract.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use ::bellpepper::gadgets::boolean::{AllocatedBit, Boolean};
use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ccs::Mat;
use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::bellpepper::{synthesize_to_ccs, BellpepperGoldilocks};
use neo_fold_clean::frontends::direct_ccs::ajtai as direct_ajtai;
use neo_fold_clean::frontends::f_prime::image::{FPrimeImageLayout, NifsCeClaimShape, NifsPayloadShape};
use neo_fold_clean::frontends::f_prime::recursive_plan::{
    build_recursive_step_image_config, AccumulatorPlanOptions, RecursiveStepImagePlan, StateXOutPlanOptions,
};
use neo_fold_clean::frontends::r1cs_f_prime::{self, R1csChainBuilder};
use neo_fold_clean::lifecycle;
use neo_fold_clean::paper::construction2::{ProofState, FINAL_FOLD_TRANSCRIPT_LABEL};
use neo_fold_clean::paper::digest::structure_digest;
use neo_fold_clean::paper::digest::{
    accumulator_digest_from_parent_claim, digest32_as_fields, pi_ccs_instance_digest_parent_authority,
};
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::relations::{CcsClaim, CcsWitness, CeClaim, Structure};
use neo_math::F;
use neo_params::{goldilocks_paper_b2, NeoParams};
use neo_reductions::optimized_engine::{
    optimized_replay_trace_with_cache_instance_digest_and_me_input_handle_and_perf, OptimizedStructureCache,
};
use neo_transcript::{Poseidon2Transcript, Transcript as _};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// One phase of the IVC test, timed and logged via `--nocapture`.
fn phase<R>(label: &str, f: impl FnOnce() -> R) -> R {
    let t = Instant::now();
    let out = f();
    eprintln!("[sha-ivc] {label:<32} {:>7.2}s", t.elapsed().as_secs_f64());
    out
}

const SHA256_AJTAI_SEED: u64 = 0x5348_4132_3556_5345;
const TERMINAL_REPLAY_FIXTURE_VERSION: u32 = 1;

/// Local perf fixture for replaying the terminal Π_CCS prover input.
///
/// This is a developer cache, not protocol authority. The file is written
/// under `target/perf-fixtures`, validated on load by recomputing the
/// structure digest and shape, and regenerated if stale.
#[derive(Serialize, Deserialize)]
struct TerminalOptimizedProveFixture {
    header: TerminalOptimizedProveFixtureHeader,
    params: NeoParams,
    structure: Structure,
    mcs_list: Vec<CcsClaim>,
    mcs_witnesses: Vec<CcsWitness>,
    me_inputs: Vec<CeClaim>,
    me_witnesses: Vec<Mat<F>>,
    public_instance_digest: [F; 4],
    me_input_accumulator_handle: [F; 4],
}

#[derive(Serialize, Deserialize)]
struct TerminalOptimizedProveFixtureHeader {
    version: u32,
    ajtai_seed: u64,
    structure_n: usize,
    structure_m: usize,
    structure_t: usize,
    structure_digest: [F; 4],
    params_kappa: u32,
    params_d: u32,
    params_m: u64,
    params_lambda: u32,
    mcs_count: usize,
    me_count: usize,
}

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

fn expected_sha256_public_inputs(preimage: &[u8]) -> Vec<F> {
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

/// Test-only `Params` profile. Reuses the production Goldilocks ring
/// (Q, ETA, D, B_BASE, K_RHO, T, EXTENSION_DEGREE); only `kappa`, `m`,
/// `lambda` are shrunk so the lifecycle fits under the 5-minute cap.
/// Every Π_RLC / Π_DEC algebraic identity holds bit-for-bit at this
/// profile — only the Ajtai-SIS security parameter is reduced.
fn sha256_tiny_neo_params() -> NeoParams {
    NeoParams::new(
        goldilocks_paper_b2::Q,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::D as u32,
        /* kappa  */ 2,
        /* m      */ 1u64 << 15,
        goldilocks_paper_b2::B_BASE,
        goldilocks_paper_b2::K_RHO,
        goldilocks_paper_b2::T,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        /* lambda */ 40,
    )
    .expect("tiny NeoParams must satisfy the Π_RLC guard")
}

fn sha256_tiny_params() -> Params {
    Params::test_only_from_neo_params(sha256_tiny_neo_params())
}

/// Plan with the empirically-discovered fixed-point CE shape under
/// [`sha256_tiny_params`] for the SHA-256 R1CS shape.
/// `c_data_entries = KAPPA * D = 108` and `child_count = K_RHO = 14`
/// are params-derived. `r_len = s_col_len = 23 = ceil(log2(structure.m))`
/// — SHA-256's ~24.6K Bellpepper variables push the limbs region above
/// the smaller `r1cs_compiler` test's, so r_len here is one round
/// larger than that test's `r_len = 21`. If any of these shifts, the
/// recursive compile fails with `PostParentShapeMismatch` and the
/// error message names the new shape — update the constants from that.
fn sha256_tiny_lifecycle_plan(m: usize, m_in: usize) -> RecursiveStepImagePlan {
    const TINY_C_DATA_ENTRIES: usize = 108;
    const TINY_CHILD_COUNT: u64 = 14;
    const TINY_R_LEN: usize = 23;

    let limbs = m * POSEIDON2_GOLDILOCKS_BITS + 1;
    let ce_shape = NifsCeClaimShape {
        c_data_entries: TINY_C_DATA_ENTRIES,
        x_rows: 54,
        x_active_cols: 5,
        r_len: TINY_R_LEN,
        y_ring_inner_lens: vec![64; 8],
        y_zcol_len: 64,
        s_col_len: TINY_R_LEN,
    };
    let probe_plan = RecursiveStepImagePlan {
        limbs,
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
            c_data_entries: TINY_C_DATA_ENTRIES,
            child_count: TINY_CHILD_COUNT,
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
        app_public_input_var_indices: (0..m_in).collect(),
        semantic_state_in_var_indices: Vec::new(),
        semantic_state_out_var_indices: Vec::new(),
    });
    plan
}

impl TerminalOptimizedProveFixture {
    fn validate(&self) -> Result<(), String> {
        if self.header.version != TERMINAL_REPLAY_FIXTURE_VERSION {
            return Err(format!(
                "fixture version {} != {}",
                self.header.version, TERMINAL_REPLAY_FIXTURE_VERSION
            ));
        }
        let digest = structure_digest(&self.structure);
        if digest != self.header.structure_digest {
            return Err("structure digest mismatch".into());
        }
        let shape = (self.structure.n, self.structure.m, self.structure.t());
        let expected_shape = (
            self.header.structure_n,
            self.header.structure_m,
            self.header.structure_t,
        );
        if shape != expected_shape {
            return Err(format!(
                "structure shape {shape:?} != fixture header {expected_shape:?}"
            ));
        }
        if self.params.kappa != self.header.params_kappa
            || self.params.d != self.header.params_d
            || self.params.m != self.header.params_m
            || self.params.lambda != self.header.params_lambda
        {
            return Err("params fingerprint mismatch".into());
        }
        if self.mcs_list.len() != self.header.mcs_count || self.mcs_witnesses.len() != self.header.mcs_count {
            return Err("MCS count mismatch".into());
        }
        if self.me_inputs.len() != self.header.me_count || self.me_witnesses.len() != self.header.me_count {
            return Err("ME count mismatch".into());
        }
        Ok(())
    }
}

fn terminal_replay_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/perf-fixtures/neo-fold-clean/sha256-r1cs-fprime-terminal-optimized-prove-v1.bincode")
}

fn load_or_build_terminal_replay_fixture() -> TerminalOptimizedProveFixture {
    let path = terminal_replay_fixture_path();
    if let Ok(bytes) = fs::read(&path) {
        match bincode::deserialize::<TerminalOptimizedProveFixture>(&bytes) {
            Ok(fixture) => match fixture.validate() {
                Ok(()) => {
                    eprintln!("[sha-ivc] loaded replay fixture: {}", path.display());
                    return fixture;
                }
                Err(err) => eprintln!("[sha-ivc] stale replay fixture ({}): {err}", path.display()),
            },
            Err(err) => eprintln!("[sha-ivc] unreadable replay fixture ({}): {err}", path.display()),
        }
    }

    let fixture = build_terminal_replay_fixture();
    fixture.validate().expect("fresh replay fixture validates");
    let bytes = bincode::serialize(&fixture).expect("serialize terminal replay fixture");
    fs::create_dir_all(path.parent().expect("fixture path has parent")).expect("create fixture directory");
    fs::write(&path, bytes).expect("write terminal replay fixture");
    eprintln!("[sha-ivc] wrote replay fixture: {}", path.display());
    fixture
}

fn build_terminal_replay_fixture() -> TerminalOptimizedProveFixture {
    let artifact_a = phase("fixture synth A", || {
        synthesize_to_ccs(Sha256Circuit {
            preimage: b"abc".to_vec(),
        })
        .expect("synthesize SHA-256(abc)")
    });
    let m = artifact_a.shape.inputs + artifact_a.shape.aux;
    let plan = sha256_tiny_lifecycle_plan(m, artifact_a.shape.inputs);
    let params = sha256_tiny_neo_params();
    let prep = phase("fixture preprocess", || {
        r1cs_f_prime::preprocess_sparse_seeded_with_params(
            &artifact_a.sparse_r1cs,
            &plan,
            Params::test_only_from_neo_params(params.clone()),
            SHA256_AJTAI_SEED,
        )
        .expect("SHA-256 R1CS-F' tiny-params preprocess")
    });

    let mut chain = R1csChainBuilder::new(&prep).expect("start chain");
    let compiled_a = phase("fixture step 0 append", || {
        chain
            .append_assignment(artifact_a.assignment.clone())
            .expect("base step appends")
    });
    let artifact_b = phase("fixture synth B", || {
        synthesize_to_ccs(Sha256Circuit {
            preimage: b"xyz".to_vec(),
        })
        .expect("synthesize SHA-256(xyz)")
    });
    let compiled_b = phase("fixture step 1 append", || {
        chain
            .append_assignment(artifact_b.assignment.clone())
            .expect("recursive step appends")
    });
    drop(compiled_a);
    drop(compiled_b);

    let audit = chain.audit().expect("audit after recursive append");
    let ProofState::Active { running, latest } = &audit.proof.state.proof else {
        panic!("two-step SHA chain must be active before finalization");
    };
    let parent = running
        .parent_authority
        .as_ref()
        .expect("non-empty running accumulator carries parent authority");
    let mcs_list: Vec<CcsClaim> = latest
        .instances
        .iter()
        .map(|inst| inst.claim.clone())
        .collect();
    let mcs_witnesses: Vec<CcsWitness> = latest
        .instances
        .iter()
        .map(|inst| inst.witness.clone())
        .collect();
    let public_instance_digest = pi_ccs_instance_digest_parent_authority(&mcs_list, running.claims.len(), Some(parent));
    let me_input_accumulator_handle =
        digest32_as_fields(accumulator_digest_from_parent_claim(running.claims.len(), parent));
    let structure = prep.prep.structure().clone();
    let structure_digest = structure_digest(&structure);
    TerminalOptimizedProveFixture {
        header: TerminalOptimizedProveFixtureHeader {
            version: TERMINAL_REPLAY_FIXTURE_VERSION,
            ajtai_seed: SHA256_AJTAI_SEED,
            structure_n: structure.n,
            structure_m: structure.m,
            structure_t: structure.t(),
            structure_digest,
            params_kappa: params.kappa,
            params_d: params.d,
            params_m: params.m,
            params_lambda: params.lambda,
            mcs_count: mcs_list.len(),
            me_count: running.claims.len(),
        },
        params,
        structure,
        mcs_list,
        mcs_witnesses,
        me_inputs: running.claims.clone(),
        me_witnesses: running.witnesses.clone(),
        public_instance_digest,
        me_input_accumulator_handle,
    }
}

#[test]
#[ignore = "perf-only replay fixture; run manually to profile terminal optimized_prove without rebuilding the full SHA chain"]
fn sha256_terminal_optimized_prove_replay_perf() {
    let fixture = load_or_build_terminal_replay_fixture();

    let params = Params::test_only_from_neo_params(fixture.params.clone());
    let log = direct_ajtai::setup_seeded(&params, &fixture.structure, fixture.header.ajtai_seed);
    let cache = phase("replay optimized cache build", || {
        OptimizedStructureCache::build(&fixture.structure).expect("build optimized replay cache")
    });
    let mut tr = Poseidon2Transcript::new(FINAL_FOLD_TRANSCRIPT_LABEL);
    let (_terminal, _proof) = phase("replay terminal optimized_prove", || {
        optimized_replay_trace_with_cache_instance_digest_and_me_input_handle_and_perf(
            &mut tr,
            &fixture.params,
            &fixture.structure,
            &fixture.mcs_list,
            &fixture.mcs_witnesses,
            &fixture.me_inputs,
            &fixture.me_witnesses,
            fixture.public_instance_digest,
            fixture.me_input_accumulator_handle,
            &log,
            &cache,
        )
        .expect("terminal optimized replay accepts fixture")
    });
}

#[test]
fn sha256_bellpepper_ivc_chain_two_steps() {
    let total = Instant::now();

    // 1. Synthesize the first SHA-256 ("abc"). Cross-check the gadget
    //    produced the real digest before sinking it into the chain.
    let artifact_a = phase("synth A (Bellpepper SHA)", || {
        synthesize_to_ccs(Sha256Circuit {
            preimage: b"abc".to_vec(),
        })
        .expect("synthesize SHA-256(abc)")
    });
    eprintln!(
        "[sha-ivc]   shape: constraints={}, inputs={}, aux={}",
        artifact_a.shape.constraints, artifact_a.shape.inputs, artifact_a.shape.aux,
    );
    assert_eq!(
        artifact_a.public_inputs(),
        expected_sha256_public_inputs(b"abc"),
        "Bellpepper SHA-256(abc) must expose the real digest bits"
    );

    // 2. Preprocess R1CS-F' under tiny params. The verifier pins
    //    artifact_a's sparse-CCS shape as the per-step `F'_j`; both
    //    chain steps must satisfy this same shape.
    //
    //    Builds: the F' structure (~1.57M bitness rows + ~25K R1CS
    //    recompose rows), the optimized engine cache (sparse + SuperNeo
    //    eval tables + matrix digest), structure_digest, vk, and the
    //    Ajtai PP. Single largest one-time cost in the test.
    let m = artifact_a.shape.inputs + artifact_a.shape.aux;
    let plan = sha256_tiny_lifecycle_plan(m, artifact_a.shape.inputs);
    let prep = phase("preprocess R1CS-F'", || {
        r1cs_f_prime::preprocess_sparse_seeded_with_params(
            &artifact_a.sparse_r1cs,
            &plan,
            sha256_tiny_params(),
            SHA256_AJTAI_SEED,
        )
        .expect("SHA-256 R1CS-F' tiny-params preprocess")
    });
    eprintln!(
        "[sha-ivc]   structure.n={}, structure.m={}, plan.limbs={}",
        prep.prep.structure().n,
        prep.prep.structure().m,
        plan.limbs,
    );

    // 3. Step 0 (base): SHA-256("abc"). `R1csChainBuilder` owns
    //    compile → lifecycle::prove for this step. Internally:
    //      - compile_step reuses the cached `prep.structure` Arc
    //        (built once at preprocess time), runs R1CS satisfaction,
    //        encodes the image, runs a satisfaction self-check against
    //        the cached structure.
    //      - lifecycle::prove runs Π_CCS sumcheck + Π_RLC + Π_DEC.
    let mut chain = R1csChainBuilder::new(&prep).expect("start chain");
    let compiled_a = phase("step 0 append (base)", || {
        chain
            .append_assignment(artifact_a.assignment.clone())
            .expect("base step (SHA-256 abc) appends")
    });
    assert!(
        compiled_a.encoded.image.decode_is_base(),
        "step 0 must take the base branch"
    );

    // 4. Second SHA-256 with a different preimage of the same byte
    //    length. Same R1CS shape (matrices are preimage-independent),
    //    different digest.
    let artifact_b = phase("synth B (Bellpepper SHA)", || {
        synthesize_to_ccs(Sha256Circuit {
            preimage: b"xyz".to_vec(),
        })
        .expect("synthesize SHA-256(xyz)")
    });
    assert_eq!(
        artifact_b.shape, artifact_a.shape,
        "same-length preimages must produce the same R1CS shape"
    );
    assert_ne!(
        artifact_b.public_inputs(),
        artifact_a.public_inputs(),
        "different preimages must produce different SHA-256 digests"
    );

    // 5. Step 1 (recursive): SHA-256("xyz"). Heaviest single phase —
    //    inside `append_assignment` the builder does THREE things:
    //      a. `prepare_next_fold`: extend a CLONED audit with step 0's
    //         latest to derive the NIFS proof + post_running that the
    //         recursive compile needs. This is a full Π_CCS+RLC+DEC
    //         prover pass on the cloned audit. The post-fold audit is
    //         **stashed** and reused at deposit time (see (c)).
    //      b. `compile_step`: reuse the cached `prep.structure` Arc,
    //         run NIFS.V on the prior fold under the per-step F'
    //         transcript, encode the image with the prior fold
    //         authority embedded in the NIFS payload, satisfaction
    //         self-check against the cached structure.
    //      c. deposit: swap the freshly compiled recursive instance into
    //         the stashed post-fold audit's `latest` — **no** second
    //         prover pass (the fold from (a) is reused; the chunk digest
    //         is shape+count-only so it is identical for the real
    //         instance). So step 1's wall time ≈ 1 × (prove cost) +
    //         1 × (compile + NIFS.V), not 2× prove.
    let compiled_b = phase("step 1 append (recursive)", || {
        chain
            .append_assignment(artifact_b.assignment.clone())
            .expect("recursive step (SHA-256 xyz) appends")
    });
    assert!(
        !compiled_b.encoded.image.decode_is_base(),
        "step 1 must take the recursive branch"
    );

    // 6. HyperNova fixed-`F'_j` invariant: base and recursive R1CS-F'
    //    compiles share one verifier-owned structure (`prep.plan`),
    //    so their encoded steps share one `structure_digest`. This is
    //    the load-bearing IVC property.
    assert!(
        std::sync::Arc::ptr_eq(&compiled_a.encoded.structure, &compiled_b.encoded.structure),
        "base and recursive SHA-256 R1CS-F' compiles must share one verifier-owned structure \
         (HyperNova \u{00A7}6.3 Construction 2 fixed-`F'_j` invariant)"
    );

    // 7. Each step's `state_x_out` absorbs its app public input (the
    //    257 SHA digest bits via `app_public_input_var_indices`), so
    //    different preimages yield different `public_output_digest`s.
    //    The verifier-visible chain output really commits to which SHA
    //    was proven, not just "some SHA was proven".
    assert_ne!(
        compiled_a.public_output_digest, compiled_b.public_output_digest,
        "state_x_out must depend on each step's SHA-256 digest"
    );

    // 8. Builder advanced the lifecycle once per appended step.
    //    audit.steps is the per-step `StepProof` trail the audit-form
    //    verifier replays.
    assert_eq!(
        chain
            .audit()
            .expect("audit after recursive append")
            .steps
            .len(),
        2,
        "builder must extend the lifecycle once per compiled SHA-256 step"
    );
    assert_eq!(
        chain.context().chain_state.step_count,
        2,
        "builder must thread chain state across base and recursive appends"
    );

    // 9. Drop the per-step compiled outputs (no longer needed by the
    //    terminal verifier path). Each `EncodedFPrimeStep` carries an
    //    `Arc<FPrimeStructure>` cloned from `prep.structure`, so the
    //    structure itself is freed only when `prep` is dropped below;
    //    dropping a compiled step frees the per-step image + witness
    //    (a few MiB each) and decrements the Arc.
    let drop_compiled = Instant::now();
    drop(compiled_a);
    drop(compiled_b);
    eprintln!(
        "[sha-ivc] {:<32} {:>7.2}s",
        "drop compiled steps",
        drop_compiled.elapsed().as_secs_f64()
    );

    // 10. Terminal verifier: finalize the chain and run the non-replay
    //     uncompressed verifier — production's IVC verifier surface.
    let proof = phase("chain.finish()", || {
        chain.finish().expect("finish SHA-256 R1CS-F' chain")
    });
    phase("verify_uncompressed", || {
        lifecycle::verify_uncompressed(&prep.prep, &proof).expect("verify_uncompressed accepts SHA-256 R1CS-F' chain")
    });

    // 11. Drop the remaining heavy allocations explicitly so the wall
    //     time is attributed to a labeled phase.
    let drop_rest = Instant::now();
    drop(proof);
    drop(prep);
    drop(artifact_a);
    drop(artifact_b);
    eprintln!(
        "[sha-ivc] {:<32} {:>7.2}s",
        "drop prep + proof + artifacts",
        drop_rest.elapsed().as_secs_f64()
    );

    eprintln!(
        "[sha-ivc] {:<32} {:>7.2}s",
        "TOTAL (incl. drops)",
        total.elapsed().as_secs_f64()
    );
}

/// Distinct 3-byte preimage for index `i` (same length ⇒ same R1CS
/// shape, different digest ⇒ genuinely different SHA instance).
fn nth_preimage(i: usize) -> Vec<u8> {
    vec![
        b'a' + (i % 26) as u8,
        b'a' + ((i / 26) % 26) as u8,
        b'a' + ((i / 676) % 26) as u8,
    ]
}

/// Real SHA-256 batched R1CS-F' chain at **K = 4** per chunk: two chunks
/// of four genuinely distinct SHA-256 instances, finalized and verified
/// end-to-end. A lighter, focused companion to
/// [`sha256_fold_cost_k61_vs_k2_benchmark`] (which is K=61 and ~8 min) —
/// this one proves the batched same-shape SHA path is correct, not just
/// fast.
///
/// `#[ignore]`d because it still folds 8 SHA-256 instances (~1 min under
/// [`sha256_tiny_params`]); opt in with
/// `cargo test --release -p neo-fold-clean --test system_sha256_bellpepper_ccs \
///   sha256_bellpepper_ivc_chain_k4_chunks -- --ignored --exact --nocapture`.
#[test]
#[ignore]
fn sha256_bellpepper_ivc_chain_k4_chunks() {
    let total = Instant::now();
    const K: usize = 4;

    let ref_artifact = synthesize_to_ccs(Sha256Circuit {
        preimage: nth_preimage(0),
    })
    .expect("reference synth");
    let m = ref_artifact.shape.inputs + ref_artifact.shape.aux;
    let plan = sha256_tiny_lifecycle_plan(m, ref_artifact.shape.inputs);
    let prep = phase("preprocess R1CS-F'", || {
        r1cs_f_prime::preprocess_sparse_seeded_with_params(
            &ref_artifact.sparse_r1cs,
            &plan,
            sha256_tiny_params(),
            SHA256_AJTAI_SEED,
        )
        .expect("preprocess")
    });

    let mk_batch = |chunk: usize| -> Vec<Vec<F>> {
        (0..K)
            .map(|i| {
                synthesize_to_ccs(Sha256Circuit {
                    preimage: nth_preimage(chunk * K + i),
                })
                .expect("synth")
                .assignment
            })
            .collect()
    };

    let mut chain = R1csChainBuilder::new(&prep).expect("chain");

    let compiled_0 = phase("chunk 0 (base, K=4)", || {
        chain
            .append_assignments(mk_batch(0))
            .expect("base K=4 chunk")
    });
    assert_eq!(compiled_0.len(), K);
    assert!(
        compiled_0.iter().all(|c| c.encoded.image.decode_is_base()),
        "every step in chunk 0 must take the base branch"
    );

    let compiled_1 = phase("chunk 1 (recursive, K=4)", || {
        chain
            .append_assignments(mk_batch(1))
            .expect("recursive K=4 chunk")
    });
    assert_eq!(compiled_1.len(), K);
    assert!(
        compiled_1.iter().all(|c| !c.encoded.image.decode_is_base()),
        "every step in chunk 1 must take the recursive branch"
    );

    // Distinct preimages ⇒ distinct per-step `state_x_out` (the SHA
    // digest is bound into the chain output via `app_public_input`).
    let mut digests: Vec<_> = compiled_0
        .iter()
        .chain(compiled_1.iter())
        .map(|c| c.public_output_digest)
        .collect();
    digests.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    digests.dedup();
    assert_eq!(
        digests.len(),
        2 * K,
        "all 8 distinct SHA must yield distinct chain outputs"
    );

    // Chain advanced once per chunk: chunk_count = 2, step_count = 2K = 8.
    let ctx = chain.context();
    assert_eq!(ctx.chain_state.chunk_count, 2);
    assert_eq!(ctx.chain_state.step_count, (2 * K) as u64);

    let audit = chain.audit().expect("audit after two K=4 chunks");
    assert_eq!(audit.steps.len(), 2, "two extends, one per K-chunk");
    assert_eq!(audit.public_batches[0].len(), K);
    assert_eq!(audit.public_batches[1].len(), K);

    let proof = phase("chain.finish()", || chain.finish().expect("finish K=4 chain"));
    phase("verify_uncompressed", || {
        lifecycle::verify_uncompressed(&prep.prep, &proof).expect("verify_uncompressed accepts K=4 SHA chain")
    });

    eprintln!("[sha-ivc-k4] TOTAL {:.2}s", total.elapsed().as_secs_f64());
}

/// **Real** SHA-256 fold-cost benchmark: K=61 vs K=2 per chunk.
///
/// Folds genuinely distinct SHA-256 instances (24,803 R1CS constraints,
/// structure ≈ 4.3M rows) through the batched R1CS-F' chain at two chunk
/// sizes and times every phase. Two chunks per K so the `finish` fold is
/// the steady-state `K + k_rho = K + 14` claim case.
///
/// Heavy and `#[ignore]`d — opt in with
/// `cargo test --release --features perf-timers -p neo-fold-clean \
///   --test system_sha256_bellpepper_ccs \
///   sha256_fold_cost_k61_vs_k2_benchmark -- --ignored --exact --nocapture`.
/// Runs well over the 5-minute slice; only run when explicitly measuring.
#[test]
#[ignore]
fn sha256_fold_cost_k61_vs_k2_benchmark() {
    let bench_total = Instant::now();
    let ref_artifact = synthesize_to_ccs(Sha256Circuit {
        preimage: nth_preimage(0),
    })
    .expect("reference synth");
    let m = ref_artifact.shape.inputs + ref_artifact.shape.aux;
    let plan = sha256_tiny_lifecycle_plan(m, ref_artifact.shape.inputs);
    let prep = phase("preprocess R1CS-F' (once)", || {
        r1cs_f_prime::preprocess_sparse_seeded_with_params(
            &ref_artifact.sparse_r1cs,
            &plan,
            sha256_tiny_params(),
            SHA256_AJTAI_SEED,
        )
        .expect("preprocess")
    });
    eprintln!(
        "[bench] SHA-256 shape: constraints={}, m={}, structure.n={}, structure.m={}, kappa={}",
        ref_artifact.shape.constraints,
        m,
        prep.prep.structure().n,
        prep.prep.structure().m,
        prep.prep.params.kappa(),
    );

    for k in [2usize, 61usize] {
        eprintln!("\n[bench] ===================== K = {k} =====================");

        let mk_batch = |chunk: usize| -> Vec<Vec<F>> {
            (0..k)
                .map(|i| {
                    synthesize_to_ccs(Sha256Circuit {
                        preimage: nth_preimage(chunk * k + i),
                    })
                    .expect("synth")
                    .assignment
                })
                .collect()
        };

        let t_synth = Instant::now();
        let batch_0 = mk_batch(0);
        let batch_1 = mk_batch(1);
        let synth_s = t_synth.elapsed().as_secs_f64();

        let mut chain = R1csChainBuilder::new(&prep).expect("chain");

        let t_base = Instant::now();
        chain
            .append_assignments(batch_0)
            .expect("base chunk (K fresh deposited, no fold)");
        let base_s = t_base.elapsed().as_secs_f64();

        // Recursive chunk: `prepare_next_fold` runs ONE NIFS prove
        // (folds the base chunk's K fresh + 0 running) and stashes the
        // post-fold audit; `compile_chunk` runs K compiles + one
        // `verify_prior_fold` (NIFS.V); the deposit then swaps the real
        // instances into the stashed audit — no second prove.
        let t_rec = Instant::now();
        chain
            .append_assignments(batch_1)
            .expect("recursive chunk (folds K fresh + 0 running)");
        let rec_s = t_rec.elapsed().as_secs_f64();

        // Finalize: NIFS folds the trailing K fresh + k_rho(14) running =
        // K + 14 claims. This is the steady-state fold.
        let t_fin = Instant::now();
        let proof = chain.finish().expect("finish");
        let fin_s = t_fin.elapsed().as_secs_f64();

        let t_ver = Instant::now();
        lifecycle::verify_uncompressed(&prep.prep, &proof).expect("verify_uncompressed");
        let ver_s = t_ver.elapsed().as_secs_f64();

        let ops = 2 * k; // total SHA folded across the two chunks
        eprintln!(
            "[bench] K={k:<2} synth={synth_s:6.2}s  base={base_s:6.2}s  recursive={rec_s:6.2}s  \
             finish={fin_s:6.2}s  verify={ver_s:6.2}s",
        );
        eprintln!(
            "[bench] K={k:<2} steady-state fold (finish, {}+14 claims) = {fin_s:6.2}s = {:.3}s/op",
            k,
            fin_s / k as f64,
        );
        eprintln!(
            "[bench] K={k:<2} all fold+compile work (base+recursive+finish) = {:6.2}s over {ops} ops = {:.3}s/op",
            base_s + rec_s + fin_s,
            (base_s + rec_s + fin_s) / ops as f64,
        );
        drop(proof);
    }
    eprintln!(
        "\n[bench] TOTAL benchmark wall {:.1}s",
        bench_total.elapsed().as_secs_f64()
    );
}
