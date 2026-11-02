//! A real light fold (Lean `Spec/Folding/Light`): step 1 of the light
//! circuit folds step 0's committed instance into the default slot with a
//! light PiCCS proof and the light PiRLC (`ρ` on the fresh output, 1 on the
//! running output). Every row of step 1's relation holds, on the source
//! columns and through the slot layout.

mod option_b_support;

use std::ops::Range;

use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix;
use neo_ajtai::{s_mul_add, Commitment};
use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, GeometricRowRun, Mat};
use neo_math::ring::{cf_inv, Rq};
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::{rlc_with_commit_refs, FoldingMode};
use neo_reductions::common::{sample_rot_rhos_n_typed, RotRho, RotRing};
use neo_reductions::optimized_engine::optimized_prove_with_profile;
use neo_reductions::superneo_eval::{MatrixRowSink, MatrixRows, MatrixShape, MatrixWindow};
use neo_reductions::{FoldProfile, NormSources, PiCcsError};
use neo_transcript::{domain_chunk_v1_2, Poseidon2Transcript};
use nightstream_fprime::{
    derive_light_transcript, encode_digest, FoldWords, KWords, LogicalLayout, OptionBState, PlanRows, Slot, StepWords,
};
use option_b_support::{assert_every_row_holds, canonical, hash_chain_step, packages, words, StepZero};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const CUBE_VARIABLES: usize = 28;
const PUBLIC: usize = 270;

/// The light relation's CCS rows, read through the slot layout.
struct Rows<'a> {
    rows: &'a PlanRows,
    layout: &'a LogicalLayout,
}

impl MatrixRows for Rows<'_> {
    fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.rows.row_count(),
            columns: self.layout.width().div_ceil(D) * D,
            matrices: 4,
        }
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        for row in rows {
            for port in 0..4 {
                let mut failure = None;
                self.layout
                    .visit_runs(self.rows, row, port, |run| {
                        if failure.is_none() {
                            let run = GeometricRowRun::new(row, run.column, run.count, run.coefficient, run.ratio);
                            failure = sink.push_run(row, port, run).err();
                        }
                    })
                    .map_err(|error| PiCcsError::InvalidInput(error.to_string()))?;
                if let Some(error) = failure {
                    return Err(error);
                }
                if sink.finish_matrix_row(row, port)?.is_break() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

/// `A · B − C + S^7` over the logical columns (`GatePolynomial.terms`).
fn structure(rows: &PlanRows, layout: &LogicalLayout) -> CcsStructure<F> {
    let polynomial = SparsePoly::new(
        4,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![1, 1, 0, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 1, 0],
            },
            Term {
                coeff: F::ONE,
                exps: vec![0, 0, 0, 7],
            },
        ],
    );
    CcsStructure::new_verifier_artifact_header(rows.row_count(), layout.width(), 4, polynomial)
        .and_then(|header| header.with_domain_variables(CUBE_VARIABLES))
        .expect("light CCS header")
}

/// HyperNova Definition 12's default running instance (`Step.defaultSlot`).
fn default_claim(kappa: usize) -> CeClaim<Commitment, F, K> {
    CeClaim {
        c: Commitment::zeros(kappa),
        X: Mat::zero(D, PUBLIC / D, F::ZERO),
        r: vec![K::ZERO; CUBE_VARIABLES],
        eval_k: vec![K::ZERO; D.next_power_of_two()],
        eval_a: vec![vec![K::ZERO; D.next_power_of_two()]; 4],
        m_in: PUBLIC,
        fold_digest: [0; 32],
        adv: None,
    }
}

/// The PiRLC commitment combination (`folding::ajtai_rlc_mixer`).
fn mixer(rhos: &[Mat<F>], commits: &[Commitment]) -> Commitment {
    let mut combined = Commitment::zeros(commits[0].kappa());
    for (rho, commitment) in rhos.iter().zip(commits) {
        let ring: Rq = cf_inv(std::array::from_fn(|lane| rho[(lane, 0)]));
        s_mul_add(&mut combined, &ring, commitment);
    }
    combined
}

fn k_words(value: K) -> KWords {
    value.as_coeffs()
}

/// A claim as the 270 `K` values a slot or a PiCCS output carries: `Eval_K`,
/// then the four `Eval_A` families.
fn evaluations(claim: &CeClaim<Commitment, F, K>) -> Vec<KWords> {
    claim.eval_k[..D]
        .iter()
        .chain(claim.eval_a.iter().flat_map(|family| family[..D].iter()))
        .copied()
        .map(k_words)
        .collect()
}

fn slot(claim: &CeClaim<Commitment, F, K>, depth: u64) -> Slot {
    Slot {
        commitment: claim.c.data().to_vec(),
        evaluations: evaluations(claim),
        point: claim.r.iter().copied().map(k_words).collect(),
        public_input: (0..PUBLIC)
            .map(|column| claim.X[(column % D, column / D)])
            .collect(),
        depth,
    }
}

/// One light fold of the fresh logical assignment `fresh` into the running
/// claim and witness: the fresh commitment, the PiCCS outputs and rounds,
/// and the combined claim and witness.
struct LightFold {
    commitment: Commitment,
    outputs: Vec<CeClaim<Commitment, F, K>>,
    rounds: Vec<[KWords; 9]>,
    combined: CeClaim<Commitment, F, K>,
    witness: Mat<F>,
}

struct Light<'a> {
    rows: &'a PlanRows,
    layout: &'a LogicalLayout,
    structure: CcsStructure<F>,
    params: NeoParams,
}

impl<'a> Light<'a> {
    fn new(rows: &'a PlanRows, layout: &'a LogicalLayout) -> Self {
        let structure = structure(rows, layout);
        let base = NeoParams::nightstream_goldilocks_k16();
        let bits = base
            .padded_row_security_summary_for_shape(
                structure.n(),
                structure.m(),
                structure.t(),
                structure.max_degree(),
                neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
            )
            .expect("security summary")
            .security_bits;
        println!(
            "light relation: {} rows, {} columns, {bits} statistical bits",
            structure.n(),
            structure.m()
        );
        Self {
            rows,
            layout,
            structure,
            params: base
                .with_lambda(base.lambda().min(bits))
                .expect("parameters"),
        }
    }

    fn blocks(&self) -> usize {
        self.layout.width().div_ceil(D)
    }

    fn fold(&self, fresh: &[F], running: &CeClaim<Commitment, F, K>, running_witness: &Mat<F>) -> LightFold {
        let blocks = self.blocks();
        let witness = Mat::from_row_major(
            D,
            blocks,
            (0..D)
                .flat_map(|lane| (0..blocks).map(move |block| (block, lane)))
                .map(|(block, lane)| fresh.get(block * D + lane).copied().unwrap_or(F::ZERO))
                .collect(),
        );
        let commitment = commit_production_signed_unit_prefix_matrix(&witness).expect("commitment");
        let claim = CcsClaim {
            c: commitment.clone(),
            x: fresh[..PUBLIC].to_vec(),
            m_in: PUBLIC,
            adv: None,
        };
        let matrix_rows = Rows {
            rows: self.rows,
            layout: self.layout,
        };
        let payload = 2 * 4 * size_of::<K>() + size_of::<F>() + size_of::<K>();
        let workspace =
            MatrixWindow::required_workspace(&matrix_rows, 0..self.rows.row_count(), payload).expect("workspace") * 2;
        let profile = FoldProfile {
            domain: domain_chunk_v1_2(b"Nightstream/SuperNeo/fold/light/v1"),
            children: &[],
            norm: NormSources::Fresh,
        };
        let mut transcript = Poseidon2Transcript::new_v1_2();
        let (outputs, proof, _) = optimized_prove_with_profile(
            &mut transcript,
            &self.params,
            &self.structure,
            std::slice::from_ref(&claim),
            std::slice::from_ref(&CcsWitness { Z: witness.clone() }),
            std::slice::from_ref(running),
            std::slice::from_ref(running_witness),
            &matrix_rows,
            workspace,
            None,
            &profile,
        )
        .expect("light PiCCS proof");
        let rho = sample_rot_rhos_n_typed(&mut transcript, &self.params, &RotRing::goldilocks(), 1)
            .expect("light challenge")
            .remove(0);
        let (combined, combined_witness) = rlc_with_commit_refs(
            FoldingMode::Optimized,
            &self.structure,
            &self.params,
            &[rho, RotRho::identity()],
            &outputs,
            &[&witness, running_witness],
            D.next_power_of_two().trailing_zeros() as usize,
            mixer,
        )
        .expect("light PiRLC");
        LightFold {
            commitment,
            rounds: proof
                .sumcheck_rounds
                .iter()
                .map(|round| {
                    std::array::from_fn(|coefficient| k_words(round.get(coefficient).copied().unwrap_or(K::ZERO)))
                })
                .collect(),
            outputs,
            combined,
            witness: combined_witness,
        }
    }
}

/// Step `iteration` of the light circuit: it folds the previous step's
/// instance into slot 0, which carries `prior`, and outputs `next`.
fn light_step(
    step: &StepZero,
    iteration: u64,
    states: [[F; 4]; 3],
    message: [F; 4],
    prior: &Slot,
    fold: &LightFold,
    depth: u64,
) -> (StepWords, Slot, [F; 4]) {
    let default = Slot::default_slot();
    let state = |iteration, current, slot: &Slot| {
        OptionBState {
            keys: step.keys,
            iteration,
            z0: step.z0,
            current,
            pc: 1,
            slots: [slot.clone(), default.clone()],
        }
        .hash()
    };
    let folded = slot(&fold.combined, depth);
    let output = state(iteration + 1, states[2], &folded);
    let words = StepWords {
        iteration,
        prior_pc: 1,
        selected: 0,
        keys: step.keys,
        states,
        witness: message.to_vec(),
        slots: [
            prior.clone(),
            default.clone(),
            prior.clone(),
            folded.clone(),
            default.clone(),
            folded.clone(),
            default.clone(),
        ],
        digests: [
            prior.digest(),
            default.digest(),
            folded.digest(),
            default.digest(),
            output,
        ],
        fresh_commitment: fold.commitment.data().to_vec(),
        fresh_public_input: encode_digest(state(iteration, states[1], prior)),
        rounds: fold.rounds.clone(),
        fold: FoldWords::Light {
            outputs: [evaluations(&fold.outputs[0]), evaluations(&fold.outputs[1])],
        },
    };
    (words, folded, output)
}

/// Steps 1 and 2 of the light circuit with real light folds. Step 1 folds
/// step 0's instance into the default slot; step 2 folds step 1's instance
/// into the accumulator that step 1 made, whose witness lies outside
/// {-1, 0, 1}, so the light rule that skips the running source's norm is
/// what lets it fold.
#[test]
fn light_steps_fold_real_instances_and_hold_every_row() {
    let packages = packages();
    let light = packages.light();
    let step = StepZero::new(0);
    let zero_transcript = derive_light_transcript(
        &[0; 1188],
        &canonical(&step.fresh_public_input),
        &vec![vec![[0u64; 2]; 9]; 28],
        &vec![0; 2 * 540],
    )
    .expect("light transcript");
    let fold = FoldWords::Light {
        outputs: [vec![[F::ZERO; 2]; 270], vec![[F::ZERO; 2]; 270]],
    };
    let logical = assert_every_row_holds(light, &step.words(step.folded(&zero_transcript, 1, 1), fold));

    let rows = light.plan_rows().expect("plan rows");
    let layout = light.layout().expect("layout");
    let prover = Light::new(&rows, &layout);
    let default = Slot::default_slot();
    let first = prover.fold(
        &logical,
        &default_claim(prover.params.kappa() as usize),
        &Mat::zero(D, prover.blocks(), F::ZERO),
    );
    let message = words([30, 31, 32, 33]);
    let second_state = hash_chain_step(step.next, message);
    let (words_one, slot_one, _) = light_step(
        &step,
        1,
        [step.z0, step.next, second_state],
        message,
        &default,
        &first,
        1,
    );
    assert_eq!(words_one.fresh_public_input, encode_digest(step.output));
    let logical_one = assert_every_row_holds(light, &words_one);

    let second = prover.fold(&logical_one, &first.combined, &first.witness);
    assert!(
        first
            .witness
            .as_slice()
            .iter()
            .any(|value| !matches!(value.as_canonical_u64(), 0 | 1) && *value != -F::ONE),
        "the accumulator's witness leaves the digit range"
    );
    let message = words([40, 41, 42, 43]);
    let third_state = hash_chain_step(second_state, message);
    let (words_two, _, _) = light_step(
        &step,
        2,
        [step.z0, second_state, third_state],
        message,
        &slot_one,
        &second,
        2,
    );
    assert_every_row_holds(light, &words_two);
}
