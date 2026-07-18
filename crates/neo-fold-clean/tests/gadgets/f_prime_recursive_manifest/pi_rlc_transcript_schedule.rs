//! Fixed PiRLC transcript schedule Rust-to-Lean artifact.
//!
//! Owns: extraction of the production stage order for fifteen rho samples,
//! per-round cost attribution, and the traced Poseidon2 permutation/S-box
//! census.
//!
//! Does not own: transcript absorb contents, counter values, Poseidon2
//! functional equivalence, sampler semantics, or permission to remove rows.
//!
//! Emits constraints: no. It regenerates and profiles the production recursive
//! source relation.
//!
//! Authority boundary: materialized source dimensions and diagnostic nonlinear
//! events are read from the production builder. Low-norm rows/columns are a
//! trace-reconciled estimator output, not a materialized CCS relation.
//!
//! | Artifact branch | Mathematical obligation | Evidence tier | Lean owner |
//! |---|---|---|---|
//! | `samples` | 15 ordered samples, each with one separator and four digest rounds | diagnostic stage trace | `ProductionScheduleArtifact` |
//! | `digest` / `lanes` | Per-round source and estimated-low-norm ownership | mixed materialized/estimated | `ProductionScheduleArtifact` |
//! | `transcript`, `sampler`, `challenge` | Immediate-child cost reconciliation | trace-reconciled profiler | `ProductionScheduleArtifact` |
//! | Poseidon census | Every traced permutation owns exactly 86 S-boxes | diagnostic nonlinear trace | `ProductionScheduleArtifact` |

use std::fmt::Write as _;
use std::fs;

use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::pi_rlc_challenge_stage;
use neo_fold_clean::frontends::f_prime::gadget_native::{
    profile_r1cs_gadget_native_stages, GadgetNativeStageEstimate, GadgetNativeStageProfile,
};
use neo_fold_clean::paper::reductions::pi_rlc::PI_RLC_INPUT_CLAIMS_DIGEST_LABEL;

use super::{build_recursive_program, repo_root};

const ARTIFACT_PATH: &str = "formal/superneo-lean/SuperNeo/FPrimeRecursiveVerifier/PiRlcChallenge/Transcript/Refinement/Generated/ProductionScheduleArtifactData.lean";
const RHO_COUNT: usize = 15;
const DIGEST_ROUNDS_PER_RHO: usize = 4;
const LANES_PER_DIGEST: usize = 4;
const SBOXES_PER_PERMUTATION: usize = 86;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Cost {
    materialized_source_rows: usize,
    materialized_source_columns: usize,
    estimated_low_norm_rows: usize,
    estimated_low_norm_columns: usize,
    traced_poseidon_permutations: usize,
    traced_sboxes: usize,
}

impl Cost {
    fn from_stage(stage: &GadgetNativeStageEstimate) -> Self {
        Self {
            materialized_source_rows: stage.source_rows,
            materialized_source_columns: stage.source_cols,
            estimated_low_norm_rows: stage.encoded_rows,
            estimated_low_norm_columns: stage.encoded_cols,
            traced_poseidon_permutations: stage.poseidon_permutations,
            traced_sboxes: stage.sboxes,
        }
    }

    fn plus(self, other: Self) -> Self {
        Self {
            materialized_source_rows: self.materialized_source_rows + other.materialized_source_rows,
            materialized_source_columns: self.materialized_source_columns + other.materialized_source_columns,
            estimated_low_norm_rows: self.estimated_low_norm_rows + other.estimated_low_norm_rows,
            estimated_low_norm_columns: self.estimated_low_norm_columns + other.estimated_low_norm_columns,
            traced_poseidon_permutations: self.traced_poseidon_permutations + other.traced_poseidon_permutations,
            traced_sboxes: self.traced_sboxes + other.traced_sboxes,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DigestRound {
    rho_index: usize,
    round_index: usize,
    lane_decomposition_occurrences: usize,
    digest: Cost,
    lanes: Cost,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RhoSample {
    rho_index: usize,
    separator: Cost,
    sampler_initialization: Cost,
    rounds: Vec<DigestRound>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Artifact {
    declared_input_claims_digest_label: String,
    samples: Vec<RhoSample>,
    bind_outputs_digest: Cost,
    rho_domain_separators: Cost,
    sampler_initializations: Cost,
    digest_rounds: Cost,
    lane_decompositions: Cost,
    transcript: Cost,
    sampler: Cost,
    challenge: Cost,
}

fn aggregate(profile: &GadgetNativeStageProfile, path: &'static str) -> Cost {
    Cost::from_stage(
        &profile
            .aggregate_prefix(path)
            .unwrap_or_else(|| panic!("missing production stage {path}")),
    )
}

fn add_all(costs: impl IntoIterator<Item = Cost>) -> Cost {
    costs.into_iter().fold(Cost::default(), Cost::plus)
}

fn expect_path<'a>(
    events: &mut impl Iterator<Item = &'a GadgetNativeStageEstimate>,
    expected: &'static str,
) -> &'a GadgetNativeStageEstimate {
    let event = events
        .next()
        .unwrap_or_else(|| panic!("missing production event {expected}"));
    assert_eq!(event.label, expected, "PiRLC transcript event order drift");
    event
}

fn extract_artifact() -> Artifact {
    let builder = build_recursive_program();
    assert!(builder.is_satisfied(), "fixed plain recursive source relation");
    let source = builder.snapshot();
    let trace = builder.encoding_trace();
    let profile =
        profile_r1cs_gadget_native_stages(&source, trace, &[]).expect("fixed recursive gadget-native stage profile");

    // Synthetic replacement leaves are appended after the physical checkpoint
    // ranges. Only the prefix has the production emission order.
    let physical_stage_count = trace.stages().len() - 1;
    let mut events = profile.stages[..physical_stage_count]
        .iter()
        .filter(|stage| {
            matches!(
                stage.label,
                pi_rlc_challenge_stage::BIND_OUTPUTS_DIGEST
                    | pi_rlc_challenge_stage::RHO_DOMAIN_SEPARATOR
                    | pi_rlc_challenge_stage::SAMPLE_INITIALIZE
                    | pi_rlc_challenge_stage::TRANSCRIPT_DIGEST
                    | pi_rlc_challenge_stage::LANE_BIT_DECOMPOSITION
            )
        });

    let bind_outputs_digest = Cost::from_stage(expect_path(&mut events, pi_rlc_challenge_stage::BIND_OUTPUTS_DIGEST));
    let mut samples = Vec::with_capacity(RHO_COUNT);
    for rho_index in 0..RHO_COUNT {
        let separator = Cost::from_stage(expect_path(&mut events, pi_rlc_challenge_stage::RHO_DOMAIN_SEPARATOR));
        let sampler_initialization =
            Cost::from_stage(expect_path(&mut events, pi_rlc_challenge_stage::SAMPLE_INITIALIZE));
        let mut rounds = Vec::with_capacity(DIGEST_ROUNDS_PER_RHO);
        for round_index in 0..DIGEST_ROUNDS_PER_RHO {
            let digest = Cost::from_stage(expect_path(&mut events, pi_rlc_challenge_stage::TRANSCRIPT_DIGEST));
            let lanes =
                add_all((0..LANES_PER_DIGEST).map(|_| {
                    Cost::from_stage(expect_path(&mut events, pi_rlc_challenge_stage::LANE_BIT_DECOMPOSITION))
                }));
            rounds.push(DigestRound {
                rho_index,
                round_index,
                lane_decomposition_occurrences: LANES_PER_DIGEST,
                digest,
                lanes,
            });
        }
        samples.push(RhoSample {
            rho_index,
            separator,
            sampler_initialization,
            rounds,
        });
    }
    assert!(events.next().is_none(), "unexpected PiRLC transcript event");

    let artifact = Artifact {
        declared_input_claims_digest_label: std::str::from_utf8(PI_RLC_INPUT_CLAIMS_DIGEST_LABEL)
            .expect("ASCII PiRLC input-claims label")
            .to_owned(),
        samples,
        bind_outputs_digest,
        rho_domain_separators: aggregate(&profile, pi_rlc_challenge_stage::RHO_DOMAIN_SEPARATOR),
        sampler_initializations: aggregate(&profile, pi_rlc_challenge_stage::SAMPLE_INITIALIZE),
        digest_rounds: aggregate(&profile, pi_rlc_challenge_stage::TRANSCRIPT_DIGEST),
        lane_decompositions: aggregate(&profile, pi_rlc_challenge_stage::LANE_BIT_DECOMPOSITION),
        transcript: aggregate(&profile, pi_rlc_challenge_stage::TRANSCRIPT),
        sampler: aggregate(&profile, pi_rlc_challenge_stage::SAMPLER),
        challenge: aggregate(&profile, pi_rlc_challenge_stage::CHALLENGE),
    };

    assert_eq!(artifact.samples.len(), RHO_COUNT);
    assert_eq!(
        artifact
            .samples
            .iter()
            .map(|sample| sample.rounds.len())
            .sum::<usize>(),
        RHO_COUNT * DIGEST_ROUNDS_PER_RHO,
    );
    assert_eq!(
        add_all(artifact.samples.iter().map(|sample| sample.separator)),
        artifact.rho_domain_separators,
    );
    assert_eq!(
        add_all(
            artifact
                .samples
                .iter()
                .map(|sample| sample.sampler_initialization),
        ),
        artifact.sampler_initializations,
    );
    assert_eq!(
        add_all(
            artifact
                .samples
                .iter()
                .flat_map(|sample| sample.rounds.iter())
                .map(|round| round.digest),
        ),
        artifact.digest_rounds,
    );
    assert_eq!(
        add_all(
            artifact
                .samples
                .iter()
                .flat_map(|sample| sample.rounds.iter())
                .map(|round| round.lanes),
        ),
        artifact.lane_decompositions,
    );
    assert_eq!(
        artifact
            .bind_outputs_digest
            .plus(artifact.rho_domain_separators)
            .plus(artifact.digest_rounds)
            .plus(artifact.lane_decompositions),
        artifact.transcript,
        "transcript immediate-child reconciliation",
    );
    assert_eq!(
        artifact.transcript.plus(artifact.sampler),
        artifact.challenge,
        "challenge immediate-child reconciliation",
    );
    for cost in artifact
        .samples
        .iter()
        .flat_map(|sample| {
            std::iter::once(sample.separator).chain(
                sample
                    .rounds
                    .iter()
                    .flat_map(|round| [round.digest, round.lanes]),
            )
        })
        .chain([artifact.bind_outputs_digest])
    {
        assert_eq!(
            cost.traced_sboxes,
            SBOXES_PER_PERMUTATION * cost.traced_poseidon_permutations,
            "Poseidon2 permutation/S-box census",
        );
    }
    assert_eq!(
        (
            artifact.challenge.materialized_source_rows,
            artifact.challenge.materialized_source_columns,
            artifact.challenge.estimated_low_norm_rows,
            artifact.challenge.estimated_low_norm_columns,
        ),
        (127_611, 121_566, 198_567, 370_383),
        "current fixed PiRLC challenge dimensions",
    );
    artifact
}

fn render_cost(cost: Cost) -> String {
    format!(
        "{{ materializedSourceRows := {}, materializedSourceColumns := {}, estimatedLowNormRows := {}, estimatedLowNormColumns := {}, tracedPoseidonPermutations := {}, tracedSboxes := {} }}",
        cost.materialized_source_rows,
        cost.materialized_source_columns,
        cost.estimated_low_norm_rows,
        cost.estimated_low_norm_columns,
        cost.traced_poseidon_permutations,
        cost.traced_sboxes,
    )
}

fn render(artifact: &Artifact) -> String {
    let mut out = String::from(
        "import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Transcript.Refinement.ProductionScheduleArtifactSchema\n\n\
/-! Generated fixed plain-F-prime PiRLC transcript schedule data; do not hand-edit.\n\n\
Owns: the exact production stage order, materialized source cost, trace-reconciled\n\
low-norm estimate, and diagnostic Poseidon2 census for fifteen rho samples.\n\n\
Does not own: absorbed label/field conformance, counter-value conformance, the\n\
Poseidon2 function, sampler correctness, a materialized low-norm relation, or\n\
permission to remove rows.\n\n\
Emits constraints: no.\n\n\
Authority boundary: source counts come from the satisfied production R1CS.\n\
Estimated counts come from the validated profiler and remain estimator-only.\n\
No digest authorizes any field below.\n\n\
| Data branch | Meaning | Evidence tier |\n\
|---|---|---|\n\
| `samples` | ordered 15-by-4 production checkpoints | diagnostic stage trace |\n\
| `materializedSource*` | rows/columns emitted by the satisfied source builder | materialized source R1CS |\n\
| `estimatedLowNorm*` | compact compiler cost model | trace-reconciled estimate only |\n\
| `tracedPoseidon*` | nonlinear events attributed to exact source ranges | diagnostic nonlinear trace |\n\
-/\n\n\
namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.ProductionScheduleArtifactData\n\n\
open ProductionScheduleArtifact\n\n",
    );
    writeln!(out, "def schemaVersion : Nat := 1").unwrap();
    writeln!(out, "def sourceCostTier : EvidenceTier := .materializedSourceR1cs").unwrap();
    writeln!(out, "def encodedCostTier : EvidenceTier := .traceReconciledEstimate").unwrap();
    writeln!(out, "def nonlinearCensusTier : EvidenceTier := .diagnosticTrace").unwrap();
    writeln!(out, "def stageOrderTraced : Bool := true").unwrap();
    writeln!(out, "def absorbContentsTraced : Bool := false").unwrap();
    writeln!(out, "def counterValuesTraced : Bool := false").unwrap();
    writeln!(out, "def poseidonFunctionTraced : Bool := false").unwrap();
    writeln!(
        out,
        "def declaredInputClaimsDigestLabel : String := {:?}",
        artifact.declared_input_claims_digest_label,
    )
    .unwrap();
    writeln!(
        out,
        "def challengePath : String := {:?}",
        pi_rlc_challenge_stage::CHALLENGE
    )
    .unwrap();
    writeln!(
        out,
        "def transcriptPath : String := {:?}",
        pi_rlc_challenge_stage::TRANSCRIPT
    )
    .unwrap();
    writeln!(out, "def samplerPath : String := {:?}", pi_rlc_challenge_stage::SAMPLER).unwrap();
    writeln!(
        out,
        "def bindOutputsDigestPath : String := {:?}",
        pi_rlc_challenge_stage::BIND_OUTPUTS_DIGEST,
    )
    .unwrap();
    writeln!(
        out,
        "def rhoDomainSeparatorPath : String := {:?}",
        pi_rlc_challenge_stage::RHO_DOMAIN_SEPARATOR,
    )
    .unwrap();
    writeln!(
        out,
        "def samplerInitializePath : String := {:?}",
        pi_rlc_challenge_stage::SAMPLE_INITIALIZE,
    )
    .unwrap();
    writeln!(
        out,
        "def digestRoundsPath : String := {:?}",
        pi_rlc_challenge_stage::TRANSCRIPT_DIGEST,
    )
    .unwrap();
    writeln!(
        out,
        "def laneDecompositionPath : String := {:?}",
        pi_rlc_challenge_stage::LANE_BIT_DECOMPOSITION,
    )
    .unwrap();

    out.push_str("def samples : List RhoSample := [\n");
    for (sample_index, sample) in artifact.samples.iter().enumerate() {
        let prefix = if sample_index == 0 { "  " } else { ", " };
        writeln!(
            out,
            "{prefix}{{ rhoIndex := {}, separator := {}, samplerInitialization := {}, rounds := [",
            sample.rho_index,
            render_cost(sample.separator),
            render_cost(sample.sampler_initialization),
        )
        .unwrap();
        for (round_index, round) in sample.rounds.iter().enumerate() {
            let round_prefix = if round_index == 0 { "      " } else { "    , " };
            writeln!(
                out,
                "{round_prefix}{{ rhoIndex := {}, roundIndex := {}, laneDecompositionOccurrences := {}, digest := {}, lanes := {} }}",
                round.rho_index,
                round.round_index,
                round.lane_decomposition_occurrences,
                render_cost(round.digest),
                render_cost(round.lanes),
            )
            .unwrap();
        }
        out.push_str("    ] }\n");
    }
    out.push_str("]\n");
    for (name, cost) in [
        ("bindOutputsDigest", artifact.bind_outputs_digest),
        ("rhoDomainSeparators", artifact.rho_domain_separators),
        ("samplerInitializations", artifact.sampler_initializations),
        ("digestRounds", artifact.digest_rounds),
        ("laneDecompositions", artifact.lane_decompositions),
        ("transcript", artifact.transcript),
        ("sampler", artifact.sampler),
        ("challenge", artifact.challenge),
    ] {
        writeln!(out, "def {name} : Cost := {}", render_cost(cost)).unwrap();
    }
    out.push_str("\nend SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.ProductionScheduleArtifactData\n");
    out
}

#[test]
fn pi_rlc_transcript_schedule_artifact_matches_fixed_recursive_production() {
    let artifact = extract_artifact();

    let mut corrupted = artifact.clone();
    corrupted.samples[0].rounds[0].lane_decomposition_occurrences -= 1;
    assert_ne!(corrupted, artifact, "schedule mutation must fail closed");
    let mut corrupted = artifact.clone();
    corrupted.samples[0].rounds[0]
        .digest
        .traced_poseidon_permutations += 1;
    assert_ne!(corrupted, artifact, "Poseidon census mutation must fail closed");

    let rendered = render(&artifact);
    let path = repo_root().join(ARTIFACT_PATH);
    let committed = fs::read_to_string(&path).unwrap_or_default();
    if committed != rendered {
        let expected = path.with_extension("lean.expected");
        fs::create_dir_all(expected.parent().expect("artifact parent"))
            .expect("create PiRLC transcript artifact directory");
        fs::write(&expected, &rendered).expect("write expected PiRLC transcript artifact");
        panic!(
            "PiRLC transcript schedule artifact drifted; review {}",
            expected.display(),
        );
    }
}
