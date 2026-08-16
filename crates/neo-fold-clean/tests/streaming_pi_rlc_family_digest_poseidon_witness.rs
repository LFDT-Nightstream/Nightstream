//! Deterministic Rust-to-Lean witness data for the changed fourth
//! permutation in the PiRLC family-state digest domain.

#[path = "gadgets/lean_artifact_support.rs"]
#[allow(dead_code)]
mod lean_artifact_support;

use std::fmt::Write as _;
use std::path::Path;

use lean_artifact_support::sha256_hex;
use neo_fold_clean::engine::r1cs_circuit::{enforce_poseidon2_permutation, R1csBuilder};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const SCHEMA_VERSION: usize = 1;
const ARTIFACT_KIND: &str = "nebula/f-prime/streaming-pirlc-family-digest-domain-witness";

const CHECKPOINT4_INPUT: [u64; 8] = [
    27_422_324_158_721_583,
    30_796_712_690_673_199,
    27_414_614_995_316_581,
    29_678_212_865_747_309,
    5_510_511_107_940_736_673,
    9_313_957_564_688_670_493,
    1_768_918_467_504_634_943,
    14_404_461_253_576_933_351,
];

struct Checkpoint4 {
    output: [u64; 8],
    witness: Vec<F>,
}

fn build_checkpoint4() -> Checkpoint4 {
    let mut builder = R1csBuilder::new();
    let input_vars = CHECKPOINT4_INPUT.map(|value| builder.alloc(F::from_u64(value)));
    let output_vars = enforce_poseidon2_permutation(&mut builder, &input_vars);
    assert_eq!(builder.rows(), 600);
    assert_eq!(builder.cols(), 609);
    assert!(builder.is_satisfied());

    Checkpoint4 {
        output: output_vars.map(|column| builder.witness()[column.col()].as_canonical_u64()),
        witness: builder.witness().to_vec(),
    }
}

fn render_nat_list(name: &str, values: impl IntoIterator<Item = u64>) -> String {
    let mut rendered = format!("def {name} : List Nat :=\n  [");
    for (index, value) in values.into_iter().enumerate() {
        if index > 0 {
            rendered.push(',');
            if index % 8 == 0 {
                rendered.push_str("\n   ");
            } else {
                rendered.push(' ');
            }
        }
        write!(rendered, "{value}").unwrap();
    }
    rendered.push_str("]\n");
    rendered
}

fn render_artifact() -> String {
    let checkpoint = build_checkpoint4();
    let mut payload = String::new();
    writeln!(payload, "def schemaVersion : Nat := {SCHEMA_VERSION}").unwrap();
    writeln!(payload, "def artifactKind : String := \"{ARTIFACT_KIND}\"\n").unwrap();
    writeln!(
        payload,
        "{}",
        render_nat_list("checkpoint4InputValues", CHECKPOINT4_INPUT)
    )
    .unwrap();
    writeln!(
        payload,
        "{}",
        render_nat_list("checkpoint4OutputValues", checkpoint.output)
    )
    .unwrap();
    writeln!(
        payload,
        "{}",
        render_nat_list(
            "checkpoint4Witness",
            checkpoint
                .witness
                .iter()
                .map(|value| value.as_canonical_u64())
        )
    )
    .unwrap();
    writeln!(payload, "theorem checkpoint4WitnessLength :").unwrap();
    writeln!(payload, "    checkpoint4Witness.length = 609 := by").unwrap();
    writeln!(payload, "  native_decide").unwrap();

    let hash = sha256_hex(&payload);
    format!(
        "import Nightstream.Implementation.R1CS.Core.Semantics\n\n\
         /-! Generated fixed witness for the changed fourth Poseidon2\n\
         permutation in the PiRLC family-state digest application-domain\n\
         frame. Regenerate only through the Rust drift gate. -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyDigestWitness\n\n\
         def artifactSha256 : String := \"{hash}\"\n\n\
         {payload}\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyDigestWitness\n"
    )
}

fn generated_artifact_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
         FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyDigestWitness.lean",
    )
}

#[test]
fn streaming_pi_rlc_family_digest_poseidon_witness_artifact_is_current() {
    let path = generated_artifact_path();
    let rendered = render_artifact();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        let expected = path.with_extension("lean.expected");
        std::fs::write(&expected, rendered).expect("write expected PiRLC family-digest witness artifact");
        panic!(
            "PiRLC family-digest witness artifact drifted; inspect {}",
            expected.display()
        );
    }
}

#[test]
#[ignore = "explicit deterministic artifact regeneration"]
fn regenerate_streaming_pi_rlc_family_digest_poseidon_witness_artifact() {
    std::fs::write(generated_artifact_path(), render_artifact())
        .expect("write generated PiRLC family-digest witness artifact");
}
