//! Deterministic Rust-to-Lean witness data for the four fixed claim-digest
//! domain-separation permutations.

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
const ARTIFACT_KIND: &str = "nebula/f-prime/streaming-claim-digest-domain-witnesses";

const INPUTS: [[u64; 8]; 4] = [
    [1, 44, 27_428_916_078_536_046, 32_774_695_491_433_326, 0, 0, 0, 0],
    [
        32_492_151_232_362_031,
        12_721_823_848_622_437,
        31_362_922_327_076_711,
        12_728_458_466_782_051,
        14_285_079_556_494_616_407,
        10_815_961_559_651_698_320,
        18_260_705_026_218_357_875,
        8_424_582_804_285_107_233,
    ],
    [
        13_426,
        54,
        30_521_782_141_150_574,
        31_069_335_676_202_596,
        2_250_958_222_046_521_952,
        15_440_103_732_085_447_511,
        6_547_395_164_335_706_312,
        10_272_340_994_444_577_429,
    ],
    [
        27_422_324_158_721_583,
        30_796_712_690_673_199,
        27_414_614_995_316_581,
        30_508_344_144_718_189,
        5_510_511_107_940_736_673,
        9_313_957_564_688_670_493,
        1_768_918_467_504_634_943,
        14_404_461_253_576_933_351,
    ],
];

const OUTPUTS: [[u64; 8]; 4] = [
    [
        15_335_109_097_073_140_235,
        17_619_563_798_142_362_813,
        11_645_649_210_628_966_215,
        4_436_364_367_798_357_556,
        14_285_079_556_494_616_407,
        10_815_961_559_651_698_320,
        18_260_705_026_218_357_875,
        8_424_582_804_285_107_233,
    ],
    [
        12_824_850_162_859_434_436,
        16_884_249_588_232_806_831,
        1_238_414_030_862_021_266,
        16_194_180_760_988_878_864,
        2_250_958_222_046_521_952,
        15_440_103_732_085_447_511,
        6_547_395_164_335_706_312,
        10_272_340_994_444_577_429,
    ],
    [
        6_958_520_756_929_626_742,
        16_947_407_995_347_177_160,
        4_955_651_861_384_673_240,
        11_357_146_294_475_889_773,
        5_510_511_107_940_736_673,
        9_313_957_564_688_670_493,
        1_768_918_467_504_634_943,
        14_404_461_253_576_933_351,
    ],
    [
        16_534_366_849_561_726_655,
        6_810_547_603_550_404_849,
        7_420_078_321_807_019_432,
        14_323_236_552_110_360_532,
        1_298_986_797_814_860_681,
        17_392_165_756_113_845_022,
        8_388_603_933_087_874_784,
        14_187_929_483_296_301_137,
    ],
];

fn build_witness(input: [u64; 8]) -> Vec<F> {
    let mut builder = R1csBuilder::new();
    let input_vars = input.map(|value| builder.alloc(F::from_u64(value)));
    let output = enforce_poseidon2_permutation(&mut builder, &input_vars);
    assert_eq!(builder.rows(), 600);
    assert_eq!(builder.cols(), 609);
    assert!(builder.is_satisfied());
    let output_values = output.map(|column| builder.witness()[column.col()].as_canonical_u64());
    assert_eq!(output_values, OUTPUTS[input_checkpoint(input)]);
    builder.witness().to_vec()
}

fn input_checkpoint(input: [u64; 8]) -> usize {
    INPUTS
        .iter()
        .position(|candidate| *candidate == input)
        .expect("checkpoint input is listed")
}

fn render_witness(name: &str, witness: &[F]) -> String {
    let mut rendered = format!("def {name} : List Nat :=\n  [");
    for (index, value) in witness.iter().enumerate() {
        if index > 0 {
            rendered.push(',');
            if index % 8 == 0 {
                rendered.push_str("\n   ");
            } else {
                rendered.push(' ');
            }
        }
        write!(rendered, "{}", value.as_canonical_u64()).unwrap();
    }
    rendered.push_str("]\n");
    rendered
}

fn render_artifact() -> String {
    let witnesses = INPUTS.map(build_witness);
    let mut payload = String::new();
    writeln!(payload, "def schemaVersion : Nat := {SCHEMA_VERSION}").unwrap();
    writeln!(payload, "def artifactKind : String := \"{ARTIFACT_KIND}\"\n").unwrap();
    for (index, witness) in witnesses.iter().enumerate() {
        writeln!(
            payload,
            "{}",
            render_witness(&format!("checkpoint{}Witness", index + 1), witness)
        )
        .unwrap();
    }
    writeln!(payload, "theorem checkpointWitnessLengths :").unwrap();
    writeln!(payload, "    checkpoint1Witness.length = 609 ∧").unwrap();
    writeln!(payload, "      checkpoint2Witness.length = 609 ∧").unwrap();
    writeln!(payload, "      checkpoint3Witness.length = 609 ∧").unwrap();
    writeln!(payload, "      checkpoint4Witness.length = 609 := by").unwrap();
    writeln!(payload, "  native_decide").unwrap();

    let hash = sha256_hex(&payload);
    format!(
        "import Nightstream.Implementation.R1CS.Core.Semantics\n\n\
         /-! Generated fixed witnesses for the four Poseidon2 permutations in\n\
         the streaming claim-digest application-domain frame. Regenerate only\n\
         through the Rust drift gate. -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimDigestWitnesses\n\n\
         def artifactSha256 : String := \"{hash}\"\n\n\
         {payload}\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimDigestWitnesses\n"
    )
}

fn generated_artifact_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
         FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingClaimDigestWitnesses.lean",
    )
}

#[test]
fn streaming_claim_digest_poseidon_witness_artifact_is_current() {
    let path = generated_artifact_path();
    let rendered = render_artifact();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        let expected = path.with_extension("lean.expected");
        std::fs::write(&expected, rendered).expect("write expected claim-digest witness artifact");
        panic!("claim-digest witness artifact drifted; inspect {}", expected.display());
    }
}

#[test]
#[ignore = "explicit deterministic artifact regeneration"]
fn regenerate_streaming_claim_digest_poseidon_witness_artifact() {
    std::fs::write(generated_artifact_path(), render_artifact())
        .expect("write generated claim-digest witness artifact");
}
