//! Exact compact source decoders for the production claim-replay base arms.

#[path = "support/selective_decoder_lean.rs"]
mod selective_decoder_lean;
#[path = "support/selective_decoder_run_lean.rs"]
mod selective_decoder_run_lean;

use std::ops::Range;
use std::path::{Path, PathBuf};

use neo_fold_clean::frontends::nebula::f_prime::production_claim_replay_base_compact_layout_and_decoder_runs_for_ranges;
use neo_fold_clean::frontends::r1cs_f_prime::SelectiveProjectedDecoderRunProvenance;
use selective_decoder_lean::write_decoder_arm;
use selective_decoder_run_lean::write_runs;
use sha2::{Digest, Sha256};

const PROFILE_ID: &str = "nebula-f-prime-streaming-claim-replay-goldilocks-b2-k16-v6";
const ARTIFACT_ID: &str = "rust:nightstream/streaming-claim-replay-base/source-decoders/v1";
const BASE_ARTIFACT_SHA256: &str = "fc5e19007da5e0b496de5570bcf92f73c4d24b770f7c278b1bcf53384788641b";
const FULL_SOURCE_COLUMNS: usize = 261_603;
const FINAL_SOURCE_COLUMNS: usize = 193_803;
const FINAL_COLUMNS: usize = 1_595_106;

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn decoder_requests() -> [(usize, Range<usize>); 2] {
    [(0, 1..FULL_SOURCE_COLUMNS), (1, 1..FINAL_SOURCE_COLUMNS)]
}

fn write_arm(rendered: &mut String, label: &str, decoder: &SelectiveProjectedDecoderRunProvenance) {
    for (index, template) in decoder.repeated_templates().iter().enumerate() {
        write_runs(
            rendered,
            &format!("{label}TemplateRules{index:02}"),
            "RawRun",
            template.relative_runs(),
        );
    }
    write_decoder_arm(rendered, label, decoder, &format!("{label}TemplateRules"));
}

fn render_artifact() -> String {
    let requests = decoder_requests();
    let (layout, decoders) = production_claim_replay_base_compact_layout_and_decoder_runs_for_ranges(&requests)
        .expect("audit complete production claim-replay base source decoders");
    assert_eq!(decoders.len(), requests.len());
    assert_eq!(layout.selector_columns(), [648, 649]);
    assert_eq!(layout.final_columns(), FINAL_COLUMNS);
    for (decoder, (arm, source_range)) in decoders.iter().zip(&requests) {
        assert_eq!(decoder.arm(), *arm);
        assert_eq!(decoder.source_range(), source_range.clone());
        assert_eq!(decoder.final_columns(), FINAL_COLUMNS);
    }

    let mut payload = String::new();
    write_arm(&mut payload, "full", &decoders[0]);
    write_arm(&mut payload, "final", &decoders[1]);
    let artifact_sha256 = sha256_hex(payload.as_bytes());
    let rendered = format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema\n\n\
         /-! GENERATED FILE. DO NOT EDIT. Exact compact Rust source decoders\n\
         for both deferred-overlay production claim-replay base arms. -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplayBaseDecoder\n\n\
         open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema\n\n\
         def artifactSha256 : String := \"{artifact_sha256}\"\n\
         def schemaVersion : Nat := 1\n\
         def profileId : String := \"{PROFILE_ID}\"\n\
         def artifactIdentity : String := \"{ARTIFACT_ID}\"\n\
         def baseArtifactSha256 : String := \"{BASE_ARTIFACT_SHA256}\"\n\n\
         {payload}\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplayBaseDecoder\n"
    );
    assert!(rendered.lines().count() < 1_500, "generated decoder artifact line cap");
    rendered
}

fn generated_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
         FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingClaimReplayBaseDecoder.lean",
    )
}

#[test]
fn production_claim_replay_base_decoder_artifact_is_current() {
    let path = generated_artifact_path();
    let rendered = render_artifact();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        panic!("frozen Lean reference differs: {path:?}");
    }
}
