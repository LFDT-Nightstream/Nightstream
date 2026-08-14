//! Rust-to-Lean drift gate for Nebula lifecycle-branch sharing.
//!
//! The production encoder owns three lifecycle states but only two distinct
//! relation arms. Bootstrap and steady recursion must select the same physical
//! recursive relation. The generated file records that mapping. Source hashes
//! are review sentinels only and are not protocol authority.

use neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimeBranch;
use sha2::{Digest, Sha256};

const ARTIFACT_REL_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/NebulaRecursiveArmSharingArtifact.lean";
const F_PRIME_SOURCE: &str = include_str!("../../src/frontends/nebula/f_prime.rs");
const SHAPE_SOURCE: &str = include_str!("../../src/frontends/nebula/f_prime/shape.rs");

fn sha256_hex(payload: &str) -> String {
    Sha256::digest(payload.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn render() -> String {
    let mapping = [
        NebulaFPrimeBranch::Base.relation_arm_index(),
        NebulaFPrimeBranch::BootstrapRecursive.relation_arm_index(),
        NebulaFPrimeBranch::Recursive.relation_arm_index(),
    ];
    assert_eq!(mapping, [0, 1, 1]);

    let payload = format!(
        "def schemaVersion : Nat := 1\n\
         def artifactKind : String := \"nebula/f-prime-recursive-arm-sharing\"\n\
         def logicalArmCount : Nat := {}\n\
         def physicalArmCount : Nat := {}\n\
         def logicalToPhysical : List Nat := [{}, {}, {}]\n\
         def fPrimeSourceSha256 : String := \"{}\"\n\
         def shapeSourceSha256 : String := \"{}\"\n",
        mapping.len(),
        mapping.iter().copied().max().expect("nonempty mapping") + 1,
        mapping[0],
        mapping[1],
        mapping[2],
        sha256_hex(F_PRIME_SOURCE),
        sha256_hex(SHAPE_SOURCE),
    );
    let hash = sha256_hex(&payload);
    format!(
        "import Mathlib.Data.List.GetD\n\n\
         /-!\n\
         GENERATED FILE — do not edit by hand.\n\n\
         Exact logical-to-physical arm mapping read from the production Rust\n\
         `NebulaFPrimeBranch` implementation. Regenerated and drift-checked by\n\
         `gadgets_nebula_recursive_arm_lean_artifact`. Source hashes are drift\n\
         sentinels only.\n\
         -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistory.NebulaRecursiveArmSharingArtifact\n\n\
         def artifactSha256 : String := \"{hash}\"\n\
         {payload}\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistory.NebulaRecursiveArmSharingArtifact\n"
    )
}

#[test]
fn production_recursive_lifecycle_mapping_matches_lean_artifact() {
    let emitted = render();
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != emitted {
        let expected_path = format!("{path}.expected");
        std::fs::write(&expected_path, emitted).expect("write .expected artifact");
        panic!("generated Lean recursive-arm artifact drifted. Wrote {expected_path}; inspect and copy it over {path}");
    }
}
