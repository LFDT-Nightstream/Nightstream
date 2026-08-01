//! Rust-to-Lean drift gate for the canonical rectangular-paper gamma layout.
//!
//! This bounded artifact records every carried gamma exponent for one fixed
//! shape. Lean compares it with the independent `PaperJoint` coordinate model.

use neo_math::D;
use neo_reductions::engines::pi_ccs_protocol::carried_gamma_exponent;

const FRESH_COUNT: usize = 2;
const RUNNING_COUNT: usize = 2;
const MATRIX_COUNT: usize = 3;
const ARTIFACT_REL_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/Rust/CanonicalConformance/PiCcsPaperRectangular/Generated/Layout.lean";

fn render() -> String {
    let mut carried = Vec::with_capacity(RUNNING_COUNT * MATRIX_COUNT * D);
    for coefficient in 0..D {
        for matrix in 0..MATRIX_COUNT {
            for running in 0..RUNNING_COUNT {
                carried.push(carried_gamma_exponent(
                    FRESH_COUNT,
                    RUNNING_COUNT,
                    MATRIX_COUNT,
                    running,
                    matrix,
                    coefficient,
                ));
            }
        }
    }
    let carried: Vec<String> = carried.iter().map(usize::to_string).collect();

    format!(
        "import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Coefficients\n\n\
/-!\n\
GENERATED FILE - do not edit by hand.\n\n\
Exact output of Rust `carried_gamma_exponent` for one bounded complete\n\
coordinate traversal. Regenerated and drift-checked by\n\
`cargo test -p neo-reductions --release --test paper_rectangular_lean_artifact`.\n\n\
This is implementation evidence. The handwritten consumer proves equality\n\
with the independent PaperJoint coordinate model.\n\
-/\n\n\
namespace Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaperRectangular.Generated\n\n\
def freshCount : Nat := {FRESH_COUNT}\n\
def runningCount : Nat := {RUNNING_COUNT}\n\
def matrixCount : Nat := {MATRIX_COUNT}\n\
def coefficientCount : Nat := {D}\n\
def carriedCount : Nat := {}\n\
def rowVariablesWhenNltM : Nat := 5\n\
def columnVariablesWhenNltM : Nat := 6\n\
def rowVariablesWhenNgtM : Nat := 7\n\
def columnVariablesWhenNgtM : Nat := 6\n\n\
def freshGammaExponents : List Nat := [0, 1]\n\
def normGammaExponents : List Nat := [2, 3, 4, 5]\n\
def carriedGammaExponents : List Nat :=\n  [{}]\n\n\
end Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaperRectangular.Generated\n",
        carried.len(),
        carried.join(", ")
    )
}

#[test]
fn paper_rectangular_lean_artifact_matches_rust() {
    assert_eq!(D, 54, "the bounded artifact is pinned to Phi81");
    let emitted = render();
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != emitted {
        let expected_path = format!("{path}.expected");
        let parent = std::path::Path::new(&expected_path)
            .parent()
            .expect("artifact path has a parent");
        std::fs::create_dir_all(parent).expect("create generated artifact directory");
        std::fs::write(&expected_path, emitted).expect("write expected artifact");
        panic!("rectangular-paper Lean artifact drifted; inspect and promote {expected_path}");
    }
}
