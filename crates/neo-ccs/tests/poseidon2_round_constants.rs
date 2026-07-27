//! Pins `round_constants()` against the cached `PERM` instance: a permutation
//! rebuilt from the exported constants must be bit-identical to the canonical
//! one, so any drift in seed, draw order, or p3 internals fails here.

use neo_ccs::crypto::poseidon2_goldilocks as p2;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::Permutation;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const LEAN_ARTIFACT_REL_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/Poseidon2Goldilocks/Generated/RoundConstants.lean";

fn render_rows(rows: &[[u64; p2::WIDTH]]) -> String {
    rows.iter()
        .map(|row| {
            let entries = row
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{entries}]")
        })
        .collect::<Vec<_>>()
        .join(",\n   ")
}

fn render_values(values: &[u64]) -> String {
    values
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn emit_lean_round_constants() -> String {
    let rc = p2::round_constants();
    let mut out = String::new();
    out.push_str("import Init\n\n");
    out.push_str("/-!\nGENERATED FILE — do not edit by hand.\n\n");
    out.push_str("Canonical-u64 Poseidon2 constants exported by\n");
    out.push_str("`neo_ccs::crypto::poseidon2_goldilocks::round_constants()`.\n");
    out.push_str("Regenerated and drift-checked by\n");
    out.push_str("`cargo test -p neo-ccs --release --test poseidon2_round_constants`.\n\n");
    out.push_str("This is exact implementation-conformance evidence, not semantic authority.\n");
    out.push_str("The seed-to-constant generator and this importer remain a published TCB boundary.\n");
    out.push_str("-/\n\n");
    out.push_str("namespace Nightstream.Implementation.R1CS.Artifacts.Poseidon2Goldilocks.RoundConstants\n\n");
    out.push_str("def initial : List (List Nat) :=\n  [");
    out.push_str(&render_rows(&rc.initial));
    out.push_str("]\n\n");
    out.push_str("def internal : List Nat :=\n  [");
    out.push_str(&render_values(&rc.internal));
    out.push_str("]\n\n");
    out.push_str("def terminal : List (List Nat) :=\n  [");
    out.push_str(&render_rows(&rc.terminal));
    out.push_str("]\n\n");
    out.push_str("def internalDiagonal : List Nat :=\n  [");
    out.push_str(&render_values(&rc.diag));
    out.push_str("]\n\n");
    out.push_str("end Nightstream.Implementation.R1CS.Artifacts.Poseidon2Goldilocks.RoundConstants\n");
    out
}

#[test]
fn exported_round_constants_rebuild_the_canonical_permutation() {
    let rc = p2::round_constants();
    assert_eq!(rc.initial.len(), rc.terminal.len(), "external rounds must be symmetric");
    assert!(!rc.internal.is_empty());

    let to_row = |r: &[u64; p2::WIDTH]| r.map(Goldilocks::from_u64);
    let external = ExternalLayerConstants::new(
        rc.initial.iter().map(to_row).collect(),
        rc.terminal.iter().map(to_row).collect(),
    );
    let internal: Vec<Goldilocks> = rc
        .internal
        .iter()
        .map(|&c| Goldilocks::from_u64(c))
        .collect();
    let rebuilt = Poseidon2Goldilocks::<{ p2::WIDTH }>::new(&external, &internal);

    let mut rng = StdRng::seed_from_u64(0x7032_5f72_635f_7631);
    for _ in 0..64 {
        let state: [Goldilocks; p2::WIDTH] = core::array::from_fn(|_| Goldilocks::from_u64(rng.random::<u64>()));
        let canonical = p2::permute_state(state);
        let ours = rebuilt.permute(state);
        let canon_u64 = canonical.map(|x| x.as_canonical_u64());
        let ours_u64 = ours.map(|x| x.as_canonical_u64());
        assert_eq!(canon_u64, ours_u64, "rebuilt permutation diverged");
    }
}

#[test]
fn exported_round_constants_match_lean_artifact() {
    let emitted = emit_lean_round_constants();
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), LEAN_ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != emitted {
        let expected_path = format!("{path}.expected");
        let parent = std::path::Path::new(&expected_path)
            .parent()
            .expect("artifact path has a parent");
        std::fs::create_dir_all(parent).expect("create generated artifact directory");
        std::fs::write(&expected_path, &emitted).expect("write .expected artifact");
        panic!(
            "Poseidon2 round-constant Lean artifact drifted. Wrote {expected_path}; inspect it and promote it deliberately."
        );
    }
}
