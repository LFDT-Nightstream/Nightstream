//! Pins `round_constants()` against the cached permutation and the selected
//! F-prime Lean specification.

use neo_ccs::crypto::poseidon2_goldilocks as p2;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::Permutation;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn lean_numbers_after(source: &str, declaration: &str) -> Vec<u64> {
    let body = source
        .split_once(declaration)
        .expect("selected Lean Poseidon2 constant declaration")
        .1
        .split_once("\n\n")
        .expect("next Lean declaration")
        .0;
    body.split(|c: char| !(c.is_ascii_hexdigit() || c == 'x'))
        .filter(|word| !word.is_empty())
        .map(|word| {
            if let Some(hex) = word.strip_prefix("0x") {
                u64::from_str_radix(hex, 16).expect("Lean hexadecimal constant")
            } else {
                word.parse().expect("Lean decimal constant")
            }
        })
        .collect()
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
    // Plonky3's AArch64 fused constructor borrows; other targets take ownership.
    #[cfg(target_arch = "aarch64")]
    let rebuilt = Poseidon2Goldilocks::<{ p2::WIDTH }>::new(&external, &internal);
    #[cfg(not(target_arch = "aarch64"))]
    let rebuilt = Poseidon2Goldilocks::<{ p2::WIDTH }>::new(external, internal);

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
fn exported_round_constants_match_fprime_lean_spec() {
    let source = include_str!("../../../formal/nightstream-fprime/NightstreamFPrime/Spec/Poseidon2.lean");
    let rust = p2::round_constants();
    let initial = lean_numbers_after(source, "def initialConstants : List (List Nat) :=");
    let internal = lean_numbers_after(source, "def internalConstants : List Nat :=");
    let terminal = lean_numbers_after(source, "def terminalConstants : List (List Nat) :=");
    let diagonal = lean_numbers_after(source, "def internalDiagonal : List Nat :=");
    assert_eq!(
        (rust.initial.len(), rust.internal.len(), rust.terminal.len()),
        (4, 22, 4)
    );
    assert_eq!(initial, rust.initial.into_iter().flatten().collect::<Vec<_>>());
    assert_eq!(internal, rust.internal);
    assert_eq!(terminal, rust.terminal.into_iter().flatten().collect::<Vec<_>>());
    assert_eq!(diagonal, rust.diag);
}
