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
