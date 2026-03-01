//! Shout semantic checks for explicit tables.
//!
//! This test builds explicit Shout tables plus a corresponding Ajtai-encoded `LutWitness`, then
//! runs `neo_memory::shout::check_shout_semantics` to get precise, step-indexed failures (instead
//! of only seeing a generic sumcheck mismatch later in the pipeline).
//!
//! Tables exercised here:
//! - Byte identity (`k=256`, `n_side=256`, `ell=8`): `table[addr] = addr`.
//!   Useful for byte range-checks / byte decomposition.
//! - Nibble square (`k=16`, `n_side=16`, `ell=4`): `table[addr] = addr^2`.
//!   Useful for 4-bit range-checks / nibble decomposition with a non-trivial mapping.
//!
//! We also include a small addr-pre (address-domain) sumcheck test that uses *two lanes* doing a
//! lookup to the *same* address at the *same* time index, to corroborate that Shout lookups are
//! multiset lookups (no “distinct key” requirement across lanes).
//!
//! Running:
//! - Run this integration test binary: `cargo test -p neo-memory --test shout_byte_decomp_semantics --release`
//! - Or filter by name (this test is prefixed accordingly): `cargo test -p neo-memory shout_byte_decomp_semantics --release`

use neo_math::{KExtensions, F};
use neo_memory::ajtai::encode_vector_for_ccs_m;
use neo_memory::shout::check_shout_semantics;
use neo_memory::sparse_time::SparseIdxVec;
use neo_memory::twist_oracle::AddressLookupOracle;
use neo_memory::witness::{LutInstance, LutWitness};
use neo_params::NeoParams;
use neo_reductions::sumcheck::{run_batched_sumcheck_prover, verify_batched_sumcheck_rounds, BatchedClaim};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn build_single_lane_explicit_lut_witness(
    params: &NeoParams,
    n_side: usize,
    ell: usize,
    table: Vec<F>,
    addrs: &[u64],
) -> (LutInstance<(), F>, LutWitness<F>, Vec<F>) {
    assert!(n_side.is_power_of_two(), "n_side must be power-of-two");
    assert_eq!(n_side.trailing_zeros() as usize, ell, "ell must be log2(n_side)");

    let steps = addrs.len();
    for &addr in addrs {
        assert!(
            (addr as usize) < n_side,
            "addr out of range: addr={addr}, n_side={n_side}"
        );
    }

    let inst = LutInstance::<(), F> {
        table_id: 0,
        comms: Vec::new(),
        k: n_side,
        d: 1,
        n_side,
        steps,
        lanes: 1,
        ell,
        table_spec: None,
        table: table.clone(),
        table_digest: None,
        addr_group: None,
        selector_group: None,
    };

    // Layout: [addr_bits(ell), has_lookup, val].
    let mut mats = Vec::with_capacity(ell + 2);

    // addr_bits are little-endian.
    for bit in 0..ell {
        let col: Vec<F> = addrs
            .iter()
            .map(|&addr| if ((addr >> bit) & 1) == 1 { F::ONE } else { F::ZERO })
            .collect();
        mats.push(encode_vector_for_ccs_m(params, steps, &col).expect("pack addr bits"));
    }

    let has_lookup: Vec<F> = vec![F::ONE; steps];
    mats.push(encode_vector_for_ccs_m(params, steps, &has_lookup).expect("pack has_lookup"));

    let expected_vals: Vec<F> = addrs.iter().map(|&addr| table[addr as usize]).collect();
    mats.push(encode_vector_for_ccs_m(params, steps, &expected_vals).expect("pack expected_vals"));

    (inst, LutWitness { mats }, expected_vals)
}

#[test]
fn shout_byte_decomp_semantics_explicit_tables_byte_identity_and_nibble_square() {
    let params = NeoParams::goldilocks_127();

    // Byte identity table: for any byte `b`, lookup(b) = b.
    // Repeated bytes are allowed: Shout does not require distinct keys.
    let byte_identity: Vec<F> = (0u64..256).map(F::from_u64).collect();
    let bytes: Vec<u8> = vec![0x00, 0xAB, 0xAB, 0x00, 0xFF, 0xAB, 0xFF, 0x00];
    let addrs: Vec<u64> = bytes.iter().map(|&b| b as u64).collect();
    let (inst, wit, expected_vals) =
        build_single_lane_explicit_lut_witness(&params, /*n_side=*/ 256, /*ell=*/ 8, byte_identity, &addrs);
    check_shout_semantics(&params, &inst, &wit, &expected_vals).expect("byte identity semantics");

    // Nibble square table: for any 4-bit `x`, lookup(x) = x^2 (in the base field).
    let nibble_square: Vec<F> = (0u64..16).map(|x| F::from_u64(x * x)).collect();
    let addrs: Vec<u64> = vec![0, 1, 2, 2, 3, 15, 15, 4, 0];
    let (inst, wit, expected_vals) =
        build_single_lane_explicit_lut_witness(&params, /*n_side=*/ 16, /*ell=*/ 4, nibble_square, &addrs);
    check_shout_semantics(&params, &inst, &wit, &expected_vals).expect("nibble square semantics");
}

fn build_single_sparse_lookup_lane(
    pow2_cycle: usize,
    ell_addr: usize,
    t: usize,
    addr: u64,
) -> (Vec<SparseIdxVec<neo_math::K>>, SparseIdxVec<neo_math::K>) {
    assert!(pow2_cycle.is_power_of_two());
    assert!(t < pow2_cycle);
    assert!(ell_addr <= 64, "addr must fit in u64 bits");

    let mut addr_bits = Vec::with_capacity(ell_addr);
    for bit in 0..ell_addr {
        let bit_is_one = ((addr >> bit) & 1) == 1;
        let entries = if bit_is_one {
            vec![(t, neo_math::K::ONE)]
        } else {
            Vec::new()
        };
        addr_bits.push(SparseIdxVec::from_entries(pow2_cycle, entries));
    }
    let has_lookup = SparseIdxVec::from_entries(pow2_cycle, vec![(t, neo_math::K::ONE)]);
    (addr_bits, has_lookup)
}

#[test]
fn shout_addr_pre_allows_duplicate_addresses_across_lanes_in_same_step() {
    // Address domain: 8-bit table.
    let ell_addr = 8usize;
    let table: Vec<neo_math::K> = (0u64..256)
        .map(|x| neo_math::K::from(F::from_u64(x)))
        .collect();

    // Time domain: one chunk with 2^ell_cycle rows.
    let r_cycle: Vec<neo_math::K> = vec![
        neo_math::K::from(F::from_u64(3)),
        neo_math::K::from(F::from_u64(5)),
        neo_math::K::from(F::from_u64(7)),
    ];
    let pow2_cycle = 1usize << r_cycle.len();

    // Two lanes, both doing a lookup to the same byte at the same time row.
    let t = 0usize;
    let addr = 1u64;
    let (lane0_addr_bits, lane0_has_lookup) = build_single_sparse_lookup_lane(pow2_cycle, ell_addr, t, addr);
    let (lane1_addr_bits, lane1_has_lookup) = build_single_sparse_lookup_lane(pow2_cycle, ell_addr, t, addr);

    let (mut lane0_oracle, lane0_sum) =
        AddressLookupOracle::new(&lane0_addr_bits, &lane0_has_lookup, &table, &r_cycle, ell_addr);
    let (mut lane1_oracle, lane1_sum) =
        AddressLookupOracle::new(&lane1_addr_bits, &lane1_has_lookup, &table, &r_cycle, ell_addr);

    // If “distinct keys across lanes” were required, this exact pattern would be forbidden.
    // We instead expect both claims to be well-formed (and identical).
    assert_eq!(lane0_sum, lane1_sum);
    assert_ne!(lane0_sum, neo_math::K::ZERO);

    let mut tr_p = Poseidon2Transcript::new(b"test/shout/addr_pre/dup_addr_across_lanes");
    tr_p.append_message(b"ell_addr", &(ell_addr as u64).to_le_bytes());
    tr_p.append_fields(b"claimed_sum/lane0", &lane0_sum.as_coeffs());
    tr_p.append_fields(b"claimed_sum/lane1", &lane1_sum.as_coeffs());
    let mut tr_v = tr_p.clone();

    let mut claims = vec![
        BatchedClaim {
            oracle: &mut lane0_oracle,
            claimed_sum: lane0_sum,
            label: b"shout/addr_pre",
        },
        BatchedClaim {
            oracle: &mut lane1_oracle,
            claimed_sum: lane1_sum,
            label: b"shout/addr_pre",
        },
    ];

    let (r_addr_p, per_claim_results) = run_batched_sumcheck_prover(&mut tr_p, claims.as_mut_slice())
        .expect("batched addr-pre sumcheck prover should succeed");
    let per_claim_rounds: Vec<Vec<Vec<neo_math::K>>> = per_claim_results
        .iter()
        .map(|r| r.round_polys.clone())
        .collect();

    let claimed_sums = vec![lane0_sum, lane1_sum];
    let labels: Vec<&[u8]> = vec![b"shout/addr_pre".as_slice(); 2];
    let degree_bounds: Vec<usize> = vec![2; 2];
    let (r_addr_v, finals, ok) =
        verify_batched_sumcheck_rounds(&mut tr_v, &per_claim_rounds, &claimed_sums, &labels, &degree_bounds);
    assert!(ok, "batched addr-pre sumcheck verifier should accept");
    assert_eq!(r_addr_v, r_addr_p);
    assert_eq!(finals.len(), 2);
    assert_eq!(finals[0], finals[1]);
}
