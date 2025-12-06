use neo_math::K;
use neo_memory::mle::build_chi_table;
use neo_memory::shout_oracle::ShoutReadCheckOracle;
use neo_reductions::sumcheck::{run_sumcheck_prover, verify_sumcheck_rounds};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

fn k(u: u64) -> K {
    K::from(Goldilocks::from_u64(u))
}

#[test]
fn shout_lookup_oracle_runs_sumcheck() {
    // addr bits=1, cycle bits=1
    let r_cycle = vec![k(9)];

    // Only cycle 1 has a lookup to addr 1.
    let ra_table = vec![
        K::ZERO,
        K::ZERO, // cycle 0
        K::ZERO,
        K::ONE, // cycle 1 -> addr 1
    ];
    let table_vals = vec![k(7), k(11)]; // Table[0]=7, Table[1]=11
    let has_lookup = vec![K::ZERO, K::ONE];

    let chi = build_chi_table(&r_cycle);
    let expected = table_vals[1] * chi[1]; // only cycle 1 contributes

    let mut oracle = ShoutReadCheckOracle::new(ra_table, table_vals.clone(), has_lookup.clone(), &r_cycle);
    let mut tr = Poseidon2Transcript::new(b"shout/oracle/test");
    let (rounds, _chals) = run_sumcheck_prover(&mut tr, &mut oracle, expected).expect("prover");

    let mut tr_v = Poseidon2Transcript::new(b"shout/oracle/test");
    let (_c, _final_sum, ok) = verify_sumcheck_rounds(&mut tr_v, 4, expected, &rounds);
    assert!(ok);
}
