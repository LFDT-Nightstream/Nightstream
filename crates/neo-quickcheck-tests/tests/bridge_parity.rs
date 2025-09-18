#![cfg(feature = "quickcheck")]
//! Parity tests between the bridge header encoder and the circuit's public_values()
//! Ensures byte-for-byte equality, including padding-before-digest and digest limbs.

#![allow(deprecated)]

use proptest::prelude::*;
#[allow(deprecated)]
use neo_ccs::{MEInstance, MEWitness};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use spartan2::traits::circuit::SpartanCircuit;
use neo_spartan_bridge::{encode_bridge_io_header, me_to_r1cs::MeCircuit};

fn f_from_u64s(xs: &[u64]) -> Vec<F> { 
    xs.iter().copied().map(F::from_u64).collect() 
}

proptest! {
    // Keep sizes small to be CI-friendly but cover interesting padding cases.
    #[test]
    fn header_encoding_matches_public_values(
        c_len in 0usize..6,
        y_len in 0usize..6,
        r_len in 0usize..6,
        base in 2u64..10,
        c_vals in prop::collection::vec(any::<u64>(), 0..6),
        y_vals in prop::collection::vec(any::<u64>(), 0..6),
        r_vals in prop::collection::vec(any::<u64>(), 0..6),
        digest in prop::array::uniform32(any::<u8>()),
    ) {
        // Truncate random pools to requested sizes (so padding is exercised)
        let c_coords = f_from_u64s(&c_vals[..c_len.min(c_vals.len())]);
        let y_outputs = f_from_u64s(&y_vals[..y_len.min(y_vals.len())]);
        let r_point  = f_from_u64s(&r_vals[..r_len.min(r_vals.len())]);

        // ME instance used by the bridge (public API)
        let me = MEInstance {
            c_step_coords: vec![], // Pattern B: Populated by IVC layer, not folding
            u_offset: 0,
            u_len: 0,
            c_coords,
            y_outputs,
            r_point,
            base_b: base,
            header_digest: digest,
        };

        // Witness is unused by public_values(); give a tiny, well-formed dummy.
        let wit = MEWitness {
            z_digits: vec![0i64],
            weight_vectors: vec![],
            ajtai_rows: Some(vec![]),
        };

        // Circuit's public values (this impl uses Goldilocks engine internally)
        let circuit = MeCircuit::new(me.clone(), wit, None, me.header_digest);
        let publics = SpartanCircuit::<spartan2::provider::GoldilocksMerkleMleEngine>::public_values(&circuit)
            .expect("public_values");

        // Convert engine scalars to u64 and back to bypass type issues
        let pv_bytes: Vec<u8> = {
            let mut out = Vec::with_capacity(publics.len() * 8);
            for x in &publics { 
                out.extend_from_slice(&x.to_canonical_u64().to_le_bytes()); 
            }
            out
        };
        let hdr_bytes = encode_bridge_io_header(&me);

        prop_assert_eq!(pv_bytes, hdr_bytes);
    }
}
