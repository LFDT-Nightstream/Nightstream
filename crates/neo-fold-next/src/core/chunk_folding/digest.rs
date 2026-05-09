use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_math::{F, K};
use neo_reductions::engines::utils::{self, me_digest_poseidon};
use neo_transcript::{Poseidon2Transcript, Transcript};
use rayon::prelude::*;

pub(crate) fn chunk_relation_digest(
    ccs_outputs: &[CeClaim<Commitment, F, K>],
    parent: &CeClaim<Commitment, F, K>,
    children: &[CeClaim<Commitment, F, K>],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/chunk_relation_digest");
    tr.append_u64s(
        b"neo.fold.next/chunk_relation_digest/counts",
        &[ccs_outputs.len() as u64, children.len() as u64],
    );
    for digest in claim_digests(ccs_outputs) {
        tr.append_fields(b"neo.fold.next/chunk_relation_digest/ccs_output", &digest);
    }
    tr.append_fields(
        b"neo.fold.next/chunk_relation_digest/rlc_parent",
        &me_digest_poseidon(parent),
    );
    for claim in children {
        tr.append_fields(
            b"neo.fold.next/chunk_relation_digest/dec_child",
            &me_digest_poseidon(claim),
        );
    }
    tr.digest32()
}

pub(crate) fn claim_digests(claims: &[CeClaim<Commitment, F, K>]) -> Vec<[F; 4]> {
    #[cfg(not(target_arch = "wasm32"))]
    let allow_parallel = rayon::current_num_threads() > 1 && rayon::current_thread_index().is_none();
    #[cfg(target_arch = "wasm32")]
    let _allow_parallel = false;

    #[cfg(not(target_arch = "wasm32"))]
    if allow_parallel && claims.len() >= 8 {
        return claims.par_iter().map(me_digest_poseidon).collect();
    }

    let mut digests = Vec::with_capacity(claims.len());
    let mut scratch = Vec::<F>::with_capacity(2048);
    for claim in claims {
        digests.push(utils::me_digest_poseidon_into(&mut scratch, claim));
    }
    digests
}
