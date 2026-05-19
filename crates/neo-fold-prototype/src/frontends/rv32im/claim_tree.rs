//! Owns Poseidon2 claim-digest tree helpers for RV32IM running-instance memory experiments.

use neo_ajtai::Commitment;
use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_ccs::CeClaim;
use neo_math::{F, K};
use neo_reductions::engines::utils::me_digest_poseidon_into;
use p3_field::PrimeCharacteristicRing;

use crate::finalize::{digest32_as_fields, digest_fields_as_digest32};
use crate::rv32im::SimpleKernelError;

pub type Rv32imClaimDigestFields = [F; 4];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imClaimMerkleOpening {
    leaf_index: usize,
    siblings: Vec<Rv32imClaimDigestFields>,
}

impl Rv32imClaimMerkleOpening {
    pub fn leaf_index(&self) -> usize {
        self.leaf_index
    }

    pub fn siblings(&self) -> &[Rv32imClaimDigestFields] {
        &self.siblings
    }
}

fn claim_tree_zero_leaf() -> Rv32imClaimDigestFields {
    let digest = digest_fields_as_digest32(poseidon2_hash(&[
        F::from_u64(0x6374_7a65_726f_0001),
        F::ZERO,
        F::ZERO,
        F::ZERO,
    ]));
    digest32_as_fields(digest)
}

fn hash_claim_tree_node(left: Rv32imClaimDigestFields, right: Rv32imClaimDigestFields) -> Rv32imClaimDigestFields {
    let mut preimage = Vec::with_capacity(1 + left.len() + right.len());
    preimage.push(F::from_u64(0x6374_6e6f_6465_0001));
    preimage.extend(left);
    preimage.extend(right);
    digest32_as_fields(digest_fields_as_digest32(poseidon2_hash(&preimage)))
}

pub fn build_rv32im_claim_digests(claims: &[CeClaim<Commitment, F, K>]) -> Vec<Rv32imClaimDigestFields> {
    let mut scratch = Vec::with_capacity(2048);
    claims
        .iter()
        .map(|claim| me_digest_poseidon_into(&mut scratch, claim))
        .collect()
}

pub fn rv32im_claim_tree_root_from_digests(digests: &[Rv32imClaimDigestFields]) -> Rv32imClaimDigestFields {
    if digests.is_empty() {
        return claim_tree_zero_leaf();
    }

    let leaf_count = digests.len().next_power_of_two();
    let mut level = Vec::with_capacity(leaf_count);
    level.extend_from_slice(digests);
    level.resize(leaf_count, claim_tree_zero_leaf());

    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks_exact(2) {
            next.push(hash_claim_tree_node(pair[0], pair[1]));
        }
        level = next;
    }

    level[0]
}

pub fn rv32im_claim_tree_root_from_claims(claims: &[CeClaim<Commitment, F, K>]) -> Rv32imClaimDigestFields {
    rv32im_claim_tree_root_from_digests(&build_rv32im_claim_digests(claims))
}

pub fn rv32im_claim_tree_opening_from_digests(
    digests: &[Rv32imClaimDigestFields],
    leaf_index: usize,
) -> Result<Rv32imClaimMerkleOpening, SimpleKernelError> {
    if leaf_index >= digests.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM claim-tree opening index {leaf_index} is out of range for {} leaves",
            digests.len()
        )));
    }
    if digests.is_empty() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM claim-tree opening requires at least one leaf".into(),
        ));
    }

    let leaf_count = digests.len().next_power_of_two();
    let mut level = Vec::with_capacity(leaf_count);
    level.extend_from_slice(digests);
    level.resize(leaf_count, claim_tree_zero_leaf());

    let mut index = leaf_index;
    let mut siblings = Vec::with_capacity(leaf_count.trailing_zeros() as usize);
    while level.len() > 1 {
        let sibling_index = index ^ 1;
        siblings.push(level[sibling_index]);

        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks_exact(2) {
            next.push(hash_claim_tree_node(pair[0], pair[1]));
        }
        level = next;
        index /= 2;
    }

    Ok(Rv32imClaimMerkleOpening { leaf_index, siblings })
}

pub fn verify_rv32im_claim_tree_opening(
    root: Rv32imClaimDigestFields,
    leaf: Rv32imClaimDigestFields,
    opening: &Rv32imClaimMerkleOpening,
) -> bool {
    let mut acc = leaf;
    let mut index = opening.leaf_index;
    for sibling in &opening.siblings {
        acc = if index & 1 == 0 {
            hash_claim_tree_node(acc, *sibling)
        } else {
            hash_claim_tree_node(*sibling, acc)
        };
        index >>= 1;
    }
    acc == root
}
