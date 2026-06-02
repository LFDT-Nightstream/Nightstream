//! Poseidon2 Merkle-path verifier building block for terminal CE proofs.
//!
//! This does not prove the terminal CE relation by itself. It is a small
//! backend-neutral PCS/IOP primitive: bind an opened leaf digest to a public
//! root using Poseidon2-only hashing, both natively and inside `R1csBuilder`.

use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_math::F;
use thiserror::Error;

use crate::engine::r1cs_circuit::poseidon2::enforce_poseidon2_hash;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::paper::digest::pack_bytes_as_fields;

const NODE_DOMAIN: &[u8] = b"neo.fold.clean/terminal_ce_merkle_node/v1";

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TerminalCeMerkleError {
    #[error("terminal CE Merkle leaf index {index} does not fit path depth {depth}")]
    IndexTooLarge { index: usize, depth: usize },
}

pub fn terminal_ce_merkle_node(left: [F; 4], right: [F; 4]) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(NODE_DOMAIN);
    preimage.extend_from_slice(&left);
    preimage.extend_from_slice(&right);
    poseidon2_hash(&preimage)
}

pub fn terminal_ce_merkle_root_from_leaf(
    leaf: [F; 4],
    path: &[[F; 4]],
    index: usize,
) -> Result<[F; 4], TerminalCeMerkleError> {
    ensure_index_fits(index, path.len())?;
    let mut acc = leaf;
    for (level, sibling) in path.iter().copied().enumerate() {
        if ((index >> level) & 1) == 0 {
            acc = terminal_ce_merkle_node(acc, sibling);
        } else {
            acc = terminal_ce_merkle_node(sibling, acc);
        }
    }
    Ok(acc)
}

pub fn enforce_terminal_ce_merkle_root_from_leaf(
    builder: &mut R1csBuilder,
    leaf: [Var; 4],
    path: &[[Var; 4]],
    index: usize,
) -> Result<[Var; 4], TerminalCeMerkleError> {
    ensure_index_fits(index, path.len())?;
    let mut acc = leaf;
    for (level, sibling) in path.iter().copied().enumerate() {
        let (left, right) = if ((index >> level) & 1) == 0 {
            (acc, sibling)
        } else {
            (sibling, acc)
        };
        acc = enforce_terminal_ce_merkle_node(builder, left, right);
    }
    Ok(acc)
}

fn enforce_terminal_ce_merkle_node(builder: &mut R1csBuilder, left: [Var; 4], right: [Var; 4]) -> [Var; 4] {
    let mut preimage = Vec::new();
    for value in pack_bytes_as_fields(NODE_DOMAIN) {
        preimage.push(alloc_const(builder, value));
    }
    preimage.extend_from_slice(&left);
    preimage.extend_from_slice(&right);
    enforce_poseidon2_hash(builder, &preimage)
}

fn alloc_const(builder: &mut R1csBuilder, value: F) -> Var {
    let var = builder.alloc(value);
    builder.enforce_eq(&Lc::from_var(var), &Lc::from_const(value));
    var
}

fn ensure_index_fits(index: usize, depth: usize) -> Result<(), TerminalCeMerkleError> {
    if depth < usize::BITS as usize && index >= (1usize << depth) {
        return Err(TerminalCeMerkleError::IndexTooLarge { index, depth });
    }
    Ok(())
}
