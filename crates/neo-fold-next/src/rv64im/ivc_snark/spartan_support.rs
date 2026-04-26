//! Compression-owned direct Spartan imports and decider aliases.

pub(crate) use spartan2::bellpepper::poseidon2::hash_packed_goldilocks_fields;
pub(crate) use spartan2::bellpepper::{r1cs::SpartanShape, shape_cs::ShapeCS};
pub(crate) use spartan2::provider::{goldi::F as SpartanF, GoldilocksP3MerkleMleEngine};
pub(crate) use spartan2::spartan::{SpartanProverKey, SpartanVerifierKey, R1CSSNARK};
pub(crate) use spartan2::traits::circuit::SpartanCircuit;
pub(crate) use spartan2::traits::snark::{DigestHelperTrait, R1CSSNARKTrait};
pub(crate) use spartan2::SplitR1CSShape;

pub(crate) type Rv64imDeciderEngine = GoldilocksP3MerkleMleEngine;
pub(crate) type Rv64imDeciderSnark = R1CSSNARK<Rv64imDeciderEngine>;
pub(crate) type Rv64imDeciderProverKey = SpartanProverKey<Rv64imDeciderEngine>;
pub(crate) type Rv64imDeciderVerifierKey = SpartanVerifierKey<Rv64imDeciderEngine>;
