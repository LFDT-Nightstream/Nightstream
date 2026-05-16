//! Owns relation-neutral Spartan backend aliases used by compressed IVC paths.

pub(crate) use spartan2::bellpepper::poseidon2::hash_packed_goldilocks_fields;
pub(crate) use spartan2::bellpepper::{r1cs::SpartanShape, shape_cs::ShapeCS};
pub(crate) use spartan2::provider::{goldi::F as SpartanF, GoldilocksP3MerkleMleEngine};
pub(crate) use spartan2::spartan::{SpartanProverKey, SpartanVerifierKey, R1CSSNARK};
pub(crate) use spartan2::traits::circuit::SpartanCircuit;
pub(crate) use spartan2::traits::snark::{DigestHelperTrait, R1CSSNARKTrait};
pub(crate) use spartan2::SplitR1CSShape;

pub(crate) type NeoFoldDeciderEngine = GoldilocksP3MerkleMleEngine;
pub(crate) type NeoFoldDeciderSnark = R1CSSNARK<NeoFoldDeciderEngine>;
pub(crate) type NeoFoldDeciderProverKey = SpartanProverKey<NeoFoldDeciderEngine>;
pub(crate) type NeoFoldDeciderVerifierKey = SpartanVerifierKey<NeoFoldDeciderEngine>;

pub(crate) type Rv32imDeciderEngine = NeoFoldDeciderEngine;
pub(crate) type Rv32imDeciderSnark = NeoFoldDeciderSnark;
pub(crate) type Rv32imDeciderProverKey = NeoFoldDeciderProverKey;
pub(crate) type Rv32imDeciderVerifierKey = NeoFoldDeciderVerifierKey;
