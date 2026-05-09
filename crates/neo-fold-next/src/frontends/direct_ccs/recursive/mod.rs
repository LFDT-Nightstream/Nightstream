//! Owns the direct-CCS Construction-2 prover carrier.
//!
//! This type owns the non-VM direct CCS/R1CS append state. Standalone Spartan
//! compression proves the latest committed `F'` step and the final CE bundle
//! for the folded `F'` accumulator, matching the RV32IM two-part terminal
//! boundary without replaying historical chunks.

mod public_image;
mod snark;
mod state;
mod summary;

pub use public_image::DirectCcsRecursiveIvcPublicImage;
pub use snark::{
    verify_direct_ccs_recursive_ivc_snark_public, DirectCcsRecursiveIvcSnark, DirectCcsRecursiveIvcSnarkPerf,
    DirectCcsRecursiveIvcSnarkVerifierKey,
};
pub use state::DirectCcsRecursiveIvcState;
pub use summary::{
    DirectCcsFPrimeLowNormSourceR1csSummary, DirectCcsFPrimeLowNormSourceSummary, DirectCcsFPrimeVerifierBodySummary,
    DirectCcsFPrimeVerifierNifsSummary, DirectCcsRecursiveFPrimeSummary, DirectCcsRecursiveIvcSummary,
    DirectCcsRecursiveProofSummary, DirectCcsRecursiveSemanticSummary,
};

use std::time::Instant;

use neo_ajtai::{AjtaiSModule, Commitment};
use neo_ccs::{CeClaim, Mat};
use neo_math::{F, K};
use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};
use spartan2::traits::snark::DigestHelperTrait;

use super::f_prime::chain::{DirectCcsFPrimeChain, DirectCcsFPrimeEncoderStatus};
use super::public_image::DirectCcsIvcPublicImage;
use super::snark::{DirectCcsIvcSnark, DirectCcsIvcSnarkVerifierKey};
use super::state::{
    DirectCcsFPrimeSnarkError, DirectCcsFPrimeSnarkPerf, DirectCcsIvcState, DirectCcsProgram, DirectCcsStep,
};
use super::terminal::ce_bundle::{
    canonical_direct_ce_claims, direct_ce_bundle_witnesses, measure_direct_ce_bundle_relation,
    prove_direct_ce_bundle_relation, setup_direct_ce_bundle_relation, verify_direct_ce_bundle_relation,
    DirectCcsCeBundleProof, DirectCcsCeBundleVerifierKey,
};
use super::terminal::gadgets::{
    direct_accumulator_digest_from_claims, direct_accumulator_digest_from_claims_with_base,
};
use crate::ivc::SuperNeoIvcStepRelation;
use crate::prover::CommitmentMixers;
