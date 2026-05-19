//! Owns runtime rebinding of the packaged side-opening statement to published Nightstream surfaces.

use crate::public_proof::NightstreamStatement;
use crate::rv32im::{Rv32imProofStatement, SimpleKernelError};

use super::authoritative_side::{
    build_rv32im_side_surface_public_from_opening_summaries, verify_phase0_public_claims_against_surface,
};
use super::side_opening_relation::{
    validate_rv32im_side_opening_relation_statement, Rv32imSideOpeningRelationStatement,
};
use super::Rv32imSideOpeningPublic;

pub(super) fn verify_rv32im_side_opening_statement_against_runtime_surfaces(
    nightstream_statement: &NightstreamStatement,
    public_statement: &Rv32imProofStatement,
    public: &Rv32imSideOpeningPublic,
    opening_statement: &Rv32imSideOpeningRelationStatement,
) -> Result<(), SimpleKernelError> {
    validate_rv32im_side_opening_relation_statement(opening_statement)?;
    let public_statement_digest = public_statement.recompute_digest();
    if public_statement.digest != public_statement_digest {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream carried public statement digest does not match the public statement fields".into(),
        ));
    }
    if opening_statement.public_summary
        != super::side_opening_relation::Rv32imSideOpeningPublicStatementSummary::from_public_statement(
            public_statement,
        )
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream side opening statement does not match the carried public statement".into(),
        ));
    }
    let expected_surface = build_rv32im_side_surface_public_from_opening_summaries(
        &opening_statement.stage1,
        &opening_statement.stage2,
        &opening_statement.stage3,
    );
    verify_phase0_public_claims_against_surface(nightstream_statement.core_digest(), public, &expected_surface)
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
            "RV32IM Nightstream side opening statement surface does not match the carried side-opening public: {err}"
        ))
        })?;
    Ok(())
}
