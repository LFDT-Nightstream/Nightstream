//! Owns runtime rebinding of the packaged side-opening statement to published Nightstream surfaces.

use crate::nightstream::NightstreamStatement;
use crate::rv64im::{Rv64imProofStatement, SimpleKernelError};

use super::authoritative_side::{
    build_rv64im_side_surface_public_from_opening_summaries, verify_phase0_public_claims_against_surface,
};
use super::side_opening_relation::{
    validate_rv64im_side_opening_relation_statement, Rv64imSideOpeningRelationStatement,
};
use super::Rv64imSideOpeningPublic;

pub(super) fn verify_rv64im_side_opening_statement_against_runtime_surfaces(
    nightstream_statement: &NightstreamStatement,
    public_statement: &Rv64imProofStatement,
    public: &Rv64imSideOpeningPublic,
    opening_statement: &Rv64imSideOpeningRelationStatement,
) -> Result<(), SimpleKernelError> {
    validate_rv64im_side_opening_relation_statement(opening_statement)?;
    if opening_statement.public_summary
        != super::side_opening_relation::Rv64imSideOpeningPublicStatementSummary::from_public_statement(
            public_statement,
        )
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream side opening statement does not match the carried public statement".into(),
        ));
    }
    let expected_surface = build_rv64im_side_surface_public_from_opening_summaries(
        &opening_statement.stage1,
        &opening_statement.stage2,
        &opening_statement.stage3,
    );
    verify_phase0_public_claims_against_surface(nightstream_statement.core_digest(), public, &expected_surface)
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
            "RV64IM Nightstream side opening statement surface does not match the carried side-opening public: {err}"
        ))
        })?;
    Ok(())
}
