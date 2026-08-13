//! Selected-profile artifact boundary for the authoritative R1CS F′ compiler.

use std::io::Write;

use neo_math::D;
use p3_field::PrimeField64;

use super::ivc::R1csIvcPreprocessing;
use super::{is_canonical_selective_low_norm_polynomial, validate_plan, SelectiveStructureCensus};
use crate::paper::f_prime::r1cs::F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN;
use crate::relation_artifact::{
    field_digest_words, receipt_for, same_structure, validate_with_source, write_with_source, ApplicationWire,
    FixedPointRoundWire, RelationArtifactError, RelationArtifactReceipt, SourceWire,
};

pub const R1CS_F_PRIME_CONTRACT_ID: &str = "nightstream-superneo-v1";
pub const R1CS_F_PRIME_PROFILE_ID: &str = "nightstream-superneo-fprime-v1";
pub const R1CS_F_PRIME_COMPILER_ID: &str = "neo-fold-clean/r1cs-fprime-fixed-point-v1";

const APPLICATION_MATRIX_COUNT: usize = 13;
const JOINT_MATRIX_COUNT: usize = 14;
const POLYNOMIAL_DEGREE: u32 = 8;
const ROW_VARIABLES: u32 = 24;
const SECURITY_TARGET_BITS: u32 = 96;

impl R1csIvcPreprocessing {
    /// Write the complete selected-profile relation artifact. The exact
    /// matrix payload comes from this verifier-owned preprocessing object.
    pub fn write_relation_artifact(
        &self,
        writer: impl Write,
    ) -> Result<RelationArtifactReceipt, RelationArtifactError> {
        let source = selected_source(self)?;
        write_with_source(&self.prep, source, writer)
    }

    /// Return the complete selected-profile relation artifact.
    pub fn relation_artifact_json(&self) -> Result<Vec<u8>, RelationArtifactError> {
        let mut bytes = Vec::new();
        self.write_relation_artifact(&mut bytes)?;
        Ok(bytes)
    }

    /// Validate canonical bytes against this live compiler output and
    /// verifier key. Exact matrix equality, not a carried digest, decides.
    pub fn validate_relation_artifact_json(
        &self,
        bytes: &[u8],
    ) -> Result<RelationArtifactReceipt, RelationArtifactError> {
        let source = selected_source(self)?;
        validate_with_source(&self.prep, source, bytes)
    }

    /// Validate the selected profile and return its exact derived census
    /// without serializing the complete matrix payload.
    pub fn relation_artifact_receipt(&self) -> Result<RelationArtifactReceipt, RelationArtifactError> {
        selected_source(self)?;
        receipt_for(&self.prep)
    }
}

fn selected_source(prep: &R1csIvcPreprocessing) -> Result<SourceWire, RelationArtifactError> {
    validate_selected_profile(prep)?;
    let app = prep.app();
    let app_structure = app.to_structure();
    let plan_bytes = serde_json::to_vec(prep.plan())?;
    let plan_digest = neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash_packed_bytes(&plan_bytes)
        .map(|field| field.as_canonical_u64());
    let rounds = prep
        .relation()
        .compilation_audit()
        .rounds()
        .iter()
        .map(|round| {
            Ok(FixedPointRoundWire {
                input_rows: as_u64(round.input.rows, "fixed-point input rows")?,
                input_assignment_fields: as_u64(round.input.columns, "fixed-point input columns")?,
                output_rows: as_u64(round.output.rows, "fixed-point output rows")?,
                output_assignment_fields: as_u64(round.output.columns, "fixed-point output columns")?,
            })
        })
        .collect::<Result<Vec<_>, RelationArtifactError>>()?;

    Ok(SourceWire::R1csFPrimeFixedPoint {
        contract_id: R1CS_F_PRIME_CONTRACT_ID.to_owned(),
        profile_id: R1CS_F_PRIME_PROFILE_ID.to_owned(),
        compiler_id: R1CS_F_PRIME_COMPILER_ID.to_owned(),
        app: ApplicationWire {
            rows: as_u64(app.n(), "application rows")?,
            assignment_fields: as_u64(app.m(), "application columns")?,
            public_fields: as_u64(app.m_in(), "application public columns")?,
            provenance_only_structure_digest: field_digest_words(crate::paper::digest::structure_digest(
                &app_structure,
            )),
        },
        provenance_only_plan_digest: plan_digest,
        fixed_point_rounds: rounds,
    })
}

fn validate_selected_profile(prep: &R1csIvcPreprocessing) -> Result<(), RelationArtifactError> {
    prep.app()
        .validate_shape()
        .map_err(|error| RelationArtifactError::Profile(error.to_string()))?;
    validate_plan(prep.plan(), prep.app()).map_err(|error| RelationArtifactError::Profile(error.to_string()))?;
    prep.prep
        .validate_cached_structure()
        .map_err(|error| RelationArtifactError::Profile(error.to_string()))?;
    prep.prep
        .validate_verifier_key_binding()
        .map_err(|error| RelationArtifactError::Profile(error.to_string()))?;

    let structure = prep.prep.structure();
    let relation = prep.relation();
    if !same_structure(structure, relation.structure()) {
        return Err(RelationArtifactError::Profile(
            "compiled relation differs from verifier-owned preprocessing".to_owned(),
        ));
    }
    if prep.prep.public_input_len != Some(F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN)
        || relation.public_input_len() != F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN
    {
        return Err(RelationArtifactError::Profile(
            "public carrier must contain exactly 270 fields".to_owned(),
        ));
    }
    if structure.t() != APPLICATION_MATRIX_COUNT
        || structure.t() + 1 != JOINT_MATRIX_COUNT
        || structure.max_degree() != POLYNOMIAL_DEGREE
        || !is_canonical_selective_low_norm_polynomial(&structure.f)
    {
        return Err(RelationArtifactError::Profile(
            "relation must contain the exact 13-port degree-8 selective polynomial".to_owned(),
        ));
    }
    SelectiveStructureCensus::new(structure).map_err(|error| RelationArtifactError::Profile(error.to_string()))?;
    if structure.m % D != 0 {
        return Err(RelationArtifactError::Profile(
            "assignment width must contain complete Phi81 ring columns".to_owned(),
        ));
    }
    if !prep.prep.params.has_production_core() || prep.prep.params.lambda() < SECURITY_TARGET_BITS {
        return Err(RelationArtifactError::Profile(format!(
            "parameters must use the production core with at least {SECURITY_TARGET_BITS} effective bits"
        )));
    }
    let security = prep
        .prep
        .params
        .validate_ccs_shape(structure.n, structure.m, APPLICATION_MATRIX_COUNT, POLYNOMIAL_DEGREE)
        .map_err(|error| RelationArtifactError::Profile(error.to_string()))?;
    if security.cube_variables != ROW_VARIABLES {
        return Err(RelationArtifactError::Profile(format!(
            "selected relation must use the {ROW_VARIABLES}-variable row cube"
        )));
    }
    if !prep.prep.enforces_f_prime_recursive_link() || !prep.prep.enforces_terminal_induction() {
        return Err(RelationArtifactError::Profile(
            "verifier key does not own the complete folded F-prime induction relation".to_owned(),
        ));
    }

    let audit = relation.compilation_audit();
    let rounds = audit.rounds();
    let terminal = rounds
        .last()
        .ok_or_else(|| RelationArtifactError::Profile("fixed-point audit is empty".to_owned()))?;
    if terminal.output.rows != structure.n
        || terminal.output.columns != structure.m
        || terminal.output.public_input_len != F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN
    {
        return Err(RelationArtifactError::Profile(
            "fixed-point terminal header differs from the emitted relation".to_owned(),
        ));
    }
    if rounds.windows(2).any(|pair| {
        pair[0].output.rows != pair[1].input.rows
            || pair[0].output.columns != pair[1].input.columns
            || pair[0].output.public_input_len != pair[1].input.public_input_len
    }) {
        return Err(RelationArtifactError::Profile(
            "fixed-point audit contains a discontinuity".to_owned(),
        ));
    }
    Ok(())
}

fn as_u64(value: usize, label: &'static str) -> Result<u64, RelationArtifactError> {
    u64::try_from(value).map_err(|_| RelationArtifactError::Profile(format!("{label} does not fit u64")))
}
