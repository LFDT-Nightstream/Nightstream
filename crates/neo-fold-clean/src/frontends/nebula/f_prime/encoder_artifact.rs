//! Receipt-bound restoration of the Nebula F' assignment encoder.

use std::io::{Read, Write};
use std::sync::Arc;

use neo_math::F;

use super::{
    relation_config, remap_lane_ranges, selective_polynomial, NebulaFPrimeFieldArmShape, NebulaFPrimeRelation,
    NebulaFPrimeRelationError,
};
use crate::frontends::nebula::application::NebulaApplication;
use crate::frontends::nebula::plan::NebulaPlan;
use crate::frontends::r1cs_f_prime::lowering::{
    LowNormEncoderArtifactLimits, LowNormEncoderArtifactReceipt, VerifiedLowNormEncoderArtifact,
};
use crate::paper::relations::Structure;

const PHYSICAL_ARMS: usize = 2;

/// Trusted identity of one Nebula encoder artifact and its field-native arms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeEncoderArtifactReceipt {
    encoder: LowNormEncoderArtifactReceipt,
    physical_arm_shapes: [NebulaFPrimeFieldArmShape; PHYSICAL_ARMS],
}

impl NebulaFPrimeEncoderArtifactReceipt {
    /// Rebuild a receipt from trusted profile metadata.
    pub fn from_parts(
        encoder: LowNormEncoderArtifactReceipt,
        physical_arm_shapes: [NebulaFPrimeFieldArmShape; PHYSICAL_ARMS],
    ) -> Result<Self, NebulaFPrimeRelationError> {
        validate_arm_shapes(&encoder, &physical_arm_shapes)?;
        Ok(Self {
            encoder,
            physical_arm_shapes,
        })
    }

    pub fn encoder(&self) -> &LowNormEncoderArtifactReceipt {
        &self.encoder
    }

    pub fn physical_arm_shapes(&self) -> [NebulaFPrimeFieldArmShape; PHYSICAL_ARMS] {
        self.physical_arm_shapes
    }
}

/// Exact encoder state accepted under one trusted Nebula profile receipt.
pub struct VerifiedNebulaFPrimeEncoderArtifact {
    encoder: VerifiedLowNormEncoderArtifact,
    physical_arm_shapes: [NebulaFPrimeFieldArmShape; PHYSICAL_ARMS],
}

impl VerifiedNebulaFPrimeEncoderArtifact {
    /// Load one bounded encoder artifact and check its trusted receipt.
    pub fn read(
        reader: impl Read,
        receipt: &NebulaFPrimeEncoderArtifactReceipt,
        limits: LowNormEncoderArtifactLimits,
    ) -> Result<Self, NebulaFPrimeRelationError> {
        validate_arm_shapes(&receipt.encoder, &receipt.physical_arm_shapes)?;
        let encoder = crate::frontends::r1cs_f_prime::MultiBranchLowNormR1cs::read_verified_encoder_artifact(
            reader,
            &receipt.encoder,
            limits,
        )?;
        Ok(Self {
            encoder,
            physical_arm_shapes: receipt.physical_arm_shapes,
        })
    }

    pub fn receipt(&self) -> &LowNormEncoderArtifactReceipt {
        self.encoder.receipt()
    }
}

impl NebulaFPrimeRelation {
    pub(super) fn write_encoder_artifact(
        &self,
        writer: impl Write,
        matrix_digest: [F; 4],
    ) -> Result<NebulaFPrimeEncoderArtifactReceipt, NebulaFPrimeRelationError> {
        let encoder = self
            .relation
            .write_encoder_artifact(writer, matrix_digest)?;
        NebulaFPrimeEncoderArtifactReceipt::from_parts(encoder, [self.arm_shapes[0], self.arm_shapes[1]])
    }

    pub(super) fn from_verified_encoder_artifact(
        plan: &NebulaPlan,
        application: NebulaApplication,
        expected_shape: (usize, usize, usize),
        expected_matrix_digest: [u64; 4],
        artifact: VerifiedNebulaFPrimeEncoderArtifact,
    ) -> Result<Self, NebulaFPrimeRelationError> {
        application.validate_for(plan)?;
        if artifact.receipt().relation_shape()? != expected_shape {
            return Err(NebulaFPrimeRelationError::Geometry(
                "encoder and evaluator artifacts have different relation shapes".into(),
            ));
        }
        if artifact.receipt().matrix_digest() != expected_matrix_digest {
            return Err(NebulaFPrimeRelationError::Geometry(
                "encoder and evaluator artifacts have different matrix digests".into(),
            ));
        }

        let structure = Structure::new_verifier_artifact_header(
            expected_shape.0,
            expected_shape.1,
            expected_shape.2,
            selective_polynomial(),
        )
        .map_err(|error| {
            NebulaFPrimeRelationError::Geometry(format!("encoder artifact relation header is invalid: {error}"))
        })?;
        let physical_arm_shapes = artifact.physical_arm_shapes;
        let relation = artifact.encoder.into_relation(structure)?;
        if relation.selector_cols().len() != PHYSICAL_ARMS {
            return Err(NebulaFPrimeRelationError::Geometry(
                "encoder artifact does not contain the two physical Nebula arms".into(),
            ));
        }
        let source_public_input_lens = [
            physical_arm_shapes[0].public_columns,
            physical_arm_shapes[1].public_columns,
        ];
        let remapped_ranges = remap_lane_ranges(&relation, source_public_input_lens, plan.circuit())?;
        let mut config = relation_config(plan, Some(&application));
        config.scheme = config.scheme.remap_ranges(remapped_ranges)?;
        Ok(Self {
            relation: Arc::new(relation),
            config,
            application: Some(application),
            arm_shapes: [physical_arm_shapes[0], physical_arm_shapes[1], physical_arm_shapes[1]],
            width_audit: None,
            preprocessing_digest: None,
        })
    }
}

fn validate_arm_shapes(
    encoder: &LowNormEncoderArtifactReceipt,
    shapes: &[NebulaFPrimeFieldArmShape; PHYSICAL_ARMS],
) -> Result<(), NebulaFPrimeRelationError> {
    if encoder.arm_field_counts().len() != PHYSICAL_ARMS || encoder.arm_derived_counts().len() != PHYSICAL_ARMS {
        return Err(NebulaFPrimeRelationError::Geometry(
            "encoder receipt does not describe the two physical Nebula arms".into(),
        ));
    }
    for (arm, shape) in shapes.iter().enumerate() {
        if shape.rows == 0
            || shape.columns == 0
            || shape.public_columns == 0
            || shape.public_columns > shape.columns
            || u64::try_from(shape.columns).ok() != Some(encoder.arm_field_counts()[arm])
        {
            return Err(NebulaFPrimeRelationError::Geometry(
                "encoder receipt has an invalid field-native arm shape".into(),
            ));
        }
    }
    Ok(())
}
