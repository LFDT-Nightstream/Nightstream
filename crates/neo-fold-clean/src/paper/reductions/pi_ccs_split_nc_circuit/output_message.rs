//! R1CS source-column decoder for the `Pi_CCS` output digest preimage.
//!
//! Owns: shape validation, construction of the exact pre-SIS wire order, and
//! a one-to-one record from every field index to its typed path and source
//! column.
//!
//! Does not own: output truth, padding-zero constraints, SIS/Poseidon2,
//! transcript placement, or row-removal authority.
//!
//! Emits constraints: only equality rows for verifier-owned domain/shape
//! constants. Output limbs reuse wires already accepted by FE/NC phases.
//!
//! Authority boundary: a binding proves which R1CS column supplies a digest
//! field. It does not close the delayed `y_zcol` source-authority theorem.
//!
//! | Field family | Mathematical value | R1CS source | Lean owner |
//! |---|---|---|---|
//! | outer header | domain fields, source count | verifier constants | `ActiveSemantics.serialize` |
//! | source header | domain fields, matrix count | verifier constants | `ActiveSemantics.encodeSource` |
//! | `y_ring` width | active Phi81 lane count | verifier constant | `Encoding.encodeKVector` |
//! | `y_ring` limbs | `message.yRing[source][matrix][lane].(c0,c1)` | existing output wire | `ActiveSemantics.encodeSourcePayload` |
//! | `y_zcol` width | active Phi81 lane count | verifier constant | `Encoding.encodeKVector` |
//! | `y_zcol` limbs | `message.yZcol[source][lane].(c0,c1)` | existing output wire | `ActiveSemantics.encodeSourcePayload` |

mod sis_ownership;

pub use sis_ownership::{audit_pi_ccs_output_sis, PiCcsOutputSisAudit, PiCcsOutputSisOwnerAudit};

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::{alloc_constant_var, extend_packed_bytes_as_fields_wires, stage, Error};
use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::{R1csBuilder, Var};
use crate::paper::reductions::pi_ccs_output_message::{
    FieldPath, KLimb, Profile, R1csInputOwner, OUTPUTS_DOMAIN, OUTPUTS_DOMAIN_FIELD_COUNT, OUTPUT_MESSAGE_DOMAIN,
    OUTPUT_MESSAGE_DOMAIN_FIELD_COUNT,
};

/// Output wires consumed by the active pre-SIS projection.
pub struct PiCcsOutputMessageDigestInputs<'a> {
    pub y_ring: &'a [Vec<KVar>],
    pub y_zcol: &'a [KVar],
}

/// Exact ownership record for one pre-SIS field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PiCcsOutputFieldBinding {
    index: usize,
    path: FieldPath,
    wire: Var,
}

impl PiCcsOutputFieldBinding {
    pub fn index(self) -> usize {
        self.index
    }

    pub fn path(self) -> FieldPath {
        self.path
    }

    pub fn wire(self) -> Var {
        self.wire
    }

    pub fn source_column(self) -> usize {
        self.wire.col()
    }

    pub fn r1cs_input_owner(self) -> R1csInputOwner {
        self.path.r1cs_input_owner()
    }
}

/// Checked field stream handed to SIS compression.
pub struct PiCcsOutputsPreimage {
    profile: Profile,
    fields: Vec<PiCcsOutputFieldBinding>,
}

impl PiCcsOutputsPreimage {
    pub fn profile(&self) -> Profile {
        self.profile
    }

    pub fn fields(&self) -> &[PiCcsOutputFieldBinding] {
        &self.fields
    }

    pub fn wires(&self) -> Vec<Var> {
        self.fields.iter().map(|field| field.wire).collect()
    }
}

struct PreimageBuilder {
    profile: Profile,
    fields: Vec<PiCcsOutputFieldBinding>,
}

impl PreimageBuilder {
    fn new(profile: Profile) -> Self {
        Self {
            profile,
            fields: Vec::with_capacity(profile.field_count()),
        }
    }

    fn push(&mut self, path: FieldPath, wire: Var) -> Result<(), Error> {
        let index = self.fields.len();
        let expected = self.profile.decode(index).ok_or_else(|| {
            Error::Shape(format!(
                "Pi_CCS output preimage emitted field {index} beyond exact profile length {}",
                self.profile.field_count()
            ))
        })?;
        if path != expected {
            return Err(Error::Shape(format!(
                "Pi_CCS output preimage field {index} has path {path:?}, expected {expected:?}"
            )));
        }
        self.fields
            .push(PiCcsOutputFieldBinding { index, path, wire });
        Ok(())
    }

    fn finish(self) -> Result<PiCcsOutputsPreimage, Error> {
        if self.fields.len() != self.profile.field_count() {
            return Err(Error::Shape(format!(
                "Pi_CCS output preimage emitted {} fields, expected {}",
                self.fields.len(),
                self.profile.field_count()
            )));
        }
        Ok(PiCcsOutputsPreimage {
            profile: self.profile,
            fields: self.fields,
        })
    }
}

/// Decode the complete typed output product into the exact field stream used
/// by the SIS digest. `profile` is verifier-owned; no shape is inferred from a
/// prover message at this boundary.
pub fn encode_pi_ccs_outputs_preimage(
    builder: &mut R1csBuilder,
    profile: Profile,
    inputs: &[PiCcsOutputMessageDigestInputs<'_>],
) -> Result<PiCcsOutputsPreimage, Error> {
    validate_inputs(profile, inputs)?;
    let mut output = PreimageBuilder::new(profile);

    builder.begin_encoding_stage(stage::OUTPUT_MESSAGE_PREIMAGE);
    builder.begin_encoding_stage(stage::OUTPUT_MESSAGE_PREIMAGE_OUTER_HEADER);
    let mut outer_domain = Vec::with_capacity(OUTPUTS_DOMAIN_FIELD_COUNT);
    extend_packed_bytes_as_fields_wires(builder, &mut outer_domain, OUTPUTS_DOMAIN);
    debug_assert_eq!(outer_domain.len(), OUTPUTS_DOMAIN_FIELD_COUNT);
    for (field, wire) in outer_domain.into_iter().enumerate() {
        output.push(FieldPath::OutputsDomain { field }, wire)?;
    }
    output.push(
        FieldPath::SourceCount,
        alloc_constant_var(builder, F::from_u64(profile.source_count() as u64)),
    )?;

    for (source, input) in inputs.iter().enumerate() {
        builder.begin_encoding_stage(stage::OUTPUT_MESSAGE_PREIMAGE_SOURCE_HEADERS);
        let mut source_domain = Vec::with_capacity(OUTPUT_MESSAGE_DOMAIN_FIELD_COUNT);
        extend_packed_bytes_as_fields_wires(builder, &mut source_domain, OUTPUT_MESSAGE_DOMAIN);
        debug_assert_eq!(source_domain.len(), OUTPUT_MESSAGE_DOMAIN_FIELD_COUNT);
        for (field, wire) in source_domain.into_iter().enumerate() {
            output.push(FieldPath::SourceDomain { source, field }, wire)?;
        }
        output.push(
            FieldPath::MatrixCount { source },
            alloc_constant_var(builder, F::from_u64(profile.matrix_count() as u64)),
        )?;

        for (matrix, row) in input.y_ring.iter().enumerate() {
            builder.begin_encoding_stage(stage::OUTPUT_MESSAGE_PREIMAGE_Y_RING);
            output.push(
                FieldPath::YRingWidth { source, matrix },
                alloc_constant_var(builder, F::from_u64(profile.lane_count() as u64)),
            )?;
            for (lane, value) in row[..profile.lane_count()].iter().enumerate() {
                push_k_limbs(&mut output, source, Some(matrix), lane, *value)?;
            }
        }

        builder.begin_encoding_stage(stage::OUTPUT_MESSAGE_PREIMAGE_Y_ZCOL);
        output.push(
            FieldPath::YZcolWidth { source },
            alloc_constant_var(builder, F::from_u64(profile.lane_count() as u64)),
        )?;
        for (lane, value) in input.y_zcol[..profile.lane_count()].iter().enumerate() {
            push_k_limbs(&mut output, source, None, lane, *value)?;
        }
    }

    output.finish()
}

fn validate_inputs(profile: Profile, inputs: &[PiCcsOutputMessageDigestInputs<'_>]) -> Result<(), Error> {
    if inputs.len() != profile.source_count() {
        return Err(Error::Shape(format!(
            "Pi_CCS output source count {} does not match verifier profile {}",
            inputs.len(),
            profile.source_count()
        )));
    }
    for (source, input) in inputs.iter().enumerate() {
        if input.y_ring.len() != profile.matrix_count() {
            return Err(Error::Shape(format!(
                "Pi_CCS output source {source} has {} y_ring matrices, expected {}",
                input.y_ring.len(),
                profile.matrix_count()
            )));
        }
        for (matrix, row) in input.y_ring.iter().enumerate() {
            if row.len() < profile.lane_count() {
                return Err(Error::Shape(format!(
                    "Pi_CCS output source {source} y_ring[{matrix}] has {} lanes, expected at least {}",
                    row.len(),
                    profile.lane_count()
                )));
            }
        }
        if input.y_zcol.len() < profile.lane_count() {
            return Err(Error::Shape(format!(
                "Pi_CCS output source {source} y_zcol has {} lanes, expected at least {}",
                input.y_zcol.len(),
                profile.lane_count()
            )));
        }
    }
    Ok(())
}

fn push_k_limbs(
    output: &mut PreimageBuilder,
    source: usize,
    matrix: Option<usize>,
    lane: usize,
    value: KVar,
) -> Result<(), Error> {
    let path = |limb| match matrix {
        Some(matrix) => FieldPath::YRingLimb {
            source,
            matrix,
            lane,
            limb,
        },
        None => FieldPath::YZcolLimb { source, lane, limb },
    };
    output.push(path(KLimb::C0), value.c0)?;
    output.push(path(KLimb::C1), value.c1)
}
