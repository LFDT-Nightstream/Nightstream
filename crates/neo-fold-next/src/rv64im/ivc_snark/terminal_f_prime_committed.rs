//! Owns the terminal Construction-2 committed-step instance for RV64IM.
//!
//! The boundary here is `u_i = (C_i, x_i)`: `C_i` must be an Ajtai commitment
//! to the low-norm SuperNeo image that reconstructs the terminal `F'` R2
//! assignment, and `x_i` is the public `enc_inst` image in the leading slots.

use std::collections::BTreeMap;
use std::ops::Range;

use neo_ajtai::Commitment;
use neo_ccs::{poly::SparsePoly, CcsMatrix, CcsStructure, Mat};
use neo_math::{balanced::to_balanced_i128, D, F};
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::{
    bellpepper::{r1cs::SpartanWitness, solver::SatisfyingAssignment},
    traits::{transcript::TranscriptEngineTrait, Engine},
};

use crate::rv64im::construction2::{
    commit_rv64im_main_recursion_construction2_packed_z, Rv64imMainRecursionConstruction2Commitment,
    Rv64imMainRecursionConstruction2FreshInstance, Rv64imMainRecursionConstruction2PublicBoundary,
};
use crate::rv64im::f_prime::RV64IM_ENC_INST_BITS;
use crate::rv64im::main_relation_spartan::{
    build_rv64im_terminal_f_prime_r2_circuit, Rv64imMainRecursionFPrimeBackendRelation, Rv64imMainRecursionStepCircuit,
    Rv64imMainRecursionStepSpartanPublishedTarget, Rv64imMainRecursionStepSpartanShape,
};
use crate::rv64im::SimpleKernelError;
use crate::witness_layout::{commit_cols_for_full_width, encode_vector_for_full_width};

use super::{Rv64imDeciderEngine, ShapeCS, SpartanCircuit, SpartanF, SpartanShape, SplitR1CSShape};

mod circuit;
mod labels;
use labels::collect_private_witness_labels;
mod proof_circuit;
pub(crate) use proof_circuit::terminal_f_prime_committed_step_boundary_public_values;

const U32_BIT_WIDTH: usize = 32;
const U64_BIT_WIDTH: usize = 64;

#[derive(Clone, Debug)]
pub(crate) struct Rv64imTerminalFPrimeCommittedRelation {
    public_boundary: Rv64imMainRecursionConstruction2PublicBoundary,
    r2_assignment: Rv64imTerminalFPrimeR2Assignment,
    superneo_pack_status: Rv64imTerminalFPrimeSuperNeoPackStatus,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imTerminalFPrimeCommittedStepSetup {
    step_cap: usize,
    r2_assignment: Rv64imTerminalFPrimeR2Assignment,
    f_prime_circuit: Rv64imMainRecursionStepCircuit,
    packed_cols: usize,
    public_boundary: Rv64imMainRecursionConstruction2PublicBoundary,
}

#[derive(Clone, Debug)]
struct Rv64imTerminalFPrimeSuperNeoPackStatus {
    commitment: Option<Commitment>,
    error: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imTerminalFPrimeR1csCcsRelation {
    structure: CcsStructure<F>,
    layout: Rv64imTerminalFPrimeR2ColumnLayout,
    num_spartan_public: usize,
    num_challenges: usize,
    num_variables: usize,
    num_constraints: usize,
    total_nnz: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rv64imTerminalFPrimePrivateColumnEncoding {
    UnusedPadding,
    Bit,
    U32,
    U64,
}

#[derive(Clone, Debug)]
struct Rv64imTerminalFPrimeR2ColumnLayout {
    r2_public_len: usize,
    source_labels: Vec<String>,
    source_encodings: Vec<Rv64imTerminalFPrimePrivateColumnEncoding>,
    source_offsets: Vec<usize>,
    source_by_label: BTreeMap<String, usize>,
    source_limb_width: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imTerminalFPrimeR2Assignment {
    // Terminal F' R1CS assignment exported as committed R2 relation. Only
    // source low-norm values may be committed as SuperNeo digits; unclassified
    // full-field R1CS auxiliaries are rejected before CCS matrix expansion.
    relation: Rv64imTerminalFPrimeR1csCcsRelation,
    terminal_public_values: Vec<F>,
    r2_public_values: Vec<F>,
    witness_values: Vec<F>,
    f_prime_circuit: Rv64imMainRecursionStepCircuit,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imTerminalFPrimeCommittedStepCircuit {
    assignment: Rv64imTerminalFPrimeR2Assignment,
    public_boundary: Rv64imMainRecursionConstruction2PublicBoundary,
    f_prime_circuit: Rv64imMainRecursionStepCircuit,
}

struct Rv64imTerminalFPrimeR2ShapeExport {
    circuit: Rv64imMainRecursionStepCircuit,
    split_shape: SplitR1CSShape<Rv64imDeciderEngine>,
    relation: Rv64imTerminalFPrimeR1csCcsRelation,
    expected_public_values: Vec<SpartanF>,
    private_witness_labels: Vec<String>,
}

impl Rv64imTerminalFPrimeCommittedRelation {
    pub(crate) fn from_backend(
        spartan_shape: &Rv64imMainRecursionStepSpartanShape,
        backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
    ) -> Result<Self, SimpleKernelError> {
        ensure_terminal_f_prime_output_uses_x_only_placeholder(&backend_relation.construction2_u_next)?;
        let step_cap = backend_relation
            .f_prime_advice
            .verifier_key_fs()
            .step_cap()?;
        let public_boundary =
            Rv64imMainRecursionConstruction2PublicBoundary::from_fresh_instance(&backend_relation.construction2_u_next);
        let r2_assignment = Rv64imTerminalFPrimeR2Assignment::from_backend(spartan_shape, backend_relation)?;
        let superneo_pack_status = Rv64imTerminalFPrimeSuperNeoPackStatus::from_r2_assignment(step_cap, &r2_assignment);
        let public_boundary = superneo_pack_status
            .public_boundary_for_x_i(public_boundary.x_i.clone())
            .unwrap_or(public_boundary);

        Ok(Self {
            public_boundary,
            r2_assignment,
            superneo_pack_status,
        })
    }

    pub(crate) fn public_boundary(&self) -> &Rv64imMainRecursionConstruction2PublicBoundary {
        &self.public_boundary
    }

    pub(crate) fn r1cs_ccs(&self) -> &Rv64imTerminalFPrimeR1csCcsRelation {
        self.r2_assignment.relation()
    }

    pub(crate) fn terminal_r2_superneo_pack_error(&self) -> Option<&str> {
        self.superneo_pack_status.error.as_deref()
    }

    pub(crate) fn require_superneo_assignment_commitment(&self) -> Result<(), SimpleKernelError> {
        if let Some(error) = self.terminal_r2_superneo_pack_error() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM terminal F' R2 assignment cannot be accepted as u_i.C: {error}"
            )));
        }
        let Some(commitment) = self.superneo_pack_status.commitment.as_ref() else {
            return Err(SimpleKernelError::Bridge(
                "RV64IM terminal F' R2 assignment did not produce a SuperNeo commitment".into(),
            ));
        };
        if !commitment_matches_public_boundary(commitment, &self.public_boundary) {
            return Err(SimpleKernelError::Bridge(
                "RV64IM terminal F' R2 assignment commitment does not match published u_i.C".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn committed_step_circuit(&self) -> Result<Rv64imTerminalFPrimeCommittedStepCircuit, SimpleKernelError> {
        if let Some(error) = self.terminal_r2_superneo_pack_error() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM terminal F' R2 assignment cannot be packed for committed-step proof: {error}"
            )));
        }
        Ok(Rv64imTerminalFPrimeCommittedStepCircuit {
            assignment: self.r2_assignment.clone(),
            public_boundary: self.public_boundary.clone(),
            f_prime_circuit: self.r2_assignment.f_prime_circuit.clone(),
        })
    }

    pub(crate) fn committed_step_setup(
        &self,
        step_cap: usize,
    ) -> Result<Rv64imTerminalFPrimeCommittedStepSetup, SimpleKernelError> {
        self.require_superneo_assignment_commitment()?;
        let full_width = self.r2_assignment.committed_full_width()?;
        Ok(Rv64imTerminalFPrimeCommittedStepSetup {
            step_cap,
            r2_assignment: self.r2_assignment.clone(),
            f_prime_circuit: self.r2_assignment.f_prime_circuit.clone(),
            packed_cols: commit_cols_for_full_width(full_width),
            public_boundary: self.public_boundary.clone(),
        })
    }

    pub(crate) fn validate_shape(&self) -> Result<(), SimpleKernelError> {
        if self.public_boundary.commitment_data.is_empty() {
            return Err(SimpleKernelError::Bridge(
                "RV64IM terminal F' committed-step boundary must carry u_i.C commitment data".into(),
            ));
        }
        if self.r1cs_ccs().structure().m == 0 || self.r1cs_ccs().structure().n == 0 {
            return Err(SimpleKernelError::Bridge(
                "RV64IM terminal F' committed-step R2 CCS shape is empty".into(),
            ));
        }
        self.r2_assignment.validate()?;
        Ok(())
    }
}

impl Rv64imTerminalFPrimeCommittedStepSetup {
    pub(crate) fn from_backend_shape(
        spartan_shape: &Rv64imMainRecursionStepSpartanShape,
        backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
    ) -> Result<Self, SimpleKernelError> {
        ensure_terminal_f_prime_output_uses_x_only_placeholder(&backend_relation.construction2_u_next)?;
        let step_cap = backend_relation
            .f_prime_advice
            .verifier_key_fs()
            .step_cap()?;
        let r2_assignment = Rv64imTerminalFPrimeR2Assignment::shape_from_backend(spartan_shape, backend_relation)?;
        r2_assignment.validate_shape_only()?;

        let full_width = r2_assignment.committed_full_width()?;
        let packed_cols = commit_cols_for_full_width(full_width);
        let zero_packed = Mat::zero(D, packed_cols, F::ZERO);
        let commitment = commit_rv64im_main_recursion_construction2_packed_z(full_width, step_cap, &zero_packed)
            .map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM terminal F' setup commitment context failed for full width {full_width}: {err}"
                ))
            })?;
        let fresh_instance = Rv64imMainRecursionConstruction2FreshInstance::from_parts(
            Rv64imMainRecursionConstruction2Commitment::from_commitment(commitment),
            backend_relation.construction2_u_next.x_i().clone(),
        );
        let public_boundary = Rv64imMainRecursionConstruction2PublicBoundary::from_fresh_instance(&fresh_instance);
        Ok(Self {
            step_cap,
            f_prime_circuit: r2_assignment.f_prime_circuit.clone(),
            r2_assignment,
            packed_cols,
            public_boundary,
        })
    }

    pub(crate) fn step_cap(&self) -> usize {
        self.step_cap
    }

    pub(crate) fn r1cs_ccs(&self) -> &Rv64imTerminalFPrimeR1csCcsRelation {
        self.r2_assignment.relation()
    }

    pub(crate) fn r2_witness_len(&self) -> usize {
        self.r2_assignment.witness_values().len()
    }

    pub(crate) fn terminal_r2_committed_low_norm_width(&self) -> Result<usize, SimpleKernelError> {
        self.r2_assignment.committed_full_width()
    }

    pub(crate) fn terminal_r2_superneo_packed_cols(&self) -> usize {
        self.packed_cols
    }

    pub(crate) fn terminal_r2_commitment_words(&self) -> usize {
        self.public_boundary.commitment_data.len()
    }

    pub(crate) fn terminal_r2_private_encoding_counts(&self) -> (usize, usize, usize) {
        self.r2_assignment.relation().private_encoding_counts()
    }

    pub(crate) fn terminal_r2_private_padding_inputs(&self) -> usize {
        self.r2_assignment.relation().private_padding_inputs()
    }

    pub(crate) fn committed_step_circuit(&self) -> Rv64imTerminalFPrimeCommittedStepCircuit {
        Rv64imTerminalFPrimeCommittedStepCircuit {
            assignment: self.r2_assignment.clone(),
            public_boundary: self.public_boundary.clone(),
            f_prime_circuit: self.f_prime_circuit.clone(),
        }
    }
}

impl Rv64imTerminalFPrimeSuperNeoPackStatus {
    fn from_r2_assignment(step_cap: usize, assignment: &Rv64imTerminalFPrimeR2Assignment) -> Self {
        Self::try_from_r2_assignment(step_cap, assignment).unwrap_or_else(|error| Self {
            commitment: None,
            error: Some(error),
        })
    }

    fn try_from_r2_assignment(step_cap: usize, assignment: &Rv64imTerminalFPrimeR2Assignment) -> Result<Self, String> {
        let full_width = assignment
            .committed_full_width()
            .map_err(|err| err.to_string())?;
        let params = NeoParams::goldilocks_auto_r1cs_ccs(full_width)
            .map_err(|err| format!("RV64IM terminal F' R2 SuperNeo params failed: {err}"))?;
        let full_vector = assignment
            .committed_full_vector()
            .map_err(|err| err.to_string())?;
        if let Some(error) = assignment.committed_nc_range_error(&params, &full_vector) {
            return Err(error);
        }
        let packed = encode_vector_for_full_width(&params, full_width, &full_vector)
            .map_err(|err| format!("RV64IM terminal F' R2 SuperNeo packing failed: {err}"))?;
        let commitment = commit_rv64im_main_recursion_construction2_packed_z(full_width, step_cap, &packed)
            .map_err(|err| format!("RV64IM terminal F' R2 SuperNeo commitment failed: {err}"))?;
        Ok(Self {
            commitment: Some(commitment),
            error: None,
        })
    }

    fn public_boundary_for_x_i(
        &self,
        x_i: crate::rv64im::f_prime::Rv64imEncodedPublicInput,
    ) -> Option<Rv64imMainRecursionConstruction2PublicBoundary> {
        let commitment = self.commitment.clone()?;
        let fresh_instance = Rv64imMainRecursionConstruction2FreshInstance::from_parts(
            Rv64imMainRecursionConstruction2Commitment::from_commitment(commitment),
            x_i,
        );
        Some(Rv64imMainRecursionConstruction2PublicBoundary::from_fresh_instance(
            &fresh_instance,
        ))
    }
}

fn commitment_matches_public_boundary(
    commitment: &Commitment,
    public_boundary: &Rv64imMainRecursionConstruction2PublicBoundary,
) -> bool {
    commitment.d as u64 == public_boundary.commitment_d
        && commitment.kappa as u64 == public_boundary.commitment_kappa
        && commitment.data == public_boundary.commitment_data
}

fn ensure_terminal_f_prime_output_uses_x_only_placeholder(
    construction2_u_next: &Rv64imMainRecursionConstruction2FreshInstance,
) -> Result<(), SimpleKernelError> {
    let commitment = construction2_u_next.commitment().commitment();
    if commitment.d != D
        || commitment.kappa != 1
        || commitment.data.len() != D
        || commitment.data.iter().any(|value| *value != F::ZERO)
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM terminal F' backend output must carry the x-only zero commitment placeholder; terminal R2 derives authoritative u_i.C"
                .into(),
        ));
    }
    Ok(())
}

impl Rv64imTerminalFPrimeR1csCcsRelation {
    fn from_split_shape(
        split_shape: &SplitR1CSShape<Rv64imDeciderEngine>,
        r2_public_range: Range<usize>,
        private_witness_labels: &[String],
    ) -> Result<Self, SimpleKernelError> {
        let regular_shape = split_shape.to_regular_shape();
        let (a, b, c) = regular_shape.matrices();
        let total_nnz = a.nnz() + b.nnz() + c.nnz();
        let num_variables = regular_shape.num_variables();
        let num_io = regular_shape.num_io();
        let private_witness_labels = padded_private_witness_labels(split_shape, private_witness_labels)?;
        if private_witness_labels.len() != num_variables {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM terminal F' padded witness label count mismatch: expected {num_variables}, got {}",
                private_witness_labels.len()
            )));
        }
        let layout = Rv64imTerminalFPrimeR2ColumnLayout::new(num_io, r2_public_range, &private_witness_labels)?;
        let structure = terminal_source_witness_debug_structure(layout.committed_width())?;

        Ok(Self {
            structure,
            layout,
            num_spartan_public: split_shape.num_public(),
            num_challenges: split_shape.num_challenges(),
            num_variables,
            num_constraints: regular_shape.num_constraints(),
            total_nnz,
        })
    }

    pub(crate) fn structure(&self) -> &CcsStructure<F> {
        &self.structure
    }

    pub(crate) fn r2_public_len(&self) -> usize {
        self.layout.r2_public_len
    }

    pub(crate) fn num_public(&self) -> usize {
        self.r2_public_len()
    }

    pub(crate) fn committed_width(&self) -> usize {
        self.layout.committed_width()
    }

    pub(crate) fn num_spartan_public(&self) -> usize {
        self.num_spartan_public
    }

    pub(crate) fn num_challenges(&self) -> usize {
        self.num_challenges
    }

    pub(crate) fn num_variables(&self) -> usize {
        self.num_variables
    }

    pub(crate) fn num_constraints(&self) -> usize {
        self.num_constraints
    }

    pub(crate) fn total_nnz(&self) -> usize {
        self.total_nnz
    }

    fn private_encoding_counts(&self) -> (usize, usize, usize) {
        self.layout.private_encoding_counts()
    }

    fn private_padding_inputs(&self) -> usize {
        self.layout.private_padding_inputs()
    }
}

fn terminal_source_witness_debug_structure(width: usize) -> Result<CcsStructure<F>, SimpleKernelError> {
    if width == 0 {
        return Err(SimpleKernelError::Bridge(
            "RV64IM terminal F' source witness width is empty".into(),
        ));
    }
    CcsStructure::new_sparse(vec![CcsMatrix::Identity { n: width }], SparsePoly::new(1, Vec::new()))
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal F' source structure failed: {err}")))
}

impl Rv64imTerminalFPrimeR2ColumnLayout {
    fn new(
        num_spartan_public: usize,
        r2_public_range: Range<usize>,
        private_witness_labels: &[Option<String>],
    ) -> Result<Self, SimpleKernelError> {
        let r2_public_len = validate_terminal_r2_public_range(&r2_public_range, num_spartan_public)?;
        let mut source_labels = Vec::new();
        let mut source_encodings = Vec::new();
        let mut source_offsets = Vec::new();
        let mut source_by_label = BTreeMap::new();
        let mut source_limb_width = 0usize;
        for label in private_witness_labels.iter().flatten() {
            if !is_terminal_f_prime_committed_source_label(label) {
                continue;
            }
            let encoding = Rv64imTerminalFPrimePrivateColumnEncoding::from_label(Some(label));
            if encoding == Rv64imTerminalFPrimePrivateColumnEncoding::UnusedPadding {
                continue;
            }
            source_by_label.insert(label.clone(), source_labels.len());
            source_labels.push(label.clone());
            source_encodings.push(encoding);
            source_offsets.push(source_limb_width);
            source_limb_width = source_limb_width
                .checked_add(encoding.limb_count())
                .ok_or_else(|| SimpleKernelError::Bridge("RV64IM terminal F' source limb width overflow".into()))?;
        }
        Ok(Self {
            r2_public_len,
            source_labels,
            source_encodings,
            source_offsets,
            source_by_label,
            source_limb_width,
        })
    }

    fn committed_width(&self) -> usize {
        self.r2_public_len + self.source_limb_width + 1
    }

    fn source_limb_width(&self) -> usize {
        self.source_limb_width
    }

    fn private_encoding_counts(&self) -> (usize, usize, usize) {
        let mut bit_inputs = 0usize;
        let mut u32_inputs = 0usize;
        let mut u64_inputs = 0usize;
        for encoding in &self.source_encodings {
            match encoding {
                Rv64imTerminalFPrimePrivateColumnEncoding::UnusedPadding => {}
                Rv64imTerminalFPrimePrivateColumnEncoding::Bit => bit_inputs += 1,
                Rv64imTerminalFPrimePrivateColumnEncoding::U32 => u32_inputs += 1,
                Rv64imTerminalFPrimePrivateColumnEncoding::U64 => u64_inputs += 1,
            }
        }
        (bit_inputs, u32_inputs, u64_inputs)
    }

    fn private_padding_inputs(&self) -> usize {
        0
    }

    fn witness_encoding(
        &self,
        source_idx: usize,
    ) -> Result<Rv64imTerminalFPrimePrivateColumnEncoding, SimpleKernelError> {
        self.source_encodings
            .get(source_idx)
            .copied()
            .ok_or_else(|| SimpleKernelError::Bridge("RV64IM terminal F' source witness index out of bounds".into()))
    }

    fn source_binding(&self, label: &str) -> Option<(usize, Rv64imTerminalFPrimePrivateColumnEncoding)> {
        let source_idx = *self.source_by_label.get(label)?;
        Some((self.source_offsets[source_idx], self.source_encodings[source_idx]))
    }
}

impl Rv64imTerminalFPrimePrivateColumnEncoding {
    fn from_label(label: Option<&str>) -> Self {
        let Some(label) = label else {
            return Self::UnusedPadding;
        };
        if label.contains("boolean")
            || label.contains("halted")
            || label.contains("_bit_")
            || label.contains("bit_")
            || label.ends_with("_bit")
            || label.ends_with("_bit/num")
        {
            return Self::Bit;
        }
        if label.contains("half") || label.contains("halves") {
            return Self::U32;
        }
        Self::U64
    }

    fn limb_count(self) -> usize {
        match self {
            Self::UnusedPadding => 0,
            Self::Bit => 1,
            Self::U32 => U32_BIT_WIDTH,
            Self::U64 => U64_BIT_WIDTH,
        }
    }

    fn limb_label(self, limb_idx: usize) -> String {
        match self {
            Self::UnusedPadding => "padding".to_string(),
            Self::Bit => "bit".to_string(),
            Self::U32 | Self::U64 => format!("bit{limb_idx}"),
        }
    }
}

fn is_terminal_f_prime_committed_source_label(label: &str) -> bool {
    let root = label.split('/').next().unwrap_or(label);
    matches!(
        root,
        "chunk_index_halves"
            | "z_i"
            | "pc_next_halves"
            | "terminal_halted_out_halves"
            | "step_handle_meta"
            | "state_in"
            | "state_out"
            | "expected_bridge_handoff_digest"
            | "construction2_current_input"
            | "canonical_initial_z"
            | "statement_chunk_index_halves"
            | "statement_z_0"
            | "statement_z_next"
            | "statement_step_hi_halves"
    )
}

impl Rv64imTerminalFPrimeR2Assignment {
    pub(crate) fn shape_from_backend(
        spartan_shape: &Rv64imMainRecursionStepSpartanShape,
        backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
    ) -> Result<Self, SimpleKernelError> {
        let export = terminal_f_prime_r2_shape_export(spartan_shape, backend_relation)?;
        let terminal_public_values = export
            .expected_public_values
            .iter()
            .map(|value| F::from_u64(value.to_canonical_u64()))
            .collect::<Vec<_>>();
        let r2_public_range = Rv64imMainRecursionStepSpartanPublishedTarget::terminal_r2_public_value_range_static();
        let (r2_public_values, _) = split_terminal_r2_public_values(&export.expected_public_values, r2_public_range)?;

        let mut witness_values = vec![F::ZERO; export.relation.layout.source_limb_width()];
        witness_values.push(F::ONE);

        Ok(Self {
            relation: export.relation,
            terminal_public_values,
            r2_public_values,
            witness_values,
            f_prime_circuit: export.circuit,
        })
    }

    pub(crate) fn from_backend(
        spartan_shape: &Rv64imMainRecursionStepSpartanShape,
        backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
    ) -> Result<Self, SimpleKernelError> {
        let export = terminal_f_prime_r2_shape_export(spartan_shape, backend_relation)?;
        let regular_shape = export.split_shape.to_regular_shape();
        let (ck, _) = SplitR1CSShape::commitment_key(&[&export.split_shape]).map_err(|err| {
            SimpleKernelError::Bridge(format!("RV64IM terminal F' R1CS export commitment key failed: {err}"))
        })?;
        let mut state = SatisfyingAssignment::<Rv64imDeciderEngine>::shared_witness(
            &export.split_shape,
            &ck,
            &export.circuit,
            false,
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!("RV64IM terminal F' R1CS export shared witness failed: {err}"))
        })?;
        SatisfyingAssignment::<Rv64imDeciderEngine>::precommitted_witness(
            &mut state,
            &export.split_shape,
            &ck,
            &export.circuit,
            false,
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM terminal F' R1CS export precommitted witness failed: {err}"
            ))
        })?;
        let mut transcript = <Rv64imDeciderEngine as Engine>::TE::new(b"rv64im_terminal_f_prime_r2_assignment");
        let (instance, witness) = SatisfyingAssignment::<Rv64imDeciderEngine>::r1cs_instance_and_witness(
            &mut state,
            &export.split_shape,
            &ck,
            &export.circuit,
            false,
            &mut transcript,
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM terminal F' R1CS export witness synthesis failed: {err}"
            ))
        })?;

        let regular_instance = instance.to_regular_instance().map_err(|err| {
            SimpleKernelError::Bridge(format!("RV64IM terminal F' R1CS instance flatten failed: {err}"))
        })?;
        regular_shape
            .is_sat(&ck, &regular_instance, &witness)
            .map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM terminal F' exported R1CS witness is not satisfying: {err}"
                ))
            })?;
        let public_values_spartan = regular_instance.public_values();
        if public_values_spartan != export.expected_public_values.as_slice() {
            return Err(SimpleKernelError::Bridge(
                "RV64IM terminal F' exported R1CS public IO does not match the recursive-step public target".into(),
            ));
        }

        let terminal_public_values = public_values_spartan
            .iter()
            .map(|value| F::from_u64(value.to_canonical_u64()))
            .collect::<Vec<_>>();
        let relation = export.relation;
        let private_witness_labels =
            padded_private_witness_labels(&export.split_shape, &export.private_witness_labels)?;
        let r2_public_range = Rv64imMainRecursionStepSpartanPublishedTarget::terminal_r2_public_value_range_static();
        let (r2_public_values, _) = split_terminal_r2_public_values(public_values_spartan, r2_public_range)?;

        let mut witness_values = Vec::with_capacity(
            relation
                .layout
                .source_limb_width()
                .checked_add(1)
                .ok_or_else(|| {
                    SimpleKernelError::Bridge("RV64IM terminal F' low-norm witness length overflow".into())
                })?,
        );
        for (witness_idx, value) in witness.values().iter().enumerate() {
            let Some(Some(label)) = private_witness_labels.get(witness_idx) else {
                continue;
            };
            let Some((offset, encoding)) = relation.layout.source_binding(label) else {
                continue;
            };
            if witness_values.len() < offset {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM terminal F' source witness cursor skipped an offset".into(),
                ));
            }
            while witness_values.len() < offset {
                witness_values.push(F::ZERO);
            }
            let native = F::from_u64(value.to_canonical_u64());
            witness_values.extend(low_norm_encoded_values(native, encoding)?);
        }
        if witness_values.len() != relation.layout.source_limb_width() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM terminal F' source witness length mismatch: expected {}, got {}",
                relation.layout.source_limb_width(),
                witness_values.len()
            )));
        }
        witness_values.push(F::ONE);

        let assignment = Self {
            relation,
            terminal_public_values,
            r2_public_values,
            witness_values,
            f_prime_circuit: export.circuit,
        };
        assignment.validate()?;
        Ok(assignment)
    }

    pub(crate) fn relation(&self) -> &Rv64imTerminalFPrimeR1csCcsRelation {
        &self.relation
    }

    fn witness_values(&self) -> &[F] {
        &self.witness_values
    }

    fn committed_witness_values(&self) -> &[F] {
        &self.witness_values
    }

    fn committed_full_width(&self) -> Result<usize, SimpleKernelError> {
        Ok(self.relation.committed_width())
    }

    fn committed_full_vector(&self) -> Result<Vec<F>, SimpleKernelError> {
        let full_width = self.committed_full_width()?;
        let mut full_vector = Vec::with_capacity(full_width);
        full_vector.extend_from_slice(&self.r2_public_values);
        full_vector.extend_from_slice(self.committed_witness_values());
        if full_vector.len() != full_width {
            return Err(SimpleKernelError::Bridge(
                "RV64IM terminal F' committed SuperNeo image length mismatch".into(),
            ));
        }
        Ok(full_vector)
    }

    fn committed_packed_witness(&self) -> Result<(NeoParams, Mat<F>), SimpleKernelError> {
        let full_width = self.committed_full_width()?;
        let params = NeoParams::goldilocks_auto_r1cs_ccs(full_width)
            .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal F' R2 params failed: {err}")))?;
        let full_vector = self.committed_full_vector()?;
        if let Some(error) = self.committed_nc_range_error(&params, &full_vector) {
            return Err(SimpleKernelError::Bridge(error));
        }
        let packed = encode_vector_for_full_width(&params, full_width, &full_vector)
            .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal F' R2 packing failed: {err}")))?;
        Ok((params, packed))
    }

    fn committed_nc_range_error(&self, params: &NeoParams, full_vector: &[F]) -> Option<String> {
        for (committed_idx, value) in full_vector.iter().copied().enumerate() {
            if is_superneo_digit_representable(value, params.b) {
                continue;
            }
            return Some(format!(
                "RV64IM terminal F' R2 committed value at {} is not representable in D={} balanced base-{} digits (centered value {})",
                self.committed_index_label(committed_idx),
                D,
                params.b,
                to_balanced_i128(value),
            ));
        }
        None
    }

    fn committed_index_label(&self, committed_idx: usize) -> String {
        let public_len = self.r2_public_values.len();
        if committed_idx < public_len {
            return format!("committed index {committed_idx} / u_i.x_i bit {committed_idx}");
        }
        let committed_witness_idx = committed_idx - public_len;
        if committed_witness_idx + 1 == self.witness_values.len() {
            return format!("committed index {committed_idx} / terminal R2 constant-one slot");
        }
        let Some((r1cs_var_idx, limb_idx, encoding)) = self.committed_witness_limb(committed_witness_idx) else {
            return format!(
                "committed index {committed_idx} / terminal R2 private witness limb {committed_witness_idx}"
            );
        };
        let limb = encoding.limb_label(limb_idx);
        let label = self
            .relation
            .layout
            .source_labels
            .get(r1cs_var_idx)
            .map(String::as_str)
            .unwrap_or("<unknown>");
        format!("committed index {committed_idx} / terminal R2 source witness {r1cs_var_idx}.{limb} ({label})")
    }

    fn committed_witness_limb(
        &self,
        committed_witness_idx: usize,
    ) -> Option<(usize, usize, Rv64imTerminalFPrimePrivateColumnEncoding)> {
        for (witness_idx, encoding) in self
            .relation
            .layout
            .source_encodings
            .iter()
            .copied()
            .enumerate()
        {
            let start = self.relation.layout.source_offsets[witness_idx];
            let end = start.checked_add(encoding.limb_count())?;
            if (start..end).contains(&committed_witness_idx) {
                return Some((witness_idx, committed_witness_idx - start, encoding));
            }
        }
        None
    }

    fn validate(&self) -> Result<(), SimpleKernelError> {
        self.validate_shape_only()?;
        self.committed_full_vector()?;
        Ok(())
    }

    fn validate_shape_only(&self) -> Result<(), SimpleKernelError> {
        if self.relation.num_challenges() != 0 {
            return Err(SimpleKernelError::Bridge(
                "RV64IM terminal F' R2 assignment requires challenge-free public IO".into(),
            ));
        }
        if self.r2_public_values.len() != self.relation.r2_public_len() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM terminal F' R2 public input length mismatch: expected {}, got {}",
                self.relation.r2_public_len(),
                self.r2_public_values.len()
            )));
        }
        if self.r2_public_values.len() != RV64IM_ENC_INST_BITS {
            return Err(SimpleKernelError::Bridge(
                "RV64IM terminal F' R2 public input must be the 256-bit Construction-2 enc_inst image".into(),
            ));
        }
        if self.terminal_public_values.len() != self.relation.num_spartan_public() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM terminal F' Spartan public input length mismatch: expected {}, got {}",
                self.relation.num_spartan_public(),
                self.terminal_public_values.len()
            )));
        }
        let expected_witness_values = self
            .relation
            .layout
            .source_limb_width()
            .checked_add(1)
            .ok_or_else(|| SimpleKernelError::Bridge("RV64IM terminal F' witness length overflow".into()))?;
        if self.witness_values.len() != expected_witness_values {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM terminal F' R2 witness value length mismatch: expected {}, got {}",
                expected_witness_values,
                self.witness_values.len()
            )));
        }
        if self.witness_values.last().copied() != Some(F::ONE) {
            return Err(SimpleKernelError::Bridge(
                "RV64IM terminal F' R2 assignment does not end with the R1CS constant-one slot".into(),
            ));
        }
        let total_len = self
            .r2_public_values
            .len()
            .checked_add(self.witness_values.len())
            .ok_or_else(|| SimpleKernelError::Bridge("RV64IM terminal F' R2 assignment length overflow".into()))?;
        if total_len != self.relation.committed_width() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM terminal F' R2 assignment length mismatch: expected {}, got {}",
                self.relation.committed_width(),
                total_len
            )));
        }
        self.committed_full_vector()?;
        Ok(())
    }
}

fn terminal_f_prime_r2_shape_export(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Rv64imTerminalFPrimeR2ShapeExport, SimpleKernelError> {
    let circuit = build_rv64im_terminal_f_prime_r2_circuit(spartan_shape, backend_relation)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal F' circuit build failed: {err}")))?;
    let expected_public_values = circuit
        .public_values()
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal F' public IO failed: {err}")))?;
    let r2_public_range = Rv64imMainRecursionStepSpartanPublishedTarget::terminal_r2_public_value_range_static();
    let split_shape = ShapeCS::<Rv64imDeciderEngine>::r1cs_shape(&circuit).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM terminal F' R1CS export failed for chunk_count_in={} terminal_step={} fresh_claim_count={} state_in_claim_count={} pc_i={} pc_next={}: {err}",
            backend_relation.f_prime_advice.chunk_count_in(),
            backend_relation.f_prime_advice.bridge_handoff_halted_out(),
            backend_relation.payload.step_shape.fresh_claim_count,
            backend_relation.payload.step_shape.state_in_claim_count,
            backend_relation.payload.pc_i(),
            backend_relation.payload.pc_next(),
        ))
    })?;
    let private_witness_labels = collect_private_witness_labels(&circuit)?;
    if private_witness_labels.len() != split_shape.num_variables_unpadded() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM terminal F' unpadded label count mismatch: expected {}, got {}",
            split_shape.num_variables_unpadded(),
            private_witness_labels.len()
        )));
    }
    let relation =
        Rv64imTerminalFPrimeR1csCcsRelation::from_split_shape(&split_shape, r2_public_range, &private_witness_labels)?;
    if relation.num_challenges() != 0 {
        return Err(SimpleKernelError::Bridge(
            "RV64IM terminal F' sparse CCS export does not yet support verifier-derived R1CS challenges".into(),
        ));
    }
    Ok(Rv64imTerminalFPrimeR2ShapeExport {
        circuit,
        split_shape,
        relation,
        expected_public_values,
        private_witness_labels,
    })
}

fn padded_private_witness_labels(
    split_shape: &SplitR1CSShape<Rv64imDeciderEngine>,
    private_witness_labels: &[String],
) -> Result<Vec<Option<String>>, SimpleKernelError> {
    if private_witness_labels.len() != split_shape.num_variables_unpadded() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM terminal F' unpadded witness label count mismatch: expected {}, got {}",
            split_shape.num_variables_unpadded(),
            private_witness_labels.len()
        )));
    }

    let mut padded = Vec::with_capacity(split_shape.num_variables());
    let mut cursor = 0usize;
    push_padded_witness_label_segment(
        &mut padded,
        private_witness_labels,
        &mut cursor,
        split_shape.num_shared_unpadded(),
        split_shape.num_shared(),
        "shared",
    )?;
    push_padded_witness_label_segment(
        &mut padded,
        private_witness_labels,
        &mut cursor,
        split_shape.num_precommitted_unpadded(),
        split_shape.num_precommitted(),
        "precommitted",
    )?;
    push_padded_witness_label_segment(
        &mut padded,
        private_witness_labels,
        &mut cursor,
        split_shape.num_rest_unpadded(),
        split_shape.num_rest(),
        "rest",
    )?;

    if cursor != private_witness_labels.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM terminal F' witness label padding consumed {cursor} labels but {} were supplied",
            private_witness_labels.len()
        )));
    }
    if padded.len() != split_shape.num_variables() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM terminal F' padded witness label count mismatch: expected {}, got {}",
            split_shape.num_variables(),
            padded.len()
        )));
    }
    Ok(padded)
}

fn push_padded_witness_label_segment(
    padded: &mut Vec<Option<String>>,
    labels: &[String],
    cursor: &mut usize,
    unpadded_len: usize,
    padded_len: usize,
    segment_name: &str,
) -> Result<(), SimpleKernelError> {
    if padded_len < unpadded_len {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM terminal F' {segment_name} witness segment has padded length {padded_len} below unpadded length {unpadded_len}"
        )));
    }
    let end = cursor
        .checked_add(unpadded_len)
        .ok_or_else(|| SimpleKernelError::Bridge("RV64IM terminal F' witness label cursor overflow".into()))?;
    if end > labels.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM terminal F' {segment_name} witness labels exceed collected label count"
        )));
    }
    padded.extend(labels[*cursor..end].iter().cloned().map(Some));
    padded.resize(padded.len() + (padded_len - unpadded_len), None);
    *cursor = end;
    Ok(())
}

fn split_terminal_r2_public_values(
    public_values: &[SpartanF],
    r2_public_range: Range<usize>,
) -> Result<(Vec<F>, Vec<F>), SimpleKernelError> {
    validate_terminal_r2_public_range(&r2_public_range, public_values.len())?;
    let mut r2_public_values = Vec::with_capacity(r2_public_range.len());
    let mut r2_witness_public_values = Vec::with_capacity(public_values.len().saturating_sub(r2_public_range.len()));
    for (idx, value) in public_values.iter().enumerate() {
        let value = F::from_u64(value.to_canonical_u64());
        if r2_public_range.contains(&idx) {
            r2_public_values.push(value);
        } else {
            r2_witness_public_values.push(value);
        }
    }
    Ok((r2_public_values, r2_witness_public_values))
}

fn low_norm_encoded_values(
    value: F,
    encoding: Rv64imTerminalFPrimePrivateColumnEncoding,
) -> Result<Vec<F>, SimpleKernelError> {
    match encoding {
        Rv64imTerminalFPrimePrivateColumnEncoding::UnusedPadding => {
            if value != F::ZERO {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV64IM terminal F' padded witness value is non-zero: {}",
                    value.as_canonical_u64()
                )));
            }
            Ok(Vec::new())
        }
        Rv64imTerminalFPrimePrivateColumnEncoding::Bit => {
            let canonical = value.as_canonical_u64();
            if canonical > 1 {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV64IM terminal F' boolean witness value is not binary: {canonical}"
                )));
            }
            Ok(vec![value])
        }
        Rv64imTerminalFPrimePrivateColumnEncoding::U32 => low_norm_bit_values(value, U32_BIT_WIDTH),
        Rv64imTerminalFPrimePrivateColumnEncoding::U64 => low_norm_bit_values(value, U64_BIT_WIDTH),
    }
}

fn low_norm_bit_values(value: F, bit_width: usize) -> Result<Vec<F>, SimpleKernelError> {
    let canonical = value.as_canonical_u64();
    if bit_width < U64_BIT_WIDTH && (canonical >> bit_width) != 0 {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM terminal F' witness value {canonical} does not fit in {bit_width} base-2 digits"
        )));
    }
    Ok((0..bit_width)
        .map(|bit_idx| F::from_u64((canonical >> bit_idx) & 1))
        .collect())
}

fn is_superneo_digit_representable(value: F, base: u32) -> bool {
    if base < 2 {
        return false;
    }
    let mut remainder = to_balanced_i128(value);
    let base = base as i128;
    for _ in 0..D {
        let (_, quotient) = balanced_divrem(remainder, base);
        remainder = quotient;
    }
    remainder == 0
}

fn balanced_divrem(value: i128, base: i128) -> (i128, i128) {
    debug_assert!(base >= 2);
    let mut remainder = value % base;
    let mut quotient = (value - remainder) / base;
    let half = base / 2;
    if remainder > half {
        remainder -= base;
        quotient += 1;
    } else if remainder < -half {
        remainder += base;
        quotient -= 1;
    }
    (remainder, quotient)
}

pub(crate) fn debug_check_rv64im_terminal_f_prime_r1cs_ccs_relation(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), SimpleKernelError> {
    let relation = Rv64imTerminalFPrimeCommittedRelation::from_backend(spartan_shape, backend_relation)?;
    relation.validate_shape()?;
    relation.require_superneo_assignment_commitment()
}

fn validate_terminal_r2_public_range(
    r2_public_range: &Range<usize>,
    num_io: usize,
) -> Result<usize, SimpleKernelError> {
    if r2_public_range.start >= r2_public_range.end || r2_public_range.end > num_io {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM terminal F' R2 public range {}..{} is outside Spartan public IO length {num_io}",
            r2_public_range.start, r2_public_range.end
        )));
    }
    Ok(r2_public_range.end - r2_public_range.start)
}
