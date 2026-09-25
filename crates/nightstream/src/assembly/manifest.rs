use std::collections::BTreeSet;

use serde::Deserialize;
use serde_json::Value;

use crate::application::ApplicationCircuit;

use super::{wire::Envelope, AssemblyError};

const PROFILE: [u64; 8] = [0xffff_ffff_0000_0001, 2, 16, 65536, 54, 28, 14, 13];
const CHILDREN: [&str; 14] = [
    "pilot_poseidon",
    "pi_ccs_poseidon",
    "pi_ccs_ordinary",
    "pilot_ordinary",
    "pilot_digest_binding",
    "pi_ccs_endpoint",
    "pi_rlc_sampler_poseidon",
    "pi_rlc_sampler_ordinary",
    "pi_rlc_combination",
    "pi_dec",
    "running_transition",
    "application",
    "next_preimage",
    "recursive_public_output",
];
pub(super) const SOURCE_FIELDS: [&str; 18] = [
    "hash_chains.input_start",
    "hash_chains.witness_start",
    "hash_chains.digest_start",
    "permutation_invocations.witness_start",
    "permutation_invocations.inputs.terms.column",
    "compact_row_invocations.local_start",
    "compact_row_invocations.input_ranges.column_start",
    "witness_batches.start",
    "witness_batches.recipes.var",
    "witness_batches.hints.source.var",
    "witness_instructions.target",
    "witness_instructions.a.terms.column",
    "witness_instructions.b.terms.column",
    "assertion_rows.a.terms.column",
    "assertion_rows.b.terms.column",
    "assertion_rows.c.terms.column",
    "layout.public_segments.start",
    "assignment.output_digest_expressions.var",
];

#[derive(Clone, Copy, Debug)]
pub(super) struct Counts {
    pub witness: usize,
    pub local: usize,
    pub rows: usize,
}
impl Counts {
    pub fn of(application: &ApplicationCircuit) -> Self {
        Self {
            witness: application.private_input_count(),
            local: application.generated_range().len(),
            rows: application.rows().len(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(transparent)]
pub(super) struct Dimension([usize; 4]);
impl Dimension {
    pub fn eval(&self, counts: Counts) -> Result<usize, AssemblyError> {
        self.0
            .iter()
            .zip([1, counts.witness, counts.local, counts.rows])
            .try_fold(0usize, |sum, (coefficient, count)| {
                sum.checked_add(
                    coefficient
                        .checked_mul(count)
                        .ok_or(AssemblyError::Overflow)?,
                )
                .ok_or(AssemblyError::Overflow)
            })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Port {
    pub name: String,
    pub role: String,
    pub count: Dimension,
    pub source_start: Dimension,
    pub retained_start: Dimension,
    pub slot_kind: usize,
}
impl Port {
    pub fn source_columns(&self, counts: Counts) -> Result<Vec<usize>, AssemblyError> {
        let start = self.source_start.eval(counts)?;
        let end = start
            .checked_add(self.count.eval(counts)?)
            .ok_or(AssemblyError::Overflow)?;
        Ok((start..end).collect())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Geometry {
    pub source_rows: Dimension,
    pub source_private: Dimension,
    pub source_constant: Dimension,
    pub source_public: usize,
    pub source_total: Dimension,
    pub logical_rows: Dimension,
    pub logical_width: Dimension,
    pub logical_public: usize,
    pub field_slot_width: usize,
    pub ring_degree: usize,
    pub domain: usize,
    pub one_column: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Child {
    pub id: String,
    pub opcode: usize,
    pub block_start: usize,
    pub block_count: usize,
    pub row_start: Dimension,
    pub row_count: Dimension,
    pub replaceable: bool,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Relocation {
    pub path: Vec<usize>,
    pub value: Dimension,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Run {
    pub first: Dimension,
    pub step: usize,
    pub count: Dimension,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AssignmentBlock {
    pub opcode: usize,
    pub slot_kind: usize,
    pub slot_count: Dimension,
    pub source_runs: Vec<Run>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SourceRelocation {
    pub source_private_start: usize,
    pub reference_constant: usize,
    pub reference_total: usize,
    pub source_column_fields: Vec<String>,
    pub prefix_row_start: usize,
    pub prefix_row_count: usize,
    pub application_row_start: usize,
    pub next_preimage_row_start: Dimension,
    pub next_preimage_row_count: usize,
    pub prefix_witness_end: usize,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecursivePublic {
    role: String,
    start: usize,
    count: usize,
    digest_port: String,
    marker_index: usize,
    digest_words: usize,
    first_bit: usize,
    word_bits: usize,
    zero_tail_start: usize,
    bit_order: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Terminal {
    running_claims: usize,
    fresh_claims: usize,
    all_final_rows: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SelectedReference {
    pub logical_rows: usize,
    pub logical_width: usize,
    pub application_matrix: Vec<Value>,
    pub application_local_index: usize,
    pub application_local: super::wire::AssignmentBlock,
    pub matrix_relocations: Vec<SelectedRelocation>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SelectedRelocation {
    pub path: Vec<usize>,
    pub selected: usize,
    pub ordinary: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    format: String,
    version: usize,
    id: String,
    profile: [u64; 8],
    dependencies: Vec<String>,
    parameters: Vec<String>,
    reference: [usize; 3],
    pub selected_reference: SelectedReference,
    pub geometry: Geometry,
    pub ports: Vec<Port>,
    recursive_public: RecursivePublic,
    pub application_matrix_template: Value,
    pub application_matrix_relocations: Vec<Relocation>,
    pub children: Vec<Child>,
    pub matrix_relocations: Vec<Relocation>,
    pub assignment_blocks: Vec<AssignmentBlock>,
    pub phi81_value_sources: Vec<Run>,
    pub phi81_challenge_sources: Vec<Run>,
    pub source_relocation: SourceRelocation,
    recipe_contract: Value,
    required_dimension_checks: Vec<String>,
    phase_order: Vec<String>,
    terminal: Terminal,
    contracts: Vec<String>,
    proof_scope: String,
}

impl Manifest {
    pub fn parse(bytes: &[u8]) -> Result<Self, AssemblyError> {
        let manifest: Self = serde_json::from_slice(bytes)?;
        if manifest.format != "nightstream.shared-verifier"
            || manifest.version != 2
            || manifest.id != "shared-recursive-verifier-v1"
            || manifest.profile != PROFILE
            || manifest.parameters != ["witness_words", "local_words", "application_rows"]
            || manifest.dependencies != ["poseidon2-permutation-v1", "poseidon2-external-v1", "phi81-product-v1"]
        {
            return Err(AssemblyError::Invalid("shared verifier format or profile"));
        }
        if manifest.children.len() != CHILDREN.len()
            || manifest
                .children
                .iter()
                .zip(CHILDREN)
                .enumerate()
                .any(|(opcode, (child, name))| {
                    child.id != name
                        || child.opcode != opcode
                        || child.replaceable != (name == "application")
                        || child.block_count == 0
                })
        {
            return Err(AssemblyError::Invalid("mandatory verifier child order"));
        }
        if manifest.source_relocation.source_column_fields != SOURCE_FIELDS
            || manifest.phase_order
                != [
                    "prior_state_hash",
                    "pi_ccs_sumcheck",
                    "pi_ccs_final",
                    "pi_rlc",
                    "pi_dec",
                    "application",
                    "output_hash",
                ]
            || manifest.required_dimension_checks
                != [
                    "source_rows_le_domain",
                    "source_total_le_domain",
                    "ring_padded_logical_width_le_domain",
                ]
            || manifest.terminal.running_claims != 16
            || manifest.terminal.fresh_claims != 1
            || !manifest.terminal.all_final_rows
            || manifest.contracts.is_empty()
            || manifest.proof_scope.is_empty()
        {
            return Err(AssemblyError::Invalid("shared verifier contract"));
        }
        let mut names = BTreeSet::new();
        for port in &manifest.ports {
            if !names.insert(&port.name) || port.slot_kind > 2 {
                return Err(AssemblyError::Invalid("shared verifier ports"));
            }
        }
        for (name, role) in [
            ("state_input", "private_input"),
            ("state_output", "private_output"),
            ("application_witness", "private_input"),
            ("application_local", "private_witness"),
            ("one", "constant"),
            ("prior_public_input", "public_input"),
            ("output_digest", "public_output"),
            ("verifier_context", "public_input"),
        ] {
            let port = manifest.port(name)?;
            if port.role != role || port.slot_kind != if name == "one" { 0 } else { 2 } {
                return Err(AssemblyError::Invalid("shared verifier port role or encoding"));
            }
        }
        let public = &manifest.recursive_public;
        if public.role != "public_output"
            || public.start != 0
            || public.count != manifest.geometry.logical_public
            || public.digest_port != "output_digest"
            || public.marker_index != manifest.geometry.one_column
            || public.bit_order != "little_endian"
            || public.first_bit.checked_add(
                public
                    .digest_words
                    .checked_mul(public.word_bits)
                    .ok_or(AssemblyError::Overflow)?,
            ) != Some(public.zero_tail_start)
            || public.zero_tail_start > public.count
        {
            return Err(AssemblyError::Invalid("recursive public wiring"));
        }
        let recipe = &manifest.recipe_contract;
        if recipe["row_form"] != "r1cs_a_times_b_equals_c"
            || recipe["expression_tags"] != serde_json::json!(["var", "const", "add", "mul"])
            || recipe["hint_tags"] != serde_json::json!(["bit", "inverse_or_zero", "quotient_five", "remainder_five"])
            || recipe["witness_instruction"] != "target_equals_a_times_b"
            || recipe["source_support"]
                != serde_json::json!([
                    "state_input",
                    "application_witness",
                    "state_output",
                    "application_local"
                ])
            || recipe["field_encoding"] != "balanced_ternary_41"
            || recipe["native_outputs_are_hints"] != true
        {
            return Err(AssemblyError::Invalid("application recipe contract"));
        }
        manifest.check_dimensions(manifest.reference())?;
        Ok(manifest)
    }

    pub fn reference(&self) -> Counts {
        Counts {
            witness: self.reference[0],
            local: self.reference[1],
            rows: self.reference[2],
        }
    }
    pub fn port(&self, name: &str) -> Result<&Port, AssemblyError> {
        self.ports
            .iter()
            .find(|port| port.name == name)
            .ok_or(AssemblyError::Invalid("missing verifier port"))
    }
    pub fn application_child(&self) -> &Child {
        self.children
            .iter()
            .find(|child| child.replaceable)
            .expect("validated application child")
    }

    pub fn check_dimensions(&self, counts: Counts) -> Result<(), AssemblyError> {
        let g = &self.geometry;
        if g.domain != 1usize << PROFILE[5] || g.ring_degree != PROFILE[4] as usize || g.field_slot_width != 41 {
            return Err(AssemblyError::Invalid("shared verifier encoding or domain"));
        }
        let width = g.logical_width.eval(counts)?;
        let carrier = width
            .checked_add(g.ring_degree - 1)
            .and_then(|rounded| (rounded / g.ring_degree).checked_mul(g.ring_degree))
            .ok_or(AssemblyError::Overflow)?;
        if g.source_rows.eval(counts)? > g.domain
            || g.source_total.eval(counts)? > g.domain
            || carrier > g.domain
            || g.logical_rows.eval(counts)? > g.domain
        {
            return Err(AssemblyError::Invalid("application exceeds exported domain"));
        }
        let private = g.source_private.eval(counts)?;
        if private != g.source_constant.eval(counts)?
            || private
                .checked_add(1)
                .and_then(|n| n.checked_add(g.source_public))
                != Some(g.source_total.eval(counts)?)
            || self.port("state_input")?.count.eval(counts)? != 4
            || self.port("state_output")?.count.eval(counts)? != 4
            || self.port("application_witness")?.count.eval(counts)? != counts.witness
            || self.port("application_local")?.count.eval(counts)? != counts.local
        {
            return Err(AssemblyError::Invalid("application interface dimensions"));
        }
        for port in &self.ports {
            let count = port.count.eval(counts)?;
            let source_end = port
                .source_start
                .eval(counts)?
                .checked_add(count)
                .ok_or(AssemblyError::Overflow)?;
            let encoded_width = count
                .checked_mul(if port.slot_kind == 2 { g.field_slot_width } else { 1 })
                .ok_or(AssemblyError::Overflow)?;
            let retained_end = port
                .retained_start
                .eval(counts)?
                .checked_add(encoded_width)
                .ok_or(AssemblyError::Overflow)?;
            if source_end > g.source_total.eval(counts)? || retained_end > width {
                return Err(AssemblyError::Invalid("verifier port range"));
            }
        }
        let witness = self.port("application_witness")?;
        let local = self.port("application_local")?;
        if witness.source_start.eval(counts)? != self.source_relocation.source_private_start
            || witness
                .source_start
                .eval(counts)?
                .checked_add(counts.witness)
                != Some(local.source_start.eval(counts)?)
            || local.source_start.eval(counts)?.checked_add(counts.local) != Some(private)
            || self.port("one")?.source_start.eval(counts)? != private
            || self.port("one")?.retained_start.eval(counts)? != g.one_column
        {
            return Err(AssemblyError::Invalid("application port allocation"));
        }
        let mut coordinates = g.logical_public;
        for (index, block) in self.assignment_blocks.iter().enumerate() {
            if block.opcode != index || block.slot_kind > 2 {
                return Err(AssemblyError::Invalid("assignment block order or encoding"));
            }
            if index == self.selected_reference.application_local_index {
                if coordinates != local.retained_start.eval(counts)? || block.slot_count.eval(counts)? != counts.local {
                    return Err(AssemblyError::Invalid("application retained allocation"));
                }
            }
            let block_width = block
                .slot_count
                .eval(counts)?
                .checked_mul(if block.slot_kind == 2 { g.field_slot_width } else { 1 })
                .ok_or(AssemblyError::Overflow)?;
            coordinates = coordinates
                .checked_add(block_width)
                .ok_or(AssemblyError::Overflow)?;
        }
        if self.selected_reference.application_local_index >= self.assignment_blocks.len() || coordinates != width {
            return Err(AssemblyError::Invalid("complete assignment block width"));
        }
        let mut blocks = 0usize;
        let mut rows = 0usize;
        for child in &self.children {
            if child.block_start != blocks || child.row_start.eval(counts)? != rows {
                return Err(AssemblyError::Invalid("verifier child coverage"));
            }
            blocks = blocks
                .checked_add(child.block_count)
                .ok_or(AssemblyError::Overflow)?;
            rows = rows
                .checked_add(child.row_count.eval(counts)?)
                .ok_or(AssemblyError::Overflow)?;
        }
        if rows != g.logical_rows.eval(counts)? {
            return Err(AssemblyError::Invalid("verifier child row total"));
        }
        Ok(())
    }

    pub fn check_reference(&self, reference: &Envelope) -> Result<(), AssemblyError> {
        let selected = &self.selected_reference;
        let child = self.application_child();
        let end = child
            .block_start
            .checked_add(selected.application_matrix.len())
            .ok_or(AssemblyError::Overflow)?;
        let block_count = self
            .children
            .iter()
            .map(|child| child.block_count)
            .sum::<usize>()
            .checked_sub(child.block_count)
            .ok_or(AssemblyError::Overflow)?
            .checked_add(selected.application_matrix.len())
            .ok_or(AssemblyError::Overflow)?;
        self.check_reference_geometry(reference, selected.logical_rows, selected.logical_width, block_count)?;
        if reference.matrix.get(child.block_start..end) != Some(selected.application_matrix.as_slice())
            || reference
                .assignment
                .blocks
                .get(selected.application_local_index)
                != Some(&selected.application_local)
        {
            return Err(AssemblyError::Invalid(
                "selected application suffix differs from manifest",
            ));
        }
        Ok(())
    }

    pub fn check_ordinary_reference(&self, reference: &Envelope) -> Result<(), AssemblyError> {
        self.check_reference_geometry(
            reference,
            self.geometry.logical_rows.eval(self.reference())?,
            self.geometry.logical_width.eval(self.reference())?,
            self.children.iter().map(|child| child.block_count).sum(),
        )
    }

    fn check_reference_geometry(
        &self,
        reference: &Envelope,
        logical_rows: usize,
        logical_width: usize,
        block_count: usize,
    ) -> Result<(), AssemblyError> {
        let counts = self.reference();
        let g = &self.geometry;
        let layout = &reference.source.layout;
        if reference.schema != 6
            || reference.source.schema != 8
            || reference.assignment.schema != 4
            || layout.rows != g.source_rows.eval(counts)?
            || layout.private != g.source_private.eval(counts)?
            || layout.constant != g.source_constant.eval(counts)?
            || layout.public != g.source_public
            || layout.total != g.source_total.eval(counts)?
            || reference.source.relation.rows != logical_rows
            || reference.source.relation.columns != logical_width
            || reference.logical_public != g.logical_public
            || block_count != reference.matrix.len()
        {
            return Err(AssemblyError::Invalid("manifest does not describe selected reference"));
        }
        let source = &self.source_relocation;
        if source.prefix_row_start != 0
            || source.prefix_row_count != source.application_row_start
            || source.prefix_witness_end != source.source_private_start
            || source.reference_constant != layout.constant
            || source.reference_total != layout.total
            || reference.next_preimage
                != [
                    source.next_preimage_row_start.eval(counts)?,
                    source.next_preimage_row_count,
                ]
        {
            return Err(AssemblyError::Invalid("reference source insertion"));
        }
        for (name, actual) in [
            ("state_input", &reference.application.input_columns),
            ("state_output", &reference.application.output_columns),
            ("application_witness", &reference.application.witness_columns),
        ] {
            if self.port(name)?.source_columns(counts)? != *actual {
                return Err(AssemblyError::Invalid("reference application ports"));
            }
        }
        Ok(())
    }
}
