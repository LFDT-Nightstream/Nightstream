//! Native committed-step relation assembly and measurement.

use super::*;

impl DirectCcsTerminalCommittedRelation {
    pub(crate) fn from_terminal_circuit(circuit: DirectCcsTerminalFPrimeCircuit) -> Result<Self, SimpleKernelError> {
        let assignment = DirectCcsTerminalR2Assignment::from_terminal_circuit(circuit)?;
        let commitment = assignment.commitment()?;
        let fresh_instance = Construction2FreshInstance::from_parts(
            Construction2Commitment::from_commitment(commitment),
            assignment.terminal_circuit.construction2_x_i()?,
        );
        let public_boundary = Construction2PublicBoundary::from_fresh_instance(&fresh_instance);
        Ok(Self {
            public_boundary,
            assignment,
        })
    }

    pub(crate) fn public_boundary(&self) -> &Construction2PublicBoundary {
        &self.public_boundary
    }

    pub(crate) fn committed_circuit(&self) -> DirectCcsTerminalCommittedCircuit {
        DirectCcsTerminalCommittedCircuit {
            public_boundary: self.public_boundary.clone(),
            assignment: self.assignment.clone(),
        }
    }

    pub(crate) fn measure(&self) -> Result<DirectCcsTerminalCommittedPerf, SimpleKernelError> {
        let circuit = self.committed_circuit();
        let public_inputs = circuit
            .public_values()
            .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal committed public IO failed: {err}")))?
            .len();
        let mut cs = ShapeCS::<NeoFoldDeciderEngine>::new();
        let breakdown = circuit
            .measure_with_breakdown(&mut cs)
            .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal committed shape failed: {err}")))?;
        Ok(DirectCcsTerminalCommittedPerf {
            constraints: cs.num_constraints(),
            public_inputs,
            committed_width: self.assignment.committed_width()?,
            commitment_words: self.public_boundary.commitment_data.len(),
            source_values: self.assignment.layout.source_labels.len(),
            source_bit_values: self
                .assignment
                .layout
                .source_count(TerminalPrivateColumnEncoding::Bit),
            source_u32_values: self
                .assignment
                .layout
                .source_count(TerminalPrivateColumnEncoding::U32),
            source_u64_values: self
                .assignment
                .layout
                .source_count(TerminalPrivateColumnEncoding::U64),
            unclassified_private_values: 0,
            breakdown,
            sizes: [0; 10],
            nnz: 0,
        })
    }
}
