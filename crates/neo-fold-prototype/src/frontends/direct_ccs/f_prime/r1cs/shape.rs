use super::nifs_authority::NIFS_AUTHORITY_ROWS;
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectCcsFPrimeLowNormSourceR1csShape {
    pub public_input_len: usize,
    pub variable_count: usize,
    pub constraint_count: usize,
    pub nonzero_entries: usize,
    pub source: DirectCcsFPrimeLowNormSourceImageShape,
    pub variables: DirectCcsFPrimeLowNormSourceVariableShape,
    pub constraints: DirectCcsFPrimeLowNormSourceConstraintShape,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectCcsFPrimeLowNormSourceImageShape {
    pub private_bits: usize,
    pub canonical_field_lanes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectCcsFPrimeLowNormSourceVariableShape {
    pub counter_carry_bits: usize,
    pub canonical_field_lane_aux_bits: usize,
    pub poseidon_digest_recomputation_aux_bits: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectCcsFPrimeLowNormSourceConstraintShape {
    pub bitness: usize,
    pub x_out_link: usize,
    pub construction2_boundary_link: usize,
    pub construction2_instance_digest_link: usize,
    pub construction2_commitment_shape: usize,
    pub structural_counter: usize,
    pub structural_fixed_arity: usize,
    pub structural_counter_carry_bitness: usize,
    pub canonical_field_lane: usize,
    pub poseidon_digest_recomputation: usize,
    pub nifs_v_verifier: usize,
}

impl DirectCcsFPrimeLowNormSourceR1csShape {
    pub fn from_source(source: &DirectCcsFPrimeLowNormSourceImage) -> Self {
        Self::from_source_metadata(source.len(), source.field_lane_count(), 0, 0, 0)
    }

    pub fn from_source_with_authority_estimate(source: &DirectCcsFPrimeLowNormSourceImage) -> Self {
        let (aux_bits, rows) = estimated_poseidon_digest_recomputation_cost();
        Self::from_source_metadata(
            source.len(),
            source.field_lane_count(),
            aux_bits,
            rows,
            NIFS_AUTHORITY_ROWS,
        )
    }

    pub(super) fn from_source_metadata(
        source_len: usize,
        canonical_field_lane_count: usize,
        poseidon_digest_recomputation_aux_bits: usize,
        poseidon_digest_recomputation_constraints: usize,
        nifs_v_verifier_constraints: usize,
    ) -> Self {
        let public_input_len = 1 + CONSTRUCTION2_ENC_INST_BITS;
        let variables = DirectCcsFPrimeLowNormSourceVariableShape {
            counter_carry_bits: STRUCTURAL_COUNTER_CARRY_BITS,
            canonical_field_lane_aux_bits: canonical_field_lane_count * GOLDILOCKS_CANONICAL_AUX_BITS_PER_LANE,
            poseidon_digest_recomputation_aux_bits,
        };
        let bit_constraints = source_len
            + variables.counter_carry_bits
            + variables.canonical_field_lane_aux_bits
            + poseidon_digest_recomputation_aux_bits;
        let constraints = DirectCcsFPrimeLowNormSourceConstraintShape {
            bitness: bit_constraints,
            x_out_link: CONSTRUCTION2_ENC_INST_BITS,
            construction2_boundary_link: 2 * CONSTRUCTION2_ENC_INST_BITS,
            construction2_instance_digest_link: 2 * CONSTRUCTION2_ENC_INST_BITS,
            construction2_commitment_shape: 4 * 64,
            structural_counter: STRUCTURAL_COUNTER_CONSTRAINTS,
            structural_fixed_arity: STRUCTURAL_FIXED_ARITY_CONSTRAINTS,
            structural_counter_carry_bitness: variables.counter_carry_bits,
            canonical_field_lane: canonical_field_lane_count * GOLDILOCKS_CANONICAL_CONSTRAINTS_PER_LANE,
            poseidon_digest_recomputation: poseidon_digest_recomputation_constraints,
            nifs_v_verifier: nifs_v_verifier_constraints,
        };
        let link_constraints = constraints.x_out_link
            + constraints.construction2_boundary_link
            + constraints.construction2_instance_digest_link
            + constraints.construction2_commitment_shape
            + constraints.structural_counter
            + constraints.canonical_field_lane;
        Self {
            public_input_len,
            variable_count: public_input_len
                + source_len
                + variables.counter_carry_bits
                + variables.canonical_field_lane_aux_bits
                + variables.poseidon_digest_recomputation_aux_bits,
            constraint_count: constraints.bitness
                + link_constraints
                + constraints.poseidon_digest_recomputation
                + constraints.nifs_v_verifier,
            nonzero_entries: constraints.bitness * 3 + link_constraints * 3,
            source: DirectCcsFPrimeLowNormSourceImageShape {
                private_bits: source_len,
                canonical_field_lanes: canonical_field_lane_count,
            },
            variables,
            constraints,
        }
    }

    pub fn shell_constraints(self) -> usize {
        self.constraints.bitness
            + self.constraints.x_out_link
            + self.constraints.construction2_boundary_link
            + self.constraints.construction2_instance_digest_link
            + self.constraints.construction2_commitment_shape
            + self.constraints.structural_counter
            + self.constraints.canonical_field_lane
    }

    pub fn digest_binding_constraints(self) -> usize {
        self.constraints.poseidon_digest_recomputation
    }

    pub fn authority_constraints(self) -> usize {
        if self.has_proof_authority() {
            self.digest_binding_constraints() + self.constraints.nifs_v_verifier
        } else {
            0
        }
    }

    pub fn has_proof_authority(self) -> bool {
        self.constraints.poseidon_digest_recomputation > 0 && self.constraints.nifs_v_verifier > 0
    }
}
