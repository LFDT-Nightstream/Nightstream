use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectCcsFPrimeLowNormSourceR1csShape {
    pub public_input_len: usize,
    pub variable_count: usize,
    pub constraint_count: usize,
    pub nonzero_entries: usize,
    pub bit_constraints: usize,
    pub x_out_link_constraints: usize,
    pub construction2_boundary_link_constraints: usize,
    pub construction2_instance_digest_link_constraints: usize,
    pub construction2_commitment_shape_constraints: usize,
    pub structural_counter_constraints: usize,
    pub structural_fixed_arity_constraints: usize,
    pub structural_counter_carry_bit_constraints: usize,
    pub canonical_field_lane_constraints: usize,
    pub canonical_field_lane_aux_bits: usize,
    pub canonical_field_lane_count: usize,
    pub poseidon_digest_recomputation_aux_bits: usize,
    pub poseidon_digest_recomputation_constraints: usize,
    pub nifs_v_verifier_constraints: usize,
    pub source_len: usize,
    pub counter_carry_bits: usize,
}

impl DirectCcsFPrimeLowNormSourceR1csShape {
    pub fn from_source(source: &DirectCcsFPrimeLowNormSourceImage) -> Self {
        Self::from_source_metadata(source.len(), source.field_lane_count(), 0, 0)
    }

    pub fn from_source_with_authority_estimate(source: &DirectCcsFPrimeLowNormSourceImage) -> Self {
        let (aux_bits, rows) = estimated_poseidon_digest_recomputation_cost();
        Self::from_source_metadata(source.len(), source.field_lane_count(), aux_bits, rows)
    }

    pub(super) fn from_source_metadata(
        source_len: usize,
        canonical_field_lane_count: usize,
        poseidon_digest_recomputation_aux_bits: usize,
        poseidon_digest_recomputation_constraints: usize,
    ) -> Self {
        let public_input_len = 1 + CONSTRUCTION2_ENC_INST_BITS;
        let counter_carry_bits = STRUCTURAL_COUNTER_CARRY_BITS;
        let canonical_field_lane_aux_bits = canonical_field_lane_count * GOLDILOCKS_CANONICAL_AUX_BITS_PER_LANE;
        let bit_constraints = CONSTRUCTION2_ENC_INST_BITS
            + source_len
            + counter_carry_bits
            + canonical_field_lane_aux_bits
            + poseidon_digest_recomputation_aux_bits;
        let x_out_link_constraints = CONSTRUCTION2_ENC_INST_BITS;
        let construction2_boundary_link_constraints = 2 * CONSTRUCTION2_ENC_INST_BITS;
        let construction2_instance_digest_link_constraints = 2 * CONSTRUCTION2_ENC_INST_BITS;
        let construction2_commitment_shape_constraints = 4 * 64;
        let structural_counter_constraints = STRUCTURAL_COUNTER_CONSTRAINTS;
        let structural_fixed_arity_constraints = STRUCTURAL_FIXED_ARITY_CONSTRAINTS;
        let structural_counter_carry_bit_constraints = counter_carry_bits;
        let canonical_field_lane_constraints = canonical_field_lane_count * GOLDILOCKS_CANONICAL_CONSTRAINTS_PER_LANE;
        let nifs_v_verifier_constraints = 0;
        let link_constraints = x_out_link_constraints
            + construction2_boundary_link_constraints
            + construction2_instance_digest_link_constraints
            + construction2_commitment_shape_constraints
            + structural_counter_constraints
            + canonical_field_lane_constraints;
        Self {
            public_input_len,
            variable_count: public_input_len
                + source_len
                + counter_carry_bits
                + canonical_field_lane_aux_bits
                + poseidon_digest_recomputation_aux_bits,
            constraint_count: bit_constraints
                + link_constraints
                + poseidon_digest_recomputation_constraints
                + nifs_v_verifier_constraints,
            nonzero_entries: bit_constraints * 3 + link_constraints * 3,
            bit_constraints,
            x_out_link_constraints,
            construction2_boundary_link_constraints,
            construction2_instance_digest_link_constraints,
            construction2_commitment_shape_constraints,
            structural_counter_constraints,
            structural_fixed_arity_constraints,
            structural_counter_carry_bit_constraints,
            canonical_field_lane_constraints,
            canonical_field_lane_aux_bits,
            canonical_field_lane_count,
            poseidon_digest_recomputation_aux_bits,
            poseidon_digest_recomputation_constraints,
            nifs_v_verifier_constraints,
            source_len,
            counter_carry_bits,
        }
    }

    pub fn shell_constraints(self) -> usize {
        self.bit_constraints
            + self.x_out_link_constraints
            + self.construction2_boundary_link_constraints
            + self.construction2_instance_digest_link_constraints
            + self.construction2_commitment_shape_constraints
            + self.structural_counter_constraints
            + self.canonical_field_lane_constraints
    }

    pub fn digest_binding_constraints(self) -> usize {
        self.poseidon_digest_recomputation_constraints
    }

    pub fn authority_constraints(self) -> usize {
        if self.has_proof_authority() {
            self.digest_binding_constraints() + self.nifs_v_verifier_constraints
        } else {
            0
        }
    }

    pub fn has_proof_authority(self) -> bool {
        self.poseidon_digest_recomputation_constraints > 0 && self.nifs_v_verifier_constraints > 0
    }
}
