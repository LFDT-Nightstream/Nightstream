//! Owns the terminal direct-CCS F' circuit body.
//!
//! The public flow stays deliberately shallow: validate the terminal window,
//! bind the input Construction-2 instance, replay the latest SuperNeo NIFS
//! chunk, bind public output digests, and optionally check the private final CE
//! relation.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};

use crate::construction2::{Construction2EncodedPublicInput, Construction2PublicBoundary};
use crate::spartan_backend::{NeoFoldDeciderEngine, SpartanCircuit, SpartanF};
use crate::superneo_circuit::transcript::Poseidon2TranscriptCircuit;
use crate::superneo_nifs_circuit::{synthesize_superneo_nifs_chunk, SuperNeoClaimBundle};

use super::super::public_image::{DirectCcsIvcPublicImage, DIRECT_CCS_TRIVIAL_PC};
use super::super::state::{DirectCcsFPrimeCircuit, DirectCcsTerminalFPrimeCircuit};
use super::committed::DirectCcsTerminalError;
use super::construction2_fold::synthesize_direct_construction2_fold;
use super::gadgets::{
    digest32_as_spartan_fields, direct_accumulator_digest_circuit_from_claims, enforce_direct_construction2_input_u_i,
    enforce_direct_current_boundary_transition, enforce_direct_public_trace_transition,
    enforce_direct_state_x_in_digest, enforce_direct_state_x_out_public_digest,
    enforce_direct_terminal_final_ce_consistency, field_to_spartan, u64_halves_as_spartan_fields,
};
use super::initial_carry::{alloc_initial_claim_bundle, alloc_initial_transcript};
use super::public_io::{
    alloc_digest_constant, direct_terminal_accumulator_digest_range,
    direct_terminal_construction2_accumulator_digest_range, enforce_digest_eq_constant,
    enforce_digest_fields_public_io, public_digest_input,
};

impl DirectCcsFPrimeCircuit {
    pub(crate) fn terminal_circuit(&self, prove_final_ce: bool) -> DirectCcsTerminalFPrimeCircuit {
        DirectCcsTerminalFPrimeCircuit {
            params: self.params.clone(),
            structure: self.structure.clone(),
            dims: self.dims,
            mat_digest: self.mat_digest,
            vk_fs_digest: self.vk_fs_digest,
            initial_boundary_digest: self.initial_boundary_digest,
            chunks: self.chunks.clone(),
            initial_claims: self.initial_claims.clone(),
            initial_transcript: self.initial_transcript.clone(),
            chunk_count_in: self.chunk_count_in,
            step_count_in: self.step_count_in,
            x_in: self.x_in.clone(),
            construction2_input_u_i: self.construction2_input_u_i.clone(),
            accumulator_in_digest: self.accumulator_in_digest,
            construction2_accumulator_in_digest: self.construction2_accumulator_in_digest,
            public_trace_in_digest: self.public_trace_in_digest,
            current_boundary_in_digest: self.current_boundary_in_digest,
            chunk_count_out: self.chunk_count_out,
            step_count_out: self.step_count_out,
            x_out: self.x_out.clone(),
            accumulator_out_digest: self.accumulator_out_digest,
            construction2_accumulator_out_digest: self.construction2_accumulator_out_digest,
            public_trace_out_digest: self.public_trace_out_digest,
            current_boundary_out_digest: self.current_boundary_out_digest,
            construction2_fold: self.construction2_fold.clone(),
            final_witnesses: self.final_witnesses.clone(),
            prove_final_ce,
        }
    }
}

impl DirectCcsTerminalFPrimeCircuit {
    pub(crate) fn synthesize_body_with_public_inputs<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        public_inputs: &[AllocatedNum<SpartanF>],
    ) -> Result<(), SynthesisError> {
        self.validate_terminal_window()?;
        self.enforce_construction2_input(cs)?;
        let mut transcript = alloc_initial_transcript(cs, self.initial_transcript.as_ref())?;
        let carried = alloc_initial_claim_bundle(cs, &self.initial_claims)?;
        let accumulator_in_digest = self.enforce_input_accumulator_digest(cs, &carried)?;
        self.enforce_input_state_digest(cs, &accumulator_in_digest)?;
        let replay = self.replay_terminal_chunks(cs, &mut transcript, carried, &accumulator_in_digest)?;
        self.enforce_public_outputs(cs, public_inputs, &replay)?;
        self.enforce_final_ce_if_requested(cs, &replay)?;
        Ok(())
    }

    pub(super) fn public_image(&self, construction2_u_i: Construction2PublicBoundary) -> DirectCcsIvcPublicImage {
        DirectCcsIvcPublicImage {
            mat_digest: self.mat_digest,
            vk_fs_digest: self.vk_fs_digest,
            initial_boundary_digest: self.initial_boundary_digest,
            current_boundary_digest: self.current_boundary_out_digest,
            pc: DIRECT_CCS_TRIVIAL_PC,
            chunk_count_out: self.chunk_count_out,
            step_count_out: self.step_count_out,
            x_out: self.x_out.clone(),
            accumulator_out_digest: self.accumulator_out_digest,
            public_trace_out_digest: self.public_trace_out_digest,
            construction2_accumulator_digest: self.construction2_accumulator_out_digest,
            construction2_u_i,
        }
    }

    pub(super) fn terminal_public_values(&self) -> Vec<SpartanF> {
        let mut values = Vec::with_capacity(4 + 2 + 2 + 4 + 4 + 4 + 4 + 256 + 4 + 4 + 4);
        values.extend(self.mat_digest.iter().copied().map(field_to_spartan));
        values.extend(u64_halves_as_spartan_fields(self.chunk_count_out));
        values.extend(u64_halves_as_spartan_fields(self.step_count_out));
        values.extend(digest32_as_spartan_fields(self.vk_fs_digest));
        values.extend(digest32_as_spartan_fields(self.initial_boundary_digest));
        values.extend(digest32_as_spartan_fields(self.current_boundary_out_digest));
        values.extend(digest32_as_spartan_fields(self.x_out.bytes()));
        values.extend(self.x_out.field_image().into_iter().map(field_to_spartan));
        values.extend(digest32_as_spartan_fields(self.accumulator_out_digest));
        values.extend(digest32_as_spartan_fields(self.public_trace_out_digest));
        values.extend(digest32_as_spartan_fields(self.construction2_accumulator_out_digest));
        values
    }

    pub(crate) fn construction2_x_bit_range(&self) -> std::ops::Range<usize> {
        let start = 4 + 2 + 2 + 4 + 4 + 4 + 4;
        start..start + crate::construction2::CONSTRUCTION2_ENC_INST_BITS
    }

    pub(crate) fn construction2_x_i(&self) -> Result<Construction2EncodedPublicInput, DirectCcsTerminalError> {
        if self.chunks.last().is_none() {
            return Err(DirectCcsTerminalError::Bridge(
                "direct CCS terminal F' requires one latest chunk for Construction-2 x_i".into(),
            ));
        }
        Ok(self.x_out.clone())
    }

    fn validate_terminal_window(&self) -> Result<(), SynthesisError> {
        if self.chunk_count_in.checked_add(1) != Some(self.chunk_count_out) || self.step_count_in >= self.step_count_out
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        Ok(())
    }

    fn enforce_construction2_input<CS: ConstraintSystem<SpartanF>>(&self, cs: &mut CS) -> Result<(), SynthesisError> {
        enforce_direct_construction2_input_u_i(
            &mut cs.namespace(|| "direct_terminal_construction2_input_u_i"),
            &self.construction2_input_u_i,
            &self.x_in,
            self.chunk_count_in,
            self.params.kappa as usize,
        )
    }

    fn enforce_input_accumulator_digest<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        carried: &SuperNeoClaimBundle,
    ) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
        let accumulator_in_digest = direct_accumulator_digest_circuit_from_claims(
            &mut cs.namespace(|| "direct_terminal_accumulator_in_digest"),
            &self.params,
            carried.effective_claims(),
        )?;
        enforce_digest_eq_constant(
            &mut cs.namespace(|| "direct_terminal_accumulator_in_digest_private"),
            &accumulator_in_digest,
            self.accumulator_in_digest,
            "direct_terminal_accumulator_in_digest_private",
        )?;
        Ok(accumulator_in_digest)
    }

    fn enforce_input_state_digest<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        accumulator_in_digest: &[AllocatedNum<SpartanF>; 4],
    ) -> Result<(), SynthesisError> {
        let construction2_accumulator_in_digest = alloc_digest_constant(
            &mut cs.namespace(|| "direct_terminal_construction2_accumulator_in_digest"),
            self.construction2_accumulator_in_digest,
            "direct_terminal_construction2_accumulator_in_digest",
        )?;
        enforce_direct_state_x_in_digest(
            &mut cs.namespace(|| "direct_terminal_x_in_digest"),
            self.vk_fs_digest,
            &self.mat_digest,
            self.chunk_count_in,
            self.step_count_in,
            self.initial_boundary_digest,
            self.current_boundary_in_digest,
            DIRECT_CCS_TRIVIAL_PC,
            accumulator_in_digest,
            &construction2_accumulator_in_digest,
            self.public_trace_in_digest,
            self.x_in.bytes(),
            "direct_terminal_x_in_digest",
        )
    }

    fn replay_terminal_chunks<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        transcript: &mut Poseidon2TranscriptCircuit,
        mut carried: SuperNeoClaimBundle,
        accumulator_in_digest: &[AllocatedNum<SpartanF>; 4],
    ) -> Result<TerminalChunkReplay, SynthesisError> {
        let mut last_chunk_digest = None;
        for (chunk_index, chunk) in self.chunks.iter().enumerate() {
            let (next, chunk_digest) = synthesize_superneo_nifs_chunk(
                &self.params,
                &self.structure,
                self.dims,
                &self.mat_digest,
                &mut cs.namespace(|| format!("chunk_{chunk_index}")),
                chunk_index,
                &chunk.cover,
                &chunk.replay,
                transcript,
                carried,
                Some((
                    accumulator_in_digest,
                    digest32_as_spartan_fields(self.accumulator_in_digest),
                )),
            )?;
            transcript.append_message(
                cs.namespace(|| format!("chunk_{chunk_index}_done")),
                b"neo.fold.next/chunk_done",
                &[1],
            )?;
            carried = next;
            last_chunk_digest = Some(chunk_digest);
        }
        Ok(TerminalChunkReplay {
            carried,
            last_chunk_digest: last_chunk_digest.ok_or(SynthesisError::Unsatisfiable)?,
        })
    }

    fn enforce_public_outputs<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        public_inputs: &[AllocatedNum<SpartanF>],
        replay: &TerminalChunkReplay,
    ) -> Result<(), SynthesisError> {
        let accumulator_digest = direct_accumulator_digest_circuit_from_claims(
            &mut cs.namespace(|| "direct_terminal_accumulator_digest"),
            &self.params,
            replay.carried.effective_claims(),
        )?;
        enforce_digest_fields_public_io(
            &mut cs.namespace(|| "direct_terminal_accumulator_digest_public"),
            &accumulator_digest,
            public_inputs,
            direct_terminal_accumulator_digest_range(),
            "direct_terminal_accumulator_digest_public",
        )?;
        let current_boundary_out_digest = enforce_direct_current_boundary_transition(
            &mut cs.namespace(|| "direct_terminal_current_boundary_transition"),
            public_inputs,
            self.current_boundary_in_digest,
            &replay.last_chunk_digest,
        )?;
        let public_trace_out_digest = enforce_direct_public_trace_transition(
            &mut cs.namespace(|| "direct_terminal_public_trace_transition"),
            public_inputs,
            self.public_trace_in_digest,
            &replay.last_chunk_digest,
        )?;
        let construction2_accumulator_out_digest =
            public_digest_input(public_inputs, direct_terminal_construction2_accumulator_digest_range())?;
        synthesize_direct_construction2_fold(
            &mut cs.namespace(|| "direct_terminal_construction2_fold"),
            self.construction2_fold.as_ref(),
            public_inputs,
            self.construction2_accumulator_in_digest,
        )?;
        enforce_direct_state_x_out_public_digest(
            &mut cs.namespace(|| "direct_terminal_x_out_digest"),
            public_inputs,
            self.vk_fs_digest,
            &self.mat_digest,
            self.chunk_count_out,
            self.step_count_out,
            self.initial_boundary_digest,
            &current_boundary_out_digest,
            DIRECT_CCS_TRIVIAL_PC,
            &accumulator_digest,
            &construction2_accumulator_out_digest,
            &public_trace_out_digest,
            "direct_terminal_x_out_digest",
        )
    }

    fn enforce_final_ce_if_requested<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        replay: &TerminalChunkReplay,
    ) -> Result<(), SynthesisError> {
        if !self.prove_final_ce {
            return Ok(());
        }
        enforce_direct_terminal_final_ce_consistency(
            &mut cs.namespace(|| "direct_terminal_final_ce"),
            &self.params,
            &self.structure,
            replay.carried.effective_claims(),
            &self.final_witnesses,
        )
    }
}

struct TerminalChunkReplay {
    carried: SuperNeoClaimBundle,
    last_chunk_digest: [AllocatedNum<SpartanF>; 4],
}

impl SpartanCircuit<NeoFoldDeciderEngine> for DirectCcsTerminalFPrimeCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(self.terminal_public_values())
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        let public_values = self.public_values()?;
        let public_inputs = public_values
            .into_iter()
            .enumerate()
            .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_{idx}")), || Ok(value)))
            .collect::<Result<Vec<_>, _>>()?;
        self.synthesize_body_with_public_inputs(cs, &public_inputs)
    }
}
