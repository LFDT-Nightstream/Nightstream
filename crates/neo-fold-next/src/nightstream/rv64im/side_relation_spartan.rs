//! Owns the RV64IM side binding proof.
//!
//! The circuit binds the carried Nightstream statement core digest to the
//! canonical opened-object/eval public tuple digest and proves the carried
//! Phase 0 opened-object witnesses against the public eval claims. It does not
//! own the compact selected-opening theorem or outer Nightstream linkage checks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use bellpepper_core::{num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};
use spartan2::{
    provider::{goldi::F as SpartanF, GoldilocksP3MerkleMleEngine},
    spartan::R1CSSNARK,
    traits::{circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};

use super::authoritative_side::Rv64imSideBindingStatement;
use super::authoritative_side::Rv64imSideOpeningPublic;
use super::side_relation_circuit::phase0;
use crate::finalize::digest32_as_fields;
use crate::rv64im::kernel::{
    phase0_full_width_for_schema, FamilyEvalClaimWitness, OpenedAjtaiObjectWitness, PackedColumnOracleRef,
    SimpleKernelError,
};
use crate::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;

pub type Rv64imSideBindingEngine = GoldilocksP3MerkleMleEngine;
pub type Rv64imSideBindingSnark = R1CSSNARK<Rv64imSideBindingEngine>;
pub type Rv64imSideBindingProverKey = spartan2::spartan::SpartanProverKey<Rv64imSideBindingEngine>;
pub type Rv64imSideBindingVerifierKey = spartan2::spartan::SpartanVerifierKey<Rv64imSideBindingEngine>;

type Rv64imSideBindingKeyPair = Arc<(Rv64imSideBindingProverKey, Rv64imSideBindingVerifierKey)>;

static RV64IM_SIDE_BINDING_SETUP_CACHE: OnceLock<Mutex<HashMap<[u8; 32], Rv64imSideBindingKeyPair>>> = OnceLock::new();

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imSideBindingProof {
    pub snark_data: Vec<u8>,
}

impl Rv64imSideBindingProof {
    pub fn snark_bytes_len(&self) -> usize {
        self.snark_data.len()
    }
}

#[derive(Clone)]
struct Rv64imSideBindingCircuit {
    statement: Rv64imSideBindingStatement,
    public: Rv64imSideOpeningPublic,
    opened_object_witnesses: Vec<Arc<OpenedAjtaiObjectWitness>>,
}

impl Rv64imSideBindingCircuit {
    fn expected_public_values(&self) -> Vec<SpartanF> {
        digest32_as_fields(self.statement.digest())
            .into_iter()
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
            .collect()
    }

    fn dummy(
        statement: &Rv64imSideBindingStatement,
        public: &Rv64imSideOpeningPublic,
    ) -> Result<Self, SimpleKernelError> {
        Ok(Self {
            statement: statement.clone(),
            public: public.clone(),
            opened_object_witnesses: build_dummy_opened_object_witnesses(public)?,
        })
    }

    fn from_claim_witnesses(
        statement: &Rv64imSideBindingStatement,
        public: &Rv64imSideOpeningPublic,
        claim_witnesses: &[FamilyEvalClaimWitness],
    ) -> Result<Self, SimpleKernelError> {
        Ok(Self {
            statement: statement.clone(),
            public: public.clone(),
            opened_object_witnesses: build_opened_object_witnesses_from_claim_witnesses(public, claim_witnesses)?,
        })
    }
}

impl SpartanCircuit<Rv64imSideBindingEngine> for Rv64imSideBindingCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(self.expected_public_values())
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
        let public_inputs = self
            .expected_public_values()
            .into_iter()
            .enumerate()
            .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_input_{idx}")), || Ok(value)))
            .collect::<Result<Vec<_>, _>>()?;

        let core_digest_fields = alloc_digest_fields_witness(
            &mut cs.namespace(|| "statement_core_digest_fields"),
            self.statement.nightstream_statement_core_digest,
            "statement_core_digest_field",
        )?;
        let opened_object_digests = self
            .public
            .opened_objects
            .iter()
            .enumerate()
            .map(|(idx, opened_object)| {
                alloc_packed_bytes_witness(
                    &mut cs.namespace(|| format!("opened_object_digest_{idx}")),
                    &opened_object.digest,
                    &format!("opened_object_digest_{idx}"),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let eval_digests = self
            .public
            .evals
            .iter()
            .enumerate()
            .map(|(idx, eval)| {
                alloc_packed_bytes_witness(
                    &mut cs.namespace(|| format!("eval_digest_{idx}")),
                    &eval.digest,
                    &format!("eval_digest_{idx}"),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        if self.opened_object_witnesses.len() != self.public.evals.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        for (idx, (eval, witness)) in self
            .public
            .evals
            .iter()
            .zip(self.opened_object_witnesses.iter())
            .enumerate()
        {
            let claim = &eval.claim;
            if claim.payload.column_evals.len() != witness.packed_columns.len()
                || witness.packed_columns.len() != witness.commitment_vector.len()
                || claim.point.len() != witness.row_domain_log_size as usize
                || claim.commitment_context != witness.commitment_context
            {
                return Err(SynthesisError::Unsatisfiable);
            }

            let (opened_object_digest_vars, packed_column_matrix_entries) =
                phase0::enforce_commitment_root_and_opened_object_digest(
                    cs.namespace(|| format!("phase0_commitment_{idx}")),
                    claim.payload.schema,
                    &claim.commitment_context,
                    &witness.packed_columns,
                    &witness.commitment_vector,
                    &claim.opened_object,
                    &format!("phase0_commitment_{idx}"),
                )?;
            let binding_digest_vars = alloc_digest32_witness_array(
                &mut cs.namespace(|| format!("phase0_binding_digest_{idx}")),
                claim.binding_digest,
                &format!("phase0_binding_digest_{idx}"),
            )?;
            let (point_vars, point_values) = phase0::derive_phase0_point(
                cs.namespace(|| format!("phase0_point_{idx}")),
                &opened_object_digest_vars,
                claim.opened_object.digest,
                &claim.commitment_context,
                claim.payload.schema,
                claim.id.slot,
                &binding_digest_vars,
                claim.binding_digest,
                claim.opened_object.row_domain_log_size as usize,
                &format!("phase0_point_{idx}"),
            )?;
            phase0::enforce_point_eq(
                &mut cs.namespace(|| format!("phase0_point_eq_{idx}")),
                &point_vars,
                &claim.point,
                &format!("phase0_point_eq_{idx}"),
            )?;
            let (payload_vars, _) = phase0::evaluate_payload_from_packed_rows(
                &mut cs.namespace(|| format!("phase0_payload_{idx}")),
                &witness.packed_columns,
                &packed_column_matrix_entries,
                &point_vars,
                &point_values,
                &format!("phase0_payload_{idx}"),
            )?;
            phase0::enforce_payload_eq(
                &mut cs.namespace(|| format!("phase0_payload_eq_{idx}")),
                &payload_vars,
                &expected_payload_coeffs(claim),
                &format!("phase0_payload_eq_{idx}"),
            )?;
        }

        let mut tr = Poseidon2TranscriptCircuit::new(
            cs.namespace(|| "public_instance_tr"),
            b"neo.fold.next/nightstream/rv64im/authoritative_side/public_instance",
        )?;
        tr.append_u64s(
            cs.namespace(|| "public_instance_counts"),
            b"neo.fold.next/nightstream/rv64im/authoritative_side/public_instance/counts",
            &[self.public.opened_objects.len() as u64, self.public.evals.len() as u64],
        )?;
        for (idx, digest) in opened_object_digests.iter().enumerate() {
            tr.append_packed_bytes(
                cs.namespace(|| format!("public_instance_opened_object_digest_{idx}")),
                b"neo.fold.next/nightstream/rv64im/authoritative_side/public_instance/opened_object_digest",
                &digest.0,
                &digest.1,
                self.public.opened_objects[idx].digest.len(),
            )?;
        }
        for (idx, digest) in eval_digests.iter().enumerate() {
            tr.append_packed_bytes(
                cs.namespace(|| format!("public_instance_eval_digest_{idx}")),
                b"neo.fold.next/nightstream/rv64im/authoritative_side/public_instance/eval_digest",
                &digest.0,
                &digest.1,
                self.public.evals[idx].digest.len(),
            )?;
        }
        let public_instance_digest = tr.digest32(cs.namespace(|| "public_instance_digest"))?;
        let mut statement_tr = Poseidon2TranscriptCircuit::new(
            cs.namespace(|| "statement_tr"),
            b"neo.fold.next/nightstream/rv64im/authoritative_side/statement",
        )?;
        statement_tr.append_message(
            cs.namespace(|| "statement_version"),
            b"neo.fold.next/nightstream/rv64im/authoritative_side/statement/version",
            b"v1",
        )?;
        statement_tr.append_fields(
            cs.namespace(|| "statement_tr_core_digest"),
            b"neo.fold.next/nightstream/rv64im/authoritative_side/statement/nightstream_statement_core_digest",
            &core_digest_fields.0,
            &core_digest_fields.1,
        )?;
        let public_instance_digest_values = digest32_as_fields(self.statement.public_instance_digest)
            .into_iter()
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
            .collect::<Vec<_>>();
        statement_tr.append_fields(
            cs.namespace(|| "statement_public_instance_digest"),
            b"neo.fold.next/nightstream/rv64im/authoritative_side/statement/public_instance_digest",
            &public_instance_digest,
            &public_instance_digest_values,
        )?;
        let statement_digest = statement_tr.digest32(cs.namespace(|| "statement_digest"))?;
        enforce_digest_eq_public_inputs(
            &mut cs.namespace(|| "statement_digest_match"),
            &statement_digest,
            &public_inputs,
        )?;
        Ok(())
    }
}

pub fn setup_rv64im_side_binding(
    statement: &Rv64imSideBindingStatement,
    public: &Rv64imSideOpeningPublic,
) -> Result<(Rv64imSideBindingProverKey, Rv64imSideBindingVerifierKey), SimpleKernelError> {
    Rv64imSideBindingSnark::setup(Rv64imSideBindingCircuit::dummy(statement, public)?)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side binding setup failed: {err}")))
}

pub fn setup_rv64im_side_binding_cached(
    statement: &Rv64imSideBindingStatement,
    public: &Rv64imSideOpeningPublic,
) -> Result<Rv64imSideBindingKeyPair, SimpleKernelError> {
    let shape_digest = rv64im_side_binding_shape_digest(public);
    let cache = RV64IM_SIDE_BINDING_SETUP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(keys) = cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM side binding setup cache poisoned".into()))?
        .get(&shape_digest)
        .cloned()
    {
        return Ok(keys);
    }

    let keys = Arc::new(setup_rv64im_side_binding(statement, public)?);
    cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM side binding setup cache poisoned".into()))?
        .insert(shape_digest, keys.clone());
    Ok(keys)
}

pub fn prove_rv64im_side_binding(
    pk: &Rv64imSideBindingProverKey,
    statement: &Rv64imSideBindingStatement,
    public: &Rv64imSideOpeningPublic,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Rv64imSideBindingProof, SimpleKernelError> {
    let circuit = Rv64imSideBindingCircuit::from_claim_witnesses(statement, public, claim_witnesses)?;
    let prep = Rv64imSideBindingSnark::prep_prove(pk, circuit.clone(), true)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side binding prepare failed: {err}")))?;
    let proof = Rv64imSideBindingSnark::prove(pk, circuit, &prep, true)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side binding prove failed: {err}")))?;
    let snark_data = bincode::serialize(&proof)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side binding encode failed: {err}")))?;
    Ok(Rv64imSideBindingProof { snark_data })
}

pub fn verify_rv64im_side_binding(
    vk: &Rv64imSideBindingVerifierKey,
    statement: &Rv64imSideBindingStatement,
    proof: &Rv64imSideBindingProof,
) -> Result<(), SimpleKernelError> {
    let snark: Rv64imSideBindingSnark = bincode::deserialize(&proof.snark_data)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side binding decode failed: {err}")))?;
    let public_values = snark
        .verify(vk)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side binding verify failed: {err}")))?;
    verify_side_public_io(statement, &public_values)
}

pub fn debug_check_rv64im_side_binding_circuit(
    statement: &Rv64imSideBindingStatement,
    public: &Rv64imSideOpeningPublic,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<(), SimpleKernelError> {
    let circuit = Rv64imSideBindingCircuit::from_claim_witnesses(statement, public, claim_witnesses)?;
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit
        .synthesize(&mut cs, &[], &[], None)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side binding debug synthesis failed: {err}")))?;
    if !cs.is_satisfied() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM side binding circuit unsatisfied: {}",
            cs.which_is_unsatisfied()
                .unwrap_or_else(|| "unknown".into())
        )));
    }
    Ok(())
}

pub fn measure_rv64im_side_binding_circuit_constraints(
    statement: &Rv64imSideBindingStatement,
    public: &Rv64imSideOpeningPublic,
) -> Result<usize, SimpleKernelError> {
    let circuit = Rv64imSideBindingCircuit::dummy(statement, public)?;
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit
        .synthesize(&mut cs, &[], &[], None)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side binding counting synthesis failed: {err}")))?;
    Ok(cs.num_constraints())
}

fn verify_side_public_io(
    statement: &Rv64imSideBindingStatement,
    public_values: &[SpartanF],
) -> Result<(), SimpleKernelError> {
    let expected = digest32_as_fields(statement.digest())
        .into_iter()
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
        .collect::<Vec<_>>();
    if expected != public_values {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side binding public IO mismatch".into(),
        ));
    }
    Ok(())
}

fn expected_payload_coeffs(claim: &crate::rv64im::kernel::FamilyEvalClaim) -> Vec<Vec<K>> {
    claim
        .payload
        .column_evals
        .iter()
        .map(|column| column.coeffs.to_vec())
        .collect()
}

fn build_dummy_opened_object_witnesses(
    public: &Rv64imSideOpeningPublic,
) -> Result<Vec<Arc<OpenedAjtaiObjectWitness>>, SimpleKernelError> {
    public
        .evals
        .iter()
        .map(|eval| build_dummy_opened_object_witness(&eval.claim))
        .collect()
}

fn build_dummy_opened_object_witness(
    claim: &crate::rv64im::kernel::FamilyEvalClaim,
) -> Result<Arc<OpenedAjtaiObjectWitness>, SimpleKernelError> {
    let params =
        NeoParams::goldilocks_auto_r1cs_ccs(phase0_full_width_for_schema(claim.payload.schema)).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM side binding could not derive Phase 0 dummy parameters for {:?}: {err}",
                claim.payload.schema
            ))
        })?;
    let row_len = 1usize << (claim.opened_object.row_domain_log_size as usize);
    let packed_column_count = claim.payload.column_evals.len();
    let packed_columns = (0..packed_column_count)
        .map(|column_index| PackedColumnOracleRef {
            column_index: column_index as u32,
            rows: vec![[F::ZERO; D]; row_len],
        })
        .collect::<Vec<_>>();
    let commitment_vector = (0..packed_column_count)
        .map(|_| Commitment::zeros(D, params.kappa as usize))
        .collect::<Vec<_>>();
    Ok(Arc::new(OpenedAjtaiObjectWitness {
        opened_object: claim.opened_object.clone(),
        commitment_context: claim.commitment_context.clone(),
        row_domain_log_size: claim.opened_object.row_domain_log_size,
        packed_column_count: packed_column_count as u32,
        packed_columns,
        commitment_vector,
    }))
}

fn build_opened_object_witnesses_from_claim_witnesses(
    public: &Rv64imSideOpeningPublic,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Vec<Arc<OpenedAjtaiObjectWitness>>, SimpleKernelError> {
    if claim_witnesses.len() != public.evals.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side binding prove path claim-witness count does not match the carried public eval set".into(),
        ));
    }
    public
        .evals
        .iter()
        .zip(claim_witnesses.iter())
        .map(|(eval, claim_witness)| {
            if eval.claim != claim_witness.claim {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV64IM side binding prove path {:?}/{} claim does not match the carried public eval",
                    eval.claim.payload.schema, eval.claim.id.slot
                )));
            }
            Ok(claim_witness.witness.clone())
        })
        .collect()
}

fn rv64im_side_binding_shape_digest(public: &Rv64imSideOpeningPublic) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/side_binding_shape");
    tr.append_u64s(
        b"neo.fold.next/nightstream/rv64im/side_binding_shape/counts",
        &[public.opened_objects.len() as u64, public.evals.len() as u64],
    );
    for eval in &public.evals {
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/side_binding_shape/eval",
            &[
                eval.claim.payload.schema.tag(),
                eval.claim.point.len() as u64,
                eval.claim.payload.column_evals.len() as u64,
            ],
        );
    }
    tr.digest32()
}

fn alloc_packed_bytes_witness<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    bytes: &[u8],
    label: &str,
) -> Result<(Vec<AllocatedNum<SpartanF>>, Vec<SpartanF>), SynthesisError> {
    let values = pack_bytes_without_len(bytes);
    let vars = values
        .iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || Ok(*value)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((vars, values))
}

fn alloc_digest_fields_witness<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    digest: [u8; 32],
    label: &str,
) -> Result<(Vec<AllocatedNum<SpartanF>>, Vec<SpartanF>), SynthesisError> {
    let values = digest32_as_fields(digest)
        .into_iter()
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
        .collect::<Vec<_>>();
    let vars = values
        .iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || Ok(*value)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((vars, values))
}

fn alloc_digest32_witness_array<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    digest: [u8; 32],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let values = digest32_as_fields(digest).map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    Ok(core::array::from_fn(|idx| {
        AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || Ok(values[idx]))
            .expect("digest witness allocation must succeed")
    }))
}

fn pack_bytes_without_len(bytes: &[u8]) -> Vec<SpartanF> {
    const BYTES_PER_LIMB: usize = 7;
    let mut packed = Vec::with_capacity(bytes.len().div_ceil(BYTES_PER_LIMB));
    for chunk in bytes.chunks(BYTES_PER_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        packed.push(SpartanF::from_canonical_u64(u64::from_le_bytes(limb)));
    }
    packed
}

fn enforce_digest_eq_public_inputs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[AllocatedNum<SpartanF>; 4],
    expected: &[AllocatedNum<SpartanF>],
) -> Result<(), SynthesisError> {
    if expected.len() != actual.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        cs.enforce(
            || format!("statement_digest_{idx}"),
            |lc| lc + actual.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + expected.get_variable(),
        );
    }
    Ok(())
}
