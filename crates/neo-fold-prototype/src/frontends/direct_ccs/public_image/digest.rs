//! Poseidon2 digest helpers for the direct-CCS public image.

use neo_math::F;
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::construction2::Construction2EncodedPublicInput;
use crate::finalize::{digest32_as_fields, digest_fields_as_digest32};
use crate::spartan_backend::SpartanF;

pub(crate) fn direct_state_x_out(
    vk_fs_digest: [u8; 32],
    mat_digest: &[F; 4],
    chunk_count: u64,
    step_count: u64,
    initial_boundary_digest: [u8; 32],
    current_boundary_digest: [u8; 32],
    pc: u64,
    semantic_accumulator_digest: [u8; 32],
    construction2_accumulator_digest: [u8; 32],
    public_trace_digest: [u8; 32],
) -> Construction2EncodedPublicInput {
    Construction2EncodedPublicInput::from_digest_bytes(direct_state_image_digest(
        vk_fs_digest,
        mat_digest,
        chunk_count,
        step_count,
        initial_boundary_digest,
        current_boundary_digest,
        pc,
        semantic_accumulator_digest,
        construction2_accumulator_digest,
        public_trace_digest,
    ))
}

fn direct_state_image_digest(
    vk_fs_digest: [u8; 32],
    mat_digest: &[F; 4],
    chunk_count: u64,
    step_count: u64,
    initial_boundary_digest: [u8; 32],
    current_boundary_digest: [u8; 32],
    pc: u64,
    semantic_accumulator_digest: [u8; 32],
    construction2_accumulator_digest: [u8; 32],
    public_trace_digest: [u8; 32],
) -> [u8; 32] {
    let mut preimage = direct_domain_fields(b"neo.fold.next/direct_ccs/f_prime_x_out/v2");
    preimage.extend(digest32_as_fields(vk_fs_digest));
    preimage.extend(mat_digest.iter().copied());
    preimage.extend(u64_halves_as_native_fields(chunk_count));
    preimage.extend(u64_halves_as_native_fields(step_count));
    preimage.extend(digest32_as_fields(initial_boundary_digest));
    preimage.extend(digest32_as_fields(current_boundary_digest));
    preimage.extend(u64_halves_as_native_fields(pc));
    preimage.extend(digest32_as_fields(semantic_accumulator_digest));
    preimage.extend(digest32_as_fields(construction2_accumulator_digest));
    preimage.extend(digest32_as_fields(public_trace_digest));
    digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

pub(crate) fn direct_vk_fs_digest(
    params: &NeoParams,
    mat_digest: &[F; 4],
    public_input_len: Option<usize>,
) -> [u8; 32] {
    let mut preimage = direct_domain_fields(b"neo.fold.next/direct_ccs/vk_fs/v1");
    preimage.extend(mat_digest.iter().copied());
    preimage.extend([
        F::from_u64(params.q),
        F::from_u64(params.eta as u64),
        F::from_u64(params.d as u64),
        F::from_u64(params.kappa as u64),
        F::from_u64(params.m),
        F::from_u64(params.b as u64),
        F::from_u64(params.k_rho as u64),
        F::from_u64(params.B),
        F::from_u64(params.T as u64),
        F::from_u64(params.s as u64),
        F::from_u64(params.lambda as u64),
        F::from_u64(public_input_len.map_or(u64::MAX, |len| len as u64)),
    ]);
    digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

pub(crate) fn direct_initial_boundary_digest(mat_digest: &[F; 4], public_input_len: Option<usize>) -> [u8; 32] {
    let mut preimage = direct_domain_fields(b"neo.fold.next/direct_ccs/initial_boundary/v1");
    preimage.extend(mat_digest.iter().copied());
    preimage.push(F::from_u64(public_input_len.map_or(u64::MAX, |len| len as u64)));
    digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

pub(crate) fn direct_boundary_update_digest(boundary_digest: [u8; 32], current_chunk_digest: [F; 4]) -> [u8; 32] {
    let mut preimage = direct_domain_fields(b"neo.fold.next/direct_ccs/current_boundary_update/v1");
    preimage.extend(digest32_as_fields(boundary_digest));
    preimage.extend(current_chunk_digest);
    digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

pub(crate) fn direct_public_trace_seed_digest(mat_digest: &[F; 4]) -> [u8; 32] {
    let mut preimage = direct_domain_fields(b"neo.fold.next/direct_ccs/public_trace_seed/v1");
    preimage.extend(mat_digest.iter().copied());
    digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

pub(crate) fn direct_public_trace_update_digest(
    public_trace_digest: [u8; 32],
    current_chunk_digest: [F; 4],
) -> [u8; 32] {
    let mut preimage = direct_domain_fields(b"neo.fold.next/direct_ccs/public_trace_update/v1");
    preimage.extend(digest32_as_fields(public_trace_digest));
    preimage.extend(current_chunk_digest);
    digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

fn direct_domain_fields(domain: &[u8]) -> Vec<F> {
    crate::superneo_circuit::claim::packed_bytes_field_values(domain)
        .into_iter()
        .map(|value| F::from_u64(value.to_canonical_u64()))
        .collect()
}

pub(super) fn digest32_as_spartan_fields(digest: [u8; 32]) -> [SpartanF; 4] {
    digest32_as_fields(digest).map(field_to_spartan)
}

fn u64_halves_as_native_fields(value: u64) -> [F; 2] {
    [F::from_u64(value & 0xffff_ffff), F::from_u64(value >> 32)]
}

pub(super) fn u64_halves_as_spartan_fields(value: u64) -> [SpartanF; 2] {
    [
        SpartanF::from_canonical_u64(value & 0xffff_ffff),
        SpartanF::from_canonical_u64(value >> 32),
    ]
}

pub(super) fn field_to_spartan(value: F) -> SpartanF {
    SpartanF::from_canonical_u64(value.as_canonical_u64())
}

pub(super) fn digest32_has_canonical_field_limb_bytes(digest: [u8; 32]) -> bool {
    digest.chunks_exact(8).all(|chunk| {
        let limb = u64::from_le_bytes(chunk.try_into().expect("digest32 has 8-byte limbs"));
        F::from_u64(limb).as_canonical_u64() == limb
    })
}
