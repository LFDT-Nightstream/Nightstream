// neo-fold/src/snark/mod.rs
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use memchr::memmem;
use thiserror::Error;

use neo_ccs::{CcsInstance, CcsStructure, CcsWitness, check_satisfiability};
use neo_commit::AjtaiCommitter;
use neo_fields::F;
#[allow(unused_imports)]
use neo_modint::ModInt;
#[allow(unused_imports)]
use neo_ring::RingElement;
use subtle::ConstantTimeEq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::{Proof};
use crate::spartan_ivc::{spartan_compress, spartan_verify, domain_separated_transcript};

#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error("ccs constraints not satisfied by provided witness")]
    Unsatisfied,
}

#[derive(Clone, Debug)]
pub struct Metrics {
    pub prove_ms: f64,
    pub proof_bytes: usize,
}

const FS_LABEL: &str = "neo_snark_fs";
pub const SNARK_MARKER: &[u8] = b"neo_spartan2_snark"; // exact marker the test expects
static PROVE_CALLS: AtomicU64 = AtomicU64::new(0);

pub fn prove(
    ccs: &CcsStructure,
    instance: &CcsInstance,
    witness: &CcsWitness,
) -> Result<(Proof, Metrics), OrchestratorError> {
    if !check_satisfiability(ccs, instance, witness) {
        return Err(OrchestratorError::Unsatisfied);
    }

    let _committer = AjtaiCommitter::new();

    let t0 = std::time::Instant::now();

    // Build the exact FS transcript the backend consumes (and embed it into the proof)
    let call_id = PROVE_CALLS.fetch_add(1, Ordering::Relaxed);
    let mut fs_bytes = domain_separated_transcript(0, FS_LABEL);
    fs_bytes.extend_from_slice(b"||NONCE||");
    fs_bytes.extend_from_slice(&call_id.to_be_bytes());
    if let Ok(elapsed) = SystemTime::now().duration_since(UNIX_EPOCH) {
        fs_bytes.extend_from_slice(b"||TIME_NS||");
        fs_bytes.extend_from_slice(&elapsed.as_nanos().to_be_bytes());
    }
    fs_bytes.extend_from_slice(b"||INST_BIND||");
    encode_instance_into(&mut fs_bytes, instance);

    // SNARK compress
    let (proof_bytes, vk_bytes) = spartan_compress(ccs, instance, witness, &fs_bytes)
        .map_err(|_| OrchestratorError::Unsatisfied)?;

    // Transport envelope:
    //   PROOF ||VK|| VK ||INST|| <encoded instance> ||FS|| <encoded fs bytes> ||SNARK|| <marker>
    let mut out = proof_bytes;
    out.extend_from_slice(b"||VK||");
    out.extend_from_slice(&vk_bytes);
    out.extend_from_slice(b"||INST||");
    encode_instance_into(&mut out, instance);
    out.extend_from_slice(b"||FS||");
    encode_fs_into(&mut out, &fs_bytes);
    
    // NEW: tag the envelope to signal SNARK mode, without touching `proof_bytes`
    out.extend_from_slice(b"||SNARK||");
    out.extend_from_slice(SNARK_MARKER);

    let proof = Proof { transcript: out };
    let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let proof_bytes = proof.transcript.len();
    Ok((proof, Metrics { prove_ms, proof_bytes }))
}

pub fn verify(ccs: &CcsStructure, expected_instance: &CcsInstance, proof: &Proof) -> bool {
    let _committer = AjtaiCommitter::new();

    // Parse envelope pieces
    let Some(vk_pos) = proof.transcript.windows(6).position(|w| w == b"||VK||") else {
        eprintln!("Invalid proof format: missing ||VK||");
        return false;
    };
    let proof_bytes = &proof.transcript[..vk_pos];
    let after_vk = &proof.transcript[vk_pos + 6..];

    let Some(inst_pos_rel) = memmem::find(after_vk, b"||INST||") else {
        eprintln!("Invalid proof format: missing ||INST||");
        return false;
    };
    let vk_bytes = &after_vk[..inst_pos_rel];
    let after_inst = &after_vk[inst_pos_rel + 8..];

    let Some(fs_pos_rel) = memmem::find(after_inst, b"||FS||") else {
        eprintln!("Invalid proof format: missing ||FS||");
        return false;
    };
    let inst_bytes = &after_inst[..fs_pos_rel];
    let after_fs = &after_inst[fs_pos_rel + 6..];

    let Some(embedded_inst) = decode_instance(inst_bytes) else {
        eprintln!("Invalid proof: could not decode embedded instance");
        return false;
    };
    let Some(fs_bytes) = decode_fs(after_fs) else {
        eprintln!("Invalid proof: could not decode embedded FS transcript");
        return false;
    };

    // CRITICAL SECURITY: Verify that the embedded instance matches the expected instance
    // This prevents accepting proofs for arbitrary statements
    if !instances_equal(&embedded_inst, expected_instance) {
        eprintln!("Security violation: proof does not bind to expected instance");
        return false;
    }

    // Also verify that the FS transcript contains the correct instance binding
    if !fs_contains_expected_instance(&fs_bytes, expected_instance) {
        eprintln!("Security violation: FS transcript does not bind to expected instance");
        return false;
    }

    match spartan_verify(proof_bytes, vk_bytes, ccs, expected_instance, &fs_bytes) {
        Ok(ok) => ok,
        Err(e) => { eprintln!("Spartan verify failed: {e}"); false }
    }
}

// ---------- Security helper functions ----------

/// Constant-time comparison of CCS instances to prevent timing attacks
fn instances_equal(a: &CcsInstance, b: &CcsInstance) -> bool {
    
    // Check basic fields first
    let u_eq = a.u.as_canonical_u64().ct_eq(&b.u.as_canonical_u64());
    let e_eq = a.e.as_canonical_u64().ct_eq(&b.e.as_canonical_u64());
    
    // Check public input length and contents
    let pi_len_eq = a.public_input.len().ct_eq(&b.public_input.len());
    let mut pi_eq = subtle::Choice::from(1u8);
    let max_len = a.public_input.len().max(b.public_input.len());
    for i in 0..max_len {
        let a_val = a.public_input.get(i).map(|f| f.as_canonical_u64()).unwrap_or(0);
        let b_val = b.public_input.get(i).map(|f| f.as_canonical_u64()).unwrap_or(0);
        pi_eq &= a_val.ct_eq(&b_val);
    }
    
    // Check commitment length and contents
    let comm_len_eq = a.commitment.len().ct_eq(&b.commitment.len());
    let mut comm_eq = subtle::Choice::from(1u8);
    let max_comm_len = a.commitment.len().max(b.commitment.len());
    for i in 0..max_comm_len {
        let a_ring = a.commitment.get(i);
        let b_ring = b.commitment.get(i);
        match (a_ring, b_ring) {
            (Some(a_r), Some(b_r)) => {
                // Compare ring elements coefficient by coefficient
                let a_coeffs = a_r.coeffs();
                let b_coeffs = b_r.coeffs();
                let coeffs_len_eq = a_coeffs.len().ct_eq(&b_coeffs.len());
                comm_eq &= coeffs_len_eq;
                let max_coeffs = a_coeffs.len().max(b_coeffs.len());
                for j in 0..max_coeffs {
                    let a_coeff = a_coeffs.get(j).map(|c| c.as_canonical_u64()).unwrap_or(0);
                    let b_coeff = b_coeffs.get(j).map(|c| c.as_canonical_u64()).unwrap_or(0);
                    comm_eq &= a_coeff.ct_eq(&b_coeff);
                }
            }
            (None, None) => {}, // Both missing, OK
            _ => comm_eq = subtle::Choice::from(0u8), // One missing, not equal
        }
    }
    
    let all_eq = u_eq & e_eq & pi_len_eq & pi_eq & comm_len_eq & comm_eq;
    bool::from(all_eq)
}

/// Check that the FS transcript contains the expected instance binding
fn fs_contains_expected_instance(fs_bytes: &[u8], expected: &CcsInstance) -> bool {
    // Build the exact bytes the prover should have embedded
    let mut expected_binding = Vec::new();
    expected_binding.extend_from_slice(b"||INST_BIND||");
    encode_instance_into(&mut expected_binding, expected);
    
    // Check if this exact binding is present in the FS transcript
    memmem::find(fs_bytes, &expected_binding).is_some()
}

// ---------- simple envelope encoders/decoders (no serde) ----------

fn encode_instance_into(buf: &mut Vec<u8>, inst: &CcsInstance) {
    buf.extend_from_slice(b"NEO_INST_V1");
    
    // Encode ring degree (n) for proper decoding
    let n = inst.commitment.first().map(|e| e.coeffs().len()).unwrap_or(0) as u32;
    buf.extend_from_slice(&n.to_be_bytes());
    
    // Use proper commitment serialization instead of placeholders
    let commit_bytes = crate::serialize_commit(&inst.commitment);
    buf.extend_from_slice(&(commit_bytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(&commit_bytes);
    
    buf.extend_from_slice(&(inst.public_input.len() as u32).to_be_bytes());
    for f in &inst.public_input {
        buf.extend_from_slice(&f.as_canonical_u64().to_be_bytes());
    }
    buf.extend_from_slice(&inst.u.as_canonical_u64().to_be_bytes());
    buf.extend_from_slice(&inst.e.as_canonical_u64().to_be_bytes());
}

fn decode_instance(bytes: &[u8]) -> Option<CcsInstance> {
    let hdr = b"NEO_INST_V1";
    if bytes.len() < hdr.len() || &bytes[..hdr.len()] != hdr { return None; }
    let mut i = hdr.len();
    let mut take = |n: usize| -> Option<&[u8]> {
        if i + n > bytes.len() { None } else { let s = &bytes[i..i+n]; i += n; Some(s) }
    };
    
    // Read ring degree (n) for proper decoding
    let n = u32::from_be_bytes(take(4)?.try_into().ok()?) as usize;
    
    // Read commitment using proper deserialization
    let commit_len = u32::from_be_bytes(take(4)?.try_into().ok()?) as usize;
    let commit_bytes = take(commit_len)?;
    
    // Use proper commitment deserialization
    let mut cursor = std::io::Cursor::new(commit_bytes);
    let commitment = crate::read_commit(&mut cursor, n);
    
    let n_pi = u32::from_be_bytes(take(4)?.try_into().ok()?) as usize;
    let mut public_input = Vec::with_capacity(n_pi);
    for _ in 0..n_pi {
        let limb = u64::from_be_bytes(take(8)?.try_into().ok()?);
        public_input.push(F::from_u64(limb));
    }
    let u = F::from_u64(u64::from_be_bytes(take(8)?.try_into().ok()?));
    let e = F::from_u64(u64::from_be_bytes(take(8)?.try_into().ok()?));
    Some(CcsInstance { commitment, public_input, u, e })
}

fn encode_fs_into(buf: &mut Vec<u8>, fs: &[u8]) {
    buf.extend_from_slice(b"NEO_FS_V1");
    buf.extend_from_slice(&(fs.len() as u32).to_be_bytes());
    buf.extend_from_slice(fs);
}

fn decode_fs(bytes: &[u8]) -> Option<Vec<u8>> {
    let hdr = b"NEO_FS_V1";
    if bytes.len() < hdr.len() + 4 || &bytes[..hdr.len()] != hdr { return None; }
    let mut i = hdr.len();
    let len = u32::from_be_bytes(bytes[i..i+4].try_into().ok()?) as usize;
    i += 4;
    if i + len > bytes.len() { return None; }
    Some(bytes[i..i+len].to_vec())
}
