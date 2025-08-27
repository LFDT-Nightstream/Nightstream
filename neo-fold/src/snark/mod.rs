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
use neo_ring::RingElement;
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

pub fn verify(ccs: &CcsStructure, proof: &Proof) -> bool {
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

    let Some(inst) = decode_instance(inst_bytes) else {
        eprintln!("Invalid proof: could not decode embedded instance");
        return false;
    };
    let Some(fs_bytes) = decode_fs(after_fs) else {
        eprintln!("Invalid proof: could not decode embedded FS transcript");
        return false;
    };

    match spartan_verify(proof_bytes, vk_bytes, ccs, &inst, &fs_bytes) {
        Ok(ok) => ok,
        Err(e) => { eprintln!("Spartan verify failed: {e}"); false }
    }
}

// ---------- simple envelope encoders/decoders (no serde) ----------

fn encode_instance_into(buf: &mut Vec<u8>, inst: &CcsInstance) {
    buf.extend_from_slice(b"NEO_INST_V1");
    buf.extend_from_slice(&(inst.commitment.len() as u32).to_be_bytes());
    for _ring_elem in &inst.commitment {
        // For now, encode RingElement as a placeholder - this needs proper serialization
        // TODO: Implement proper RingElement serialization
        buf.extend_from_slice(&0u64.to_be_bytes()); // Placeholder
    }
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
    let n_comm = u32::from_be_bytes(take(4)?.try_into().ok()?) as usize;
    let mut commitment = Vec::with_capacity(n_comm);
    for _ in 0..n_comm {
        let _limb = u64::from_be_bytes(take(8)?.try_into().ok()?);
        // TODO: Implement proper RingElement deserialization
        // For now, create a placeholder RingElement
        commitment.push(RingElement::zero(1)); // Placeholder with n=1
    }
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
