use serde::Deserialize;

#[derive(Clone, Debug)]
pub struct TransferWitness {
    pub ram_pairs: Vec<(u64, u32)>,
    pub output_layout_words: Vec<(u64, u32)>,
}

#[derive(Clone, Deserialize)]
struct FixtureNoteSpendWitness {
    domain: [u8; 32],
    spend_sk: [u8; 32],
    pk_ivk_owner: [u8; 32],
    depth: u32,
    anchor: [u8; 32],
    inputs: Vec<FixtureInput>,
    withdraw_amount: u64,
    withdraw_to: [u8; 32],
    outputs: Vec<FixtureOutput>,
    inv_enforce: [u8; 32],
    blacklist_root: [u8; 32],
    blacklist_proofs: Vec<FixtureBlacklistProof>,
    viewers: Vec<FixtureViewer>,
}

#[derive(Clone, Deserialize)]
struct FixtureInput {
    value: u64,
    rho: [u8; 32],
    sender_id: [u8; 32],
    position: u32,
    siblings: Vec<[u8; 32]>,
    nullifier: [u8; 32],
}

#[derive(Clone, Deserialize)]
struct FixtureOutput {
    value: u64,
    rho: [u8; 32],
    pk_spend: [u8; 32],
    pk_ivk: [u8; 32],
    cm: [u8; 32],
}

#[derive(Clone, Deserialize)]
struct FixtureBlacklistProof {
    bucket_entries: Vec<[u8; 32]>,
    bucket_inv: [u8; 32],
    siblings: Vec<[u8; 32]>,
}

#[derive(Clone, Deserialize)]
struct FixtureViewer {
    fvk_commitment: [u8; 32],
    fvk: [u8; 32],
    per_output: Vec<FixtureViewerOutput>,
}

#[derive(Clone, Deserialize)]
struct FixtureViewerOutput {
    ct_hash: [u8; 32],
    mac: [u8; 32],
}

struct RamWordWriter {
    addr: u64,
    pairs: Vec<(u64, u32)>,
}

impl RamWordWriter {
    fn new(start: u64) -> Self {
        Self {
            addr: start,
            pairs: Vec::new(),
        }
    }

    fn write_u32(&mut self, val: u32) {
        self.pairs.push((self.addr, val));
        self.addr += 4;
    }

    fn write_u64(&mut self, val: u64) {
        self.write_u32(val as u32);
        self.write_u32((val >> 32) as u32);
    }

    fn write_hash32(&mut self, d: &[u8; 32]) {
        for i in 0..4 {
            let mut word = [0u8; 8];
            word.copy_from_slice(&d[i * 8..(i + 1) * 8]);
            self.write_u64(u64::from_le_bytes(word));
        }
    }
}

pub fn build_note_spend_fixture_witness() -> TransferWitness {
    build_note_spend_fixture_witness_with_addrs(0x104, 0x100)
}

pub fn build_note_spend_fixture_witness_with_addrs(input_addr: u64, output_addr: u64) -> TransferWitness {
    let fixture: FixtureNoteSpendWitness =
        serde_json::from_str(include_str!("../fixtures/nightstream_note_spend_poseidon_fail.json"))
            .expect("parse note spend fixture JSON");

    let mut in_w = RamWordWriter::new(input_addr);
    in_w.write_hash32(&fixture.domain);
    in_w.write_hash32(&fixture.spend_sk);
    in_w.write_hash32(&fixture.pk_ivk_owner);
    in_w.write_u32(fixture.depth);
    in_w.write_hash32(&fixture.anchor);
    in_w.write_u32(fixture.inputs.len() as u32);

    for input in &fixture.inputs {
        in_w.write_u64(input.value);
        in_w.write_hash32(&input.rho);
        in_w.write_hash32(&input.sender_id);
        in_w.write_u32(input.position);
        for sib in &input.siblings {
            in_w.write_hash32(sib);
        }
    }
    for input in &fixture.inputs {
        in_w.write_hash32(&input.nullifier);
    }

    in_w.write_u64(fixture.withdraw_amount);
    in_w.write_hash32(&fixture.withdraw_to);
    in_w.write_u32(fixture.outputs.len() as u32);

    for output in &fixture.outputs {
        in_w.write_u64(output.value);
        in_w.write_hash32(&output.rho);
        in_w.write_hash32(&output.pk_spend);
        in_w.write_hash32(&output.pk_ivk);
    }
    for output in &fixture.outputs {
        in_w.write_hash32(&output.cm);
    }

    let mut inv_enforce_lo = [0u8; 8];
    inv_enforce_lo.copy_from_slice(&fixture.inv_enforce[..8]);
    in_w.write_u64(u64::from_le_bytes(inv_enforce_lo));

    in_w.write_hash32(&fixture.blacklist_root);
    for proof in &fixture.blacklist_proofs {
        for entry in &proof.bucket_entries {
            in_w.write_hash32(entry);
        }
        let mut inv_lo = [0u8; 8];
        inv_lo.copy_from_slice(&proof.bucket_inv[..8]);
        in_w.write_u64(u64::from_le_bytes(inv_lo));
        for sib in &proof.siblings {
            in_w.write_hash32(sib);
        }
    }

    in_w.write_u32(fixture.viewers.len() as u32);
    for viewer in &fixture.viewers {
        in_w.write_hash32(&viewer.fvk_commitment);
        in_w.write_hash32(&viewer.fvk);
        for out_w in &viewer.per_output {
            in_w.write_hash32(&out_w.ct_hash);
            in_w.write_hash32(&out_w.mac);
        }
    }

    let mut out_w = RamWordWriter::new(output_addr);
    out_w.write_hash32(&fixture.anchor);
    out_w.write_u32(fixture.inputs.len() as u32);
    for input in &fixture.inputs {
        out_w.write_hash32(&input.nullifier);
    }
    out_w.write_u64(fixture.withdraw_amount);
    out_w.write_hash32(&fixture.withdraw_to);
    out_w.write_u32(fixture.outputs.len() as u32);
    for output in &fixture.outputs {
        out_w.write_hash32(&output.cm);
    }
    out_w.write_hash32(&fixture.blacklist_root);
    out_w.write_u32(fixture.viewers.len() as u32);
    for viewer in &fixture.viewers {
        assert_eq!(viewer.per_output.len(), fixture.outputs.len());
        for (out_witness, output) in viewer.per_output.iter().zip(&fixture.outputs) {
            out_w.write_hash32(&output.cm);
            out_w.write_hash32(&viewer.fvk_commitment);
            out_w.write_hash32(&out_witness.ct_hash);
            out_w.write_hash32(&out_witness.mac);
        }
    }

    TransferWitness {
        ram_pairs: in_w.pairs,
        output_layout_words: out_w.pairs,
    }
}
