//! Static opaque-value lowering and native replay. The verifier owns the schema
//! and source order; roots compress those sources without interpreting their types.

use super::{opaque_control_encoding, EventBlock, SlotBinding};
use crate::WasmBuildError;
use neo_application::event_commitment::commit_block;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

const DOMAIN: u64 = u64::from_le_bytes(*b"NEOOBJ01");

fn error(message: &str) -> WasmBuildError {
    WasmBuildError::Trace(format!("opaque value: {message}"))
}

fn header(schema: [u64; 4], len: usize) -> [u64; 8] {
    [DOMAIN, 1, schema[0], schema[1], schema[2], schema[3], len as u64, 0]
}

fn canonical(words: &[u64]) -> Result<(), WasmBuildError> {
    if words.iter().any(|&word| word >= Goldilocks::ORDER_U64) {
        return Err(error("non-canonical field word"));
    }
    Ok(())
}

/// Commit a canonical, statically shaped word sequence using Poseidon2 only.
/// The embedder must derive the same structural schema identity and word order
/// on both sides of a boundary. Schema identities are verifier data, not authority
/// supplied by the prover. Length distinguishes values differing by trailing zeroes.
/// The framing block is `[NEOOBJ01, 1, schema[0..4], word_count, 0]`, followed
/// by zero-padded eight-word payload blocks, starting from an all-zero chain.
pub fn opaque_value_root(schema: [u64; 4], words: &[u64]) -> Result<[u64; 4], WasmBuildError> {
    canonical(&schema)?;
    canonical(words)?;
    let mut state = absorb([0; 4], header(schema, words.len()));
    for chunk in words.chunks(8) {
        let mut block = [0; 8];
        block[..chunk.len()].copy_from_slice(chunk);
        state = absorb(state, block);
    }
    Ok(state)
}

fn absorb(state: [u64; 4], block: [u64; 8]) -> [u64; 4] {
    commit_block(state.map(Goldilocks::from_u64), block.map(Goldilocks::from_u64)).map(|value| value.as_canonical_u64())
}

impl EventBlock {
    /// Replace four zero slots, starting at absolute block word `word`, with
    /// the root of a static value. Returns the complete gather schedule to put
    /// in an import or export template. All source effects retain their order:
    /// outer prefix, object sources, then outer suffix.
    /// Enter runs immediately after the prefix; unused slots in that ROM group
    /// remain zero placeholders but do not produce gather rows or a permutation.
    ///
    /// The root must fit in this block. Pad the outer event explicitly before
    /// using this helper if necessary. Nested opaque values are unsupported.
    /// `schema` is the embedder's canonical structural type identity, shared
    /// with the consumer of the root (see [`opaque_value_root`]).
    ///
    /// ```
    /// use neo_wasm::host_event_bindings::{EventBlock, Limb, SlotBinding};
    /// let zero = SlotBinding::Const(0);
    /// // Keep resource and method visible; replace words 3..7 with one root.
    /// let call = EventBlock::op(7, [
    ///     SlotBinding::ArgElem { arg: 0, limb: Limb::Lo },
    ///     SlotBinding::Const(2), zero, zero, zero, zero, zero,
    /// ]);
    /// let schedule = call.with_opaque(3, [1, 0, 0, 0], vec![
    ///     SlotBinding::ArgElem { arg: 1, limb: Limb::Lo },
    /// ])?;
    /// # Ok::<(), neo_wasm::WasmBuildError>(())
    /// ```
    pub fn with_opaque(
        self,
        word: usize,
        schema: [u64; 4],
        sources: Vec<SlotBinding>,
    ) -> Result<Vec<Self>, WasmBuildError> {
        canonical(&schema)?;
        if !self.absorb || word > 4 {
            return Err(error("root needs four slots in an absorbing outer block"));
        }
        if self.block[word..word + 4]
            .iter()
            .any(|s| *s != SlotBinding::Const(0))
        {
            return Err(error("root destination must contain four zero placeholders"));
        }
        if self.block.iter().chain(&sources).any(is_control) {
            return Err(error("nested or already lowered objects are unsupported"));
        }
        let mut prefix = [SlotBinding::Const(0); 8];
        prefix[..word].copy_from_slice(&self.block[..word]);
        prefix[word] = SlotBinding::EnterOpaque;
        let mut events = vec![Self {
            block: prefix,
            absorb: true,
        }];
        events.push(Self {
            block: header(schema, sources.len()).map(SlotBinding::Const),
            absorb: true,
        });
        for chunk in sources.chunks(8) {
            let mut block = [SlotBinding::Const(0); 8];
            block[..chunk.len()].copy_from_slice(chunk);
            events.push(Self { block, absorb: true });
        }
        let mut block = self.block;
        for (lane, slot) in block[..word].iter_mut().enumerate() {
            *slot = SlotBinding::OpaqueSaved { lane: lane as u8 };
        }
        for lane in 0..4 {
            block[word + lane] = SlotBinding::OpaqueRoot { lane: lane as u8 };
        }
        events.push(Self { block, absorb: true });
        Ok(events)
    }
}

fn is_control(slot: &SlotBinding) -> bool {
    opaque_control_encoding(slot).is_some()
}

/// Validate the exact lowering, including framing, padding, and four ordered
/// copy-back lanes. The resulting ROM schedule is authoritative in the circuit.
pub(super) fn validate(events: &[EventBlock]) -> Result<(), WasmBuildError> {
    let mut index = 0;
    while index < events.len() {
        let entry = &events[index];
        let Some(prefix) = entry
            .block
            .iter()
            .position(|slot| matches!(slot, SlotBinding::EnterOpaque))
        else {
            if entry.block.iter().any(is_control) {
                return Err(error("control instruction outside an object schedule"));
            }
            index += 1;
            continue;
        };
        if prefix > 4
            || !entry.absorb
            || entry.block[..prefix].iter().any(is_control)
            || entry.block[prefix + 1..]
                .iter()
                .any(|s| *s != SlotBinding::Const(0))
        {
            return Err(error("invalid suspended outer prefix"));
        }
        let framing = events
            .get(index + 1)
            .ok_or_else(|| error("missing object header"))?;
        let mut fields = [0; 8];
        for (out, slot) in fields.iter_mut().zip(framing.block) {
            let SlotBinding::Const(value) = slot else {
                return Err(error("object header must be constant"));
            };
            *out = value;
        }
        canonical(&fields)?;
        if !framing.absorb || fields[0] != DOMAIN || fields[1] != 1 || fields[7] != 0 {
            return Err(error("invalid object header"));
        }
        let len = usize::try_from(fields[6]).map_err(|_| error("length overflow"))?;
        let end = index
            .checked_add(2)
            .and_then(|i| i.checked_add(len.div_ceil(8)))
            .ok_or_else(|| error("schedule length overflow"))?;
        let exit = events
            .get(end)
            .ok_or_else(|| error("incomplete object schedule"))?;
        for (block_index, event) in events[index + 2..end].iter().enumerate() {
            if !event.absorb || event.block.iter().any(is_control) {
                return Err(error("nested object or advice in object payload"));
            }
            for (lane, slot) in event.block.iter().enumerate() {
                if block_index * 8 + lane >= len && *slot != SlotBinding::Const(0) {
                    return Err(error("nonzero object padding"));
                }
            }
        }
        if !exit.absorb {
            return Err(error("copy-back block must absorb"));
        }
        for (lane, slot) in exit.block.iter().enumerate() {
            let valid = if lane < prefix {
                *slot == SlotBinding::OpaqueSaved { lane: lane as u8 }
            } else if lane < prefix + 4 {
                *slot
                    == SlotBinding::OpaqueRoot {
                        lane: (lane - prefix) as u8,
                    }
            } else {
                !is_control(slot)
            };
            if !valid {
                return Err(error("invalid root copy-back layout"));
            }
        }
        index = end + 1;
    }
    Ok(())
}

/// Resolve internal instructions after ordinary source expansion. The returned
/// mask selects outer transcript blocks, excluding prefix and object work.
pub(super) fn replay(events: &[EventBlock], blocks: &mut [[u64; 8]]) -> Result<Vec<bool>, WasmBuildError> {
    validate(events)?;
    if events.len() != blocks.len() {
        return Err(error("expansion length mismatch"));
    }
    let mut active = false;
    let mut root = [0; 4];
    let mut saved = [0; 4];
    let mut mask = Vec::with_capacity(events.len());
    for (event, block) in events.iter().zip(blocks) {
        let enter = event
            .block
            .iter()
            .any(|slot| matches!(slot, SlotBinding::EnterOpaque));
        for (slot, value) in event.block.iter().zip(block.iter_mut()) {
            match *slot {
                SlotBinding::EnterOpaque => *value = 0,
                SlotBinding::OpaqueSaved { lane } => *value = saved[usize::from(lane)],
                SlotBinding::OpaqueRoot { lane } => {
                    *value = root[usize::from(lane)];
                    if lane == 3 {
                        active = false;
                    }
                }
                _ => {}
            }
        }
        if enter {
            saved.copy_from_slice(&block[..4]);
            root = [0; 4];
            active = true;
        } else if active {
            root = absorb(root, *block);
        }
        mask.push(event.absorb && !enter && !active);
    }
    Ok(mask)
}
