//! This module provides an implementation of `TranscriptEngineTrait` using Poseidon2 over Goldilocks.

use crate::{
  errors::SpartanError,
  poseidon2_shared::goldilocks_poseidon2_perm,
  traits::{
    Engine, PrimeFieldExt,
    transcript::{TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use core::marker::PhantomData;
use neo_params::poseidon2_goldilocks::{RATE, WIDTH};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_symmetric::Permutation;

const APP_DOMAIN: &[u8] = b"spartan2/transcript/v1|poseidon2-goldilocks";

#[derive(Debug, Clone)]
/// Poseidon2-based transcript engine for Goldilocks-backed Spartan proofs.
pub struct Poseidon2Transcript<E: Engine> {
  st: [Goldilocks; WIDTH],
  absorbed: usize,
  _p: PhantomData<E>,
}

impl<E: Engine> Poseidon2Transcript<E> {
  #[inline]
  fn absorb_elem(&mut self, x: Goldilocks) {
    if self.absorbed >= RATE {
      self.permute();
    }
    self.st[self.absorbed] = x;
    self.absorbed += 1;
  }

  #[inline]
  fn absorb_slice(&mut self, inputs: &[Goldilocks]) {
    let mut src_idx = 0usize;
    while self.absorbed < RATE && src_idx < inputs.len() {
      self.st[self.absorbed] = inputs[src_idx];
      self.absorbed += 1;
      src_idx += 1;
    }

    if self.absorbed == RATE {
      self.permute();
    }

    while inputs.len() - src_idx >= RATE {
      for i in 0..RATE {
        self.st[i] = inputs[src_idx + i];
      }
      self.permute();
      src_idx += RATE;
    }

    while src_idx < inputs.len() {
      self.st[self.absorbed] = inputs[src_idx];
      self.absorbed += 1;
      src_idx += 1;
    }
  }

  #[inline]
  fn absorb_packed_bytes_with_len(&mut self, bytes: &[u8]) {
    self.absorb_elem(Goldilocks::from_u64(bytes.len() as u64));

    const BYTES_PER_LIMB: usize = 7;
    const LIMB_CHUNK: usize = 64;
    let mut packed = [Goldilocks::ZERO; LIMB_CHUNK];
    let mut used = 0usize;
    let mut i = 0usize;

    while i + BYTES_PER_LIMB <= bytes.len() {
      let mut limb = [0u8; 8];
      limb[..BYTES_PER_LIMB].copy_from_slice(&bytes[i..i + BYTES_PER_LIMB]);
      packed[used] = Goldilocks::from_u64(u64::from_le_bytes(limb));
      used += 1;
      i += BYTES_PER_LIMB;
      if used == LIMB_CHUNK {
        self.absorb_slice(&packed);
        used = 0;
      }
    }

    if i < bytes.len() {
      let mut limb = [0u8; 8];
      limb[..(bytes.len() - i)].copy_from_slice(&bytes[i..]);
      packed[used] = Goldilocks::from_u64(u64::from_le_bytes(limb));
      used += 1;
    }

    if used > 0 {
      self.absorb_slice(&packed[..used]);
    }
  }

  #[inline]
  fn append_message(&mut self, label: &'static [u8], msg: &[u8]) {
    self.absorb_packed_bytes_with_len(label);
    self.absorb_packed_bytes_with_len(msg);
  }

  #[inline]
  fn permute(&mut self) {
    self.st = goldilocks_poseidon2_perm().permute(self.st);
    self.absorbed = 0;
  }
}

impl<E: Engine> TranscriptEngineTrait<E> for Poseidon2Transcript<E> {
  fn new(label: &'static [u8]) -> Self {
    let mut tr = Self {
      st: [Goldilocks::ZERO; WIDTH],
      absorbed: 0,
      _p: PhantomData,
    };
    tr.append_message(APP_DOMAIN, label);
    tr
  }

  fn squeeze(&mut self, label: &'static [u8]) -> Result<E::Scalar, SpartanError> {
    self.append_message(b"chal/label", label);
    let mut out = [0u8; 64];
    let mut produced = 0usize;
    while produced < out.len() {
      self.absorb_elem(Goldilocks::ONE);
      self.permute();
      for i in 0..4 {
        let limb = self.st[i].as_canonical_u64().to_le_bytes();
        let take = (out.len() - produced).min(8);
        out[produced..produced + take].copy_from_slice(&limb[..take]);
        produced += take;
        if produced >= out.len() {
          break;
        }
      }
    }
    Ok(E::Scalar::from_uniform(&out))
  }

  fn absorb<T: TranscriptReprTrait<E::GE>>(&mut self, label: &'static [u8], o: &T) {
    self.append_message(label, &o.to_transcript_bytes());
  }

  fn dom_sep(&mut self, bytes: &'static [u8]) {
    self.append_message(b"spartan2/dom-sep", bytes);
  }
}
