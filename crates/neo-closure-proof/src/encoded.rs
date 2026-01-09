//! Canonical (serde-friendly) encodings for Neo data types used in closure-proof payloads.
//!
//! These encodings are used to:
//! - avoid relying on upstream type serialization stability, and
//! - keep closure-proof payloads self-describing and deterministic.

#![forbid(unsafe_code)]

use crate::bounded::BoundedVec;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{Mat, MeInstance};
use neo_math::{D as NeoD, F as NeoF, K as NeoK, KExtensions};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const MAX_COMMITMENT_DATA_U64: usize = NeoD * 4096; // d * kappa (kappa is small in practice)
const MAX_MAT_DATA_U64: usize = NeoD * 16_384; // d * cols
const MAX_ME_R_LEN: usize = 64;
const MAX_ME_Y_ROWS: usize = 4096;
const MAX_ME_Y_ROW_LEN: usize = 512;
const MAX_OBLIGATIONS: usize = 16_384;

fn decode_canonical_f(x: u64) -> Option<NeoF> {
    let f = NeoF::from_u64(x);
    (f.as_canonical_u64() == x).then_some(f)
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EncodedK {
    pub c0: u64,
    pub c1: u64,
}

impl EncodedK {
    pub fn encode(x: &NeoK) -> Self {
        let (c0, c1) = x.to_limbs_u64();
        Self { c0, c1 }
    }

    pub fn decode(&self) -> Option<NeoK> {
        Some(NeoK::from_coeffs([
            decode_canonical_f(self.c0)?,
            decode_canonical_f(self.c1)?,
        ]))
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EncodedCommitment {
    pub d: u32,
    pub kappa: u32,
    pub data: BoundedVec<u64, MAX_COMMITMENT_DATA_U64>,
}

impl EncodedCommitment {
    pub fn encode(c: &Cmt) -> Self {
        Self {
            d: c.d as u32,
            kappa: c.kappa as u32,
            data: c
                .data
                .iter()
                .map(|x| x.as_canonical_u64())
                .collect::<Vec<_>>()
                .into(),
        }
    }

    pub fn decode(&self) -> Option<Cmt> {
        let d = self.d as usize;
        let kappa = self.kappa as usize;
        if self.data.len() != d * kappa {
            return None;
        }
        Some(Cmt {
            d,
            kappa,
            data: self
                .data
                .iter()
                .copied()
                .map(decode_canonical_f)
                .collect::<Option<Vec<_>>>()?,
        })
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EncodedMatF {
    pub rows: u32,
    pub cols: u32,
    pub data: BoundedVec<u64, MAX_MAT_DATA_U64>,
}

impl EncodedMatF {
    pub fn encode(m: &Mat<NeoF>) -> Self {
        Self {
            rows: m.rows() as u32,
            cols: m.cols() as u32,
            data: m
                .as_slice()
                .iter()
                .map(|x| x.as_canonical_u64())
                .collect::<Vec<_>>()
                .into(),
        }
    }

    pub fn decode(&self) -> Option<Mat<NeoF>> {
        let rows = self.rows as usize;
        let cols = self.cols as usize;
        if self.data.len() != rows.checked_mul(cols)? {
            return None;
        }
        Some(Mat::from_row_major(
            rows,
            cols,
            self
                .data
                .iter()
                .copied()
                .map(decode_canonical_f)
                .collect::<Option<Vec<_>>>()?,
        ))
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EncodedMeInstance {
    pub c: EncodedCommitment,
    pub X: EncodedMatF,
    pub r: BoundedVec<EncodedK, MAX_ME_R_LEN>,
    pub y: BoundedVec<BoundedVec<EncodedK, MAX_ME_Y_ROW_LEN>, MAX_ME_Y_ROWS>,
    pub y_scalars: BoundedVec<EncodedK, MAX_ME_Y_ROWS>,
    pub m_in: u32,
}

impl EncodedMeInstance {
    pub fn encode(me: &MeInstance<Cmt, NeoF, NeoK>) -> Self {
        Self {
            c: EncodedCommitment::encode(&me.c),
            X: EncodedMatF::encode(&me.X),
            r: me.r.iter().map(EncodedK::encode).collect::<Vec<_>>().into(),
            y: me
                .y
                .iter()
                .map(|row| row.iter().map(EncodedK::encode).collect::<Vec<_>>().into())
                .collect::<Vec<_>>()
                .into(),
            y_scalars: me
                .y_scalars
                .iter()
                .map(EncodedK::encode)
                .collect::<Vec<_>>()
                .into(),
            m_in: me.m_in as u32,
        }
    }

    pub fn decode(&self) -> Option<MeInstance<Cmt, NeoF, NeoK>> {
        Some(MeInstance {
            c: self.c.decode()?,
            X: self.X.decode()?,
            r: self
                .r
                .iter()
                .map(EncodedK::decode)
                .collect::<Option<Vec<_>>>()?,
            y: self
                .y
                .iter()
                .map(|row| row.iter().map(EncodedK::decode).collect::<Option<Vec<_>>>())
                .collect::<Option<Vec<_>>>()?,
            y_scalars: self
                .y_scalars
                .iter()
                .map(EncodedK::decode)
                .collect::<Option<Vec<_>>>()?,
            m_in: self.m_in as usize,
            // Fields not needed by closure verifiers today.
            fold_digest: [0u8; 32],
            c_step_coords: Vec::new(),
            u_offset: 0,
            u_len: 0,
        })
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EncodedObligations {
    pub main: BoundedVec<EncodedMeInstance, MAX_OBLIGATIONS>,
    pub val: BoundedVec<EncodedMeInstance, MAX_OBLIGATIONS>,
}

impl EncodedObligations {
    pub fn encode(obs: &neo_fold::shard::ShardObligations<Cmt, NeoF, NeoK>) -> Self {
        Self {
            main: obs
                .main
                .iter()
                .map(EncodedMeInstance::encode)
                .collect::<Vec<_>>()
                .into(),
            val: obs
                .val
                .iter()
                .map(EncodedMeInstance::encode)
                .collect::<Vec<_>>()
                .into(),
        }
    }

    pub fn decode(&self) -> Option<neo_fold::shard::ShardObligations<Cmt, NeoF, NeoK>> {
        Some(neo_fold::shard::ShardObligations {
            main: self
                .main
                .iter()
                .map(EncodedMeInstance::decode)
                .collect::<Option<Vec<_>>>()?,
            val: self
                .val
                .iter()
                .map(EncodedMeInstance::decode)
                .collect::<Option<Vec<_>>>()?,
        })
    }
}
