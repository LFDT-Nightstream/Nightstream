//! Bounded containers for decoding untrusted payloads.
//!
//! These are intended to reduce accidental DoS risk from serde formats that preallocate based on
//! attacker-controlled lengths (e.g. `Vec` length prefixes).

#![forbid(unsafe_code)]

use core::fmt;
use core::marker::PhantomData;
use core::ops::Deref;

use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize)]
#[serde(transparent)]
pub(crate) struct BoundedVec<T, const MAX: usize>(pub Vec<T>);

impl<T, const MAX: usize> From<Vec<T>> for BoundedVec<T, MAX> {
    fn from(value: Vec<T>) -> Self {
        Self(value)
    }
}

impl<T, const MAX: usize> From<BoundedVec<T, MAX>> for Vec<T> {
    fn from(value: BoundedVec<T, MAX>) -> Self {
        value.0
    }
}

impl<T, const MAX: usize> Deref for BoundedVec<T, MAX> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'de, T, const MAX: usize> Deserialize<'de> for BoundedVec<T, MAX>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct BoundedVecVisitor<T, const MAX: usize>(PhantomData<T>);

        impl<'de, T, const MAX: usize> Visitor<'de> for BoundedVecVisitor<T, MAX>
        where
            T: Deserialize<'de>,
        {
            type Value = BoundedVec<T, MAX>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "a sequence with <= {MAX} elements")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                if let Some(len) = seq.size_hint() {
                    if len > MAX {
                        return Err(de::Error::custom("sequence too large"));
                    }
                    let mut out = Vec::with_capacity(len);
                    for idx in 0..len {
                        let elem = seq
                            .next_element()?
                            .ok_or_else(|| de::Error::invalid_length(idx, &self))?;
                        out.push(elem);
                    }
                    return Ok(BoundedVec(out));
                }

                let mut out = Vec::new();
                while let Some(elem) = seq.next_element::<T>()? {
                    if out.len() >= MAX {
                        return Err(de::Error::custom("sequence too large"));
                    }
                    out.push(elem);
                }
                Ok(BoundedVec(out))
            }
        }

        deserializer.deserialize_seq(BoundedVecVisitor::<T, MAX>(PhantomData))
    }
}

