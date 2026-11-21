use std::io::{self, BufReader, Read};
use std::marker::PhantomData;

use neo_math::Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::PackageError;

pub trait FrameCodec: Send + Sync {
    type Decoder<'a>: Read;

    fn encode(source: impl Read) -> Box<[u8]>;
    fn decode(bytes: &[u8]) -> io::Result<Self::Decoder<'_>>;
}

struct IteratorReader<I>(I);

impl<I: Iterator<Item = u8>> Read for IteratorReader<I> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let mut count = 0;
        for (cell, byte) in output.iter_mut().zip(self.0.by_ref()) {
            *cell = byte;
            count += 1;
        }
        Ok(count)
    }
}

pub(super) struct Frame<C> {
    pub(super) count: usize,
    bytes: Box<[u8]>,
    codec: PhantomData<C>,
}

impl<C: FrameCodec> Frame<C> {
    fn of(count: usize, source: impl Read) -> Self {
        Self {
            count,
            bytes: C::encode(source),
            codec: PhantomData,
        }
    }

    pub(super) fn pack_u32(words: impl ExactSizeIterator<Item = u32>) -> Self {
        let count = words.len();
        let bytes = words.scan(0u32, |previous, word| {
            let delta = word.wrapping_sub(*previous) as i32;
            *previous = word;
            let mut value = ((delta << 1) ^ (delta >> 31)) as u32;
            let mut bytes = [0; 5];
            let mut len = 0;
            while value >= 128 {
                bytes[len] = (value as u8) | 128;
                value >>= 7;
                len += 1;
            }
            bytes[len] = value as u8;
            Some(bytes.into_iter().take(len + 1))
        });
        Self::of(count, IteratorReader(bytes.flatten()))
    }

    pub(super) fn pack_fields(words: &[Fq]) -> Self {
        let bytes = words
            .iter()
            .flat_map(|word| word.as_canonical_u64().to_le_bytes());
        Self::of(words.len(), IteratorReader(bytes))
    }

    pub(super) fn pack_bytes(bytes: &[u8]) -> Self {
        Self::of(bytes.len(), bytes)
    }

    pub(super) fn reader(&self) -> Result<FrameReader<C::Decoder<'_>>, PackageError> {
        let stream = C::decode(&self.bytes)?;
        Ok(FrameReader {
            input: BufReader::new(stream),
            previous: 0,
        })
    }

    pub(super) fn unpack_u32(self) -> Result<Vec<u32>, PackageError> {
        let mut reader = self.reader()?;
        reader.collect(self.count, FrameReader::u32)
    }

    pub(super) fn unpack_fields(self) -> Result<Vec<Fq>, PackageError> {
        let mut reader = self.reader()?;
        reader.collect(self.count, FrameReader::field)
    }

    pub(super) fn unpack_bytes(self) -> Result<Vec<u8>, PackageError> {
        let mut reader = self.reader()?;
        let mut bytes = vec![0; self.count];
        reader.input.read_exact(&mut bytes)?;
        reader.finish()?;
        Ok(bytes)
    }
}

pub(super) struct FrameReader<R> {
    input: BufReader<R>,
    previous: u32,
}

impl<R: Read> FrameReader<R> {
    fn collect<T>(
        &mut self,
        count: usize,
        mut read: impl FnMut(&mut Self) -> Result<T, PackageError>,
    ) -> Result<Vec<T>, PackageError> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(read(self)?);
        }
        self.finish()?;
        Ok(values)
    }

    pub(super) fn u32(&mut self) -> Result<u32, PackageError> {
        let mut value = 0u32;
        let mut byte = [0];
        for shift in (0..=28).step_by(7) {
            self.input.read_exact(&mut byte)?;
            if shift == 28 && byte[0] >= 16 {
                break;
            }
            value |= u32::from(byte[0] & 127) << shift;
            if byte[0] < 128 {
                let delta = ((value >> 1) as i32) ^ -(value as i32 & 1);
                self.previous = self.previous.wrapping_add(delta as u32);
                return Ok(self.previous);
            }
        }
        Err(PackageError::Malformed("packed circuit word"))
    }

    fn field(&mut self) -> Result<Fq, PackageError> {
        let mut bytes = [0; 8];
        self.input.read_exact(&mut bytes)?;
        let word = u64::from_le_bytes(bytes);
        if word >= Fq::ORDER_U64 {
            return Err(PackageError::Malformed("packed circuit field word"));
        }
        Ok(Fq::from_u64(word))
    }

    pub(super) fn finish(&mut self) -> Result<(), PackageError> {
        if self.input.read(&mut [0])? != 0 {
            return Err(PackageError::Malformed("trailing packed circuit words"));
        }
        Ok(())
    }
}
