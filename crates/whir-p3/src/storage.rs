//! Disk-backed buffers for very large evaluation tables.
//!
//! WHIR operates over evaluation vectors whose length is often `2^n`. At production sizes, these
//! vectors can be too large to comfortably hold in RAM. This module provides a simple abstraction
//! (`Buffer<T>`) that can store data either in-memory (`Vec<T>`) or in a temporary file backed by an
//! `mmap` mapping (`MmapBuffer<T>`).
//!
//! The design goal is to keep the rest of the code largely unchanged by exposing a slice-like API
//! (`Borrow<[T]>` / `BorrowMut<[T]>`) suitable for `p3_matrix::DenseMatrix` storage.

use std::borrow::{Borrow, BorrowMut};
use std::fs::File;
use std::io;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};

use memmap2::{MmapMut, MmapOptions};
use p3_matrix::dense::DenseStorage;

/// Default threshold above which we prefer a disk-backed buffer.
pub const DEFAULT_MMAP_THRESHOLD_BYTES: usize = 64 * 1024 * 1024;

/// Returns the threshold (in bytes) above which buffers should prefer `mmap` storage.
///
/// This can be overridden at runtime via `WHIR_P3_MMAP_THRESHOLD_BYTES`.
#[inline]
pub fn mmap_threshold_bytes() -> usize {
    std::env::var("WHIR_P3_MMAP_THRESHOLD_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MMAP_THRESHOLD_BYTES)
}

fn byte_len<T>(len: usize) -> io::Result<usize> {
    let elem = size_of::<T>();
    if elem == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "zero-sized element types are not supported",
        ));
    }
    len.checked_mul(elem)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "buffer size overflow"))
}

#[derive(Debug)]
pub struct MmapBuffer<T> {
    len: usize,
    mmap: MmapMut,
    // Keep the backing file alive for the lifetime of the mapping.
    _file: File,
    _marker: PhantomData<T>,
}

impl<T: Copy> MmapBuffer<T> {
    /// Creates a new disk-backed buffer of length `len`.
    ///
    /// The backing file is zero-initialized by the OS. For typical field element types (including
    /// `p3_*` prime fields), the all-zero byte pattern corresponds to `ZERO`.
    pub fn new_zeroed(len: usize) -> io::Result<Self> {
        let bytes = byte_len::<T>(len)?;
        let file = tempfile::tempfile()?;
        file.set_len(bytes as u64)?;
        let mut mmap = unsafe { MmapOptions::new().len(bytes).map_mut(&file)? };

        // Ensure alignment is sufficient for `T`.
        let ptr = mmap.as_mut_ptr();
        if (ptr as usize) % align_of::<T>() != 0 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "mmap base pointer is not sufficiently aligned",
            ));
        }

        Ok(Self {
            len,
            mmap,
            _file: file,
            _marker: PhantomData,
        })
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        assert!(
            new_len <= self.len,
            "truncate out of bounds: new_len={new_len} len={}",
            self.len
        );
        self.len = new_len;
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.mmap.as_ptr().cast::<T>(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.mmap.as_mut_ptr().cast::<T>(), self.len) }
    }
}

impl<T: Copy> Borrow<[T]> for MmapBuffer<T> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy> BorrowMut<[T]> for MmapBuffer<T> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Copy + Send + Sync> DenseStorage<T> for MmapBuffer<T> {
    #[inline]
    fn to_vec(self) -> Vec<T> {
        self.as_slice().to_vec()
    }
}

impl<T: Copy> Clone for MmapBuffer<T> {
    fn clone(&self) -> Self {
        let mut out = Self::new_zeroed(self.len).expect("mmap clone allocation must succeed");
        out.as_mut_slice().copy_from_slice(self.as_slice());
        out
    }
}

impl<T: Copy + PartialEq> PartialEq for MmapBuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Copy + Eq> Eq for MmapBuffer<T> {}

#[derive(Debug)]
pub enum Buffer<T> {
    Vec(Vec<T>),
    Mmap(MmapBuffer<T>),
}

impl<T: Copy> Buffer<T> {
    #[inline]
    pub const fn len(&self) -> usize {
        match self {
            Self::Vec(v) => v.len(),
            Self::Mmap(m) => m.len(),
        }
    }

    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        match self {
            Self::Vec(v) => v.truncate(new_len),
            Self::Mmap(m) => m.truncate(new_len),
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::Vec(v) => v.as_slice(),
            Self::Mmap(m) => m.as_slice(),
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Vec(v) => v.as_mut_slice(),
            Self::Mmap(m) => m.as_mut_slice(),
        }
    }

    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Self::Vec(v) => v,
            Self::Mmap(m) => m.as_slice().to_vec(),
        }
    }

    /// Allocate a buffer and choose `mmap` if the total size crosses `threshold_bytes`.
    pub fn try_alloc_zeroed_with_threshold(len: usize, threshold_bytes: usize) -> io::Result<Self>
    where
        T: Default,
    {
        let bytes = byte_len::<T>(len)?;
        if bytes >= threshold_bytes {
            Ok(Self::Mmap(MmapBuffer::new_zeroed(len)?))
        } else {
            Ok(Self::Vec(vec![T::default(); len]))
        }
    }

    /// Allocate a buffer and choose `mmap` if the total size crosses the default threshold.
    pub fn try_alloc_zeroed(len: usize) -> io::Result<Self>
    where
        T: Default,
    {
        Self::try_alloc_zeroed_with_threshold(len, mmap_threshold_bytes())
    }
}

impl<T: Copy> Borrow<[T]> for Buffer<T> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy> BorrowMut<[T]> for Buffer<T> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Copy + Send + Sync> DenseStorage<T> for Buffer<T> {
    #[inline]
    fn to_vec(self) -> Vec<T> {
        match self {
            Self::Vec(v) => v,
            Self::Mmap(m) => m.as_slice().to_vec(),
        }
    }
}

impl<T: Copy> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Vec(v) => Self::Vec(v.clone()),
            Self::Mmap(m) => Self::Mmap(m.clone()),
        }
    }
}

impl<T: Copy + PartialEq> PartialEq for Buffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Copy + Eq> Eq for Buffer<T> {}
