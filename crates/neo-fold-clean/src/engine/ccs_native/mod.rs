//! CCS-native circuit builders for use as alternatives to the
//! degree-2 `R1csBuilder`.
//!
//! Everything in this module emits **sparse `CcsStructure`s** with
//! degree-`u` polynomials directly, bypassing the R1CS → CCS lowering
//! and the degree-2 ceiling that comes with it. Witnesses are strict
//! bit-backed (every committed coordinate is `{0, 1}` except the
//! leading constant `1` slot), so the resulting `CcsInstance`s are
//! low-norm under `b = 2` and acceptable to SuperNeo Π_CCS without
//! further encoding.
//!
//! Scope: standalone components only. None of these builders touch
//! F', NIFS, or lifecycle. They are individual gadgets that produce
//! one self-contained CCS instance per call. F'-specific composition
//! lives in `frontends::f_prime::structure`.

pub mod poseidon2;
pub mod poseidon2_transcript;
