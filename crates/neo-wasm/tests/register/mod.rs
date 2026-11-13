//! Scratch prototype of a general register row (not for review): the
//! translator from WASM, its register allocator, an interpreter that counts
//! rows and original instructions, and the corpus of real programs.

#![allow(dead_code)]

pub mod alloc;
pub mod corpus;
pub mod cpu;
pub mod general;
pub mod isa;
pub mod lower;
pub mod machine;
pub mod row;
pub mod translate;
