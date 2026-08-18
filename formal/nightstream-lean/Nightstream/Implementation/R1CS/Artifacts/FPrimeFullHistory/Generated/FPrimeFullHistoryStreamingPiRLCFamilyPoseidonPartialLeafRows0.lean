import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema

/-! Generated file: final selective ports for one partial-start production
PiRLC Poseidon2 leaf row shard 0.

Owns: exact Rust-projected final ports for relative rows 0 through 42
under the direct-leaf role normalization.

Does not own: source S-box semantics, replay-batch coverage, decoder soundness,
recursive orchestration, or permission to remove constraints.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeaf

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema

def rawRow00 : RawRow where
  rowOffset := 0
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 0, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 15504881536434223753 }], geometric := [{ slot := .externalA 2, initial := 2, ratio := 3 }, { slot := .externalA 3, initial := 2, ratio := 3 }, { slot := .externalA 0, initial := 4, ratio := 3 }, { slot := .externalA 1, initial := 6, ratio := 3 }, { slot := .externalB 0, initial := 2, ratio := 3 }, { slot := .externalB 1, initial := 3, ratio := 3 }, { slot := .externalB 2, initial := 1, ratio := 3 }, { slot := .externalB 3, initial := 1, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow01 : RawRow where
  rowOffset := 1
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 1, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 2212164856944708396 }], geometric := [{ slot := .externalA 2, initial := 6, ratio := 3 }, { slot := .externalA 3, initial := 2, ratio := 3 }, { slot := .externalA 0, initial := 2, ratio := 3 }, { slot := .externalA 1, initial := 4, ratio := 3 }, { slot := .externalB 0, initial := 1, ratio := 3 }, { slot := .externalB 1, initial := 2, ratio := 3 }, { slot := .externalB 2, initial := 3, ratio := 3 }, { slot := .externalB 3, initial := 1, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow02 : RawRow where
  rowOffset := 2
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 2, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 1885257220781225929 }], geometric := [{ slot := .externalA 2, initial := 4, ratio := 3 }, { slot := .externalA 3, initial := 6, ratio := 3 }, { slot := .externalA 0, initial := 2, ratio := 3 }, { slot := .externalA 1, initial := 2, ratio := 3 }, { slot := .externalB 0, initial := 1, ratio := 3 }, { slot := .externalB 1, initial := 1, ratio := 3 }, { slot := .externalB 2, initial := 2, ratio := 3 }, { slot := .externalB 3, initial := 3, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow03 : RawRow where
  rowOffset := 3
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 3, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 17531637481572944510 }], geometric := [{ slot := .externalA 2, initial := 2, ratio := 3 }, { slot := .externalA 3, initial := 4, ratio := 3 }, { slot := .externalA 0, initial := 6, ratio := 3 }, { slot := .externalA 1, initial := 2, ratio := 3 }, { slot := .externalB 0, initial := 3, ratio := 3 }, { slot := .externalB 1, initial := 1, ratio := 3 }, { slot := .externalB 2, initial := 1, ratio := 3 }, { slot := .externalB 3, initial := 2, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow04 : RawRow where
  rowOffset := 4
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 4, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 16769640728293682348 }], geometric := [{ slot := .externalA 2, initial := 1, ratio := 3 }, { slot := .externalA 3, initial := 1, ratio := 3 }, { slot := .externalA 0, initial := 2, ratio := 3 }, { slot := .externalA 1, initial := 3, ratio := 3 }, { slot := .externalB 0, initial := 4, ratio := 3 }, { slot := .externalB 1, initial := 6, ratio := 3 }, { slot := .externalB 2, initial := 2, ratio := 3 }, { slot := .externalB 3, initial := 2, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow05 : RawRow where
  rowOffset := 5
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 5, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 445908668462176974 }], geometric := [{ slot := .externalA 2, initial := 3, ratio := 3 }, { slot := .externalA 3, initial := 1, ratio := 3 }, { slot := .externalA 0, initial := 1, ratio := 3 }, { slot := .externalA 1, initial := 2, ratio := 3 }, { slot := .externalB 0, initial := 2, ratio := 3 }, { slot := .externalB 1, initial := 4, ratio := 3 }, { slot := .externalB 2, initial := 6, ratio := 3 }, { slot := .externalB 3, initial := 2, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow06 : RawRow where
  rowOffset := 6
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 6, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 1308472042479836079 }], geometric := [{ slot := .externalA 2, initial := 2, ratio := 3 }, { slot := .externalA 3, initial := 3, ratio := 3 }, { slot := .externalA 0, initial := 1, ratio := 3 }, { slot := .externalA 1, initial := 1, ratio := 3 }, { slot := .externalB 0, initial := 2, ratio := 3 }, { slot := .externalB 1, initial := 2, ratio := 3 }, { slot := .externalB 2, initial := 4, ratio := 3 }, { slot := .externalB 3, initial := 6, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow07 : RawRow where
  rowOffset := 7
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 7, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 17465001500823438575 }], geometric := [{ slot := .externalA 2, initial := 1, ratio := 3 }, { slot := .externalA 3, initial := 2, ratio := 3 }, { slot := .externalA 0, initial := 3, ratio := 3 }, { slot := .externalA 1, initial := 1, ratio := 3 }, { slot := .externalB 0, initial := 6, ratio := 3 }, { slot := .externalB 1, initial := 2, ratio := 3 }, { slot := .externalB 2, initial := 2, ratio := 3 }, { slot := .externalB 3, initial := 4, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow08 : RawRow where
  rowOffset := 8
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 8, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 1922033642430128704 }], geometric := [{ slot := .local 0, initial := 4, ratio := 3 }, { slot := .local 1, initial := 6, ratio := 3 }, { slot := .local 2, initial := 2, ratio := 3 }, { slot := .local 3, initial := 2, ratio := 3 }, { slot := .local 4, initial := 2, ratio := 3 }, { slot := .local 5, initial := 3, ratio := 3 }, { slot := .local 6, initial := 1, ratio := 3 }, { slot := .local 7, initial := 1, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow09 : RawRow where
  rowOffset := 9
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 9, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 2657514617275794404 }], geometric := [{ slot := .local 0, initial := 2, ratio := 3 }, { slot := .local 1, initial := 4, ratio := 3 }, { slot := .local 2, initial := 6, ratio := 3 }, { slot := .local 3, initial := 2, ratio := 3 }, { slot := .local 4, initial := 1, ratio := 3 }, { slot := .local 5, initial := 2, ratio := 3 }, { slot := .local 6, initial := 3, ratio := 3 }, { slot := .local 7, initial := 1, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow10 : RawRow where
  rowOffset := 10
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 10, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 17238706657248448792 }], geometric := [{ slot := .local 0, initial := 2, ratio := 3 }, { slot := .local 1, initial := 2, ratio := 3 }, { slot := .local 2, initial := 4, ratio := 3 }, { slot := .local 3, initial := 6, ratio := 3 }, { slot := .local 4, initial := 1, ratio := 3 }, { slot := .local 5, initial := 1, ratio := 3 }, { slot := .local 6, initial := 2, ratio := 3 }, { slot := .local 7, initial := 3, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow11 : RawRow where
  rowOffset := 11
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 11, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 7348277157222259646 }], geometric := [{ slot := .local 0, initial := 6, ratio := 3 }, { slot := .local 1, initial := 2, ratio := 3 }, { slot := .local 2, initial := 2, ratio := 3 }, { slot := .local 3, initial := 4, ratio := 3 }, { slot := .local 4, initial := 3, ratio := 3 }, { slot := .local 5, initial := 1, ratio := 3 }, { slot := .local 6, initial := 1, ratio := 3 }, { slot := .local 7, initial := 2, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow12 : RawRow where
  rowOffset := 12
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 12, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 10777112892842897939 }], geometric := [{ slot := .local 0, initial := 2, ratio := 3 }, { slot := .local 1, initial := 3, ratio := 3 }, { slot := .local 2, initial := 1, ratio := 3 }, { slot := .local 3, initial := 1, ratio := 3 }, { slot := .local 4, initial := 4, ratio := 3 }, { slot := .local 5, initial := 6, ratio := 3 }, { slot := .local 6, initial := 2, ratio := 3 }, { slot := .local 7, initial := 2, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow13 : RawRow where
  rowOffset := 13
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 13, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 1771261721914735482 }], geometric := [{ slot := .local 0, initial := 1, ratio := 3 }, { slot := .local 1, initial := 2, ratio := 3 }, { slot := .local 2, initial := 3, ratio := 3 }, { slot := .local 3, initial := 1, ratio := 3 }, { slot := .local 4, initial := 2, ratio := 3 }, { slot := .local 5, initial := 4, ratio := 3 }, { slot := .local 6, initial := 6, ratio := 3 }, { slot := .local 7, initial := 2, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow14 : RawRow where
  rowOffset := 14
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 14, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 9409693344407549465 }], geometric := [{ slot := .local 0, initial := 1, ratio := 3 }, { slot := .local 1, initial := 1, ratio := 3 }, { slot := .local 2, initial := 2, ratio := 3 }, { slot := .local 3, initial := 3, ratio := 3 }, { slot := .local 4, initial := 2, ratio := 3 }, { slot := .local 5, initial := 2, ratio := 3 }, { slot := .local 6, initial := 4, ratio := 3 }, { slot := .local 7, initial := 6, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow15 : RawRow where
  rowOffset := 15
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 15, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 16619731096074499912 }], geometric := [{ slot := .local 0, initial := 3, ratio := 3 }, { slot := .local 1, initial := 1, ratio := 3 }, { slot := .local 2, initial := 1, ratio := 3 }, { slot := .local 3, initial := 2, ratio := 3 }, { slot := .local 4, initial := 6, ratio := 3 }, { slot := .local 5, initial := 2, ratio := 3 }, { slot := .local 6, initial := 2, ratio := 3 }, { slot := .local 7, initial := 4, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow16 : RawRow where
  rowOffset := 16
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 16, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 1922036059108268922 }], geometric := [{ slot := .local 8, initial := 4, ratio := 3 }, { slot := .local 9, initial := 6, ratio := 3 }, { slot := .local 10, initial := 2, ratio := 3 }, { slot := .local 11, initial := 2, ratio := 3 }, { slot := .local 12, initial := 2, ratio := 3 }, { slot := .local 13, initial := 3, ratio := 3 }, { slot := .local 14, initial := 1, ratio := 3 }, { slot := .local 15, initial := 1, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow17 : RawRow where
  rowOffset := 17
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 17, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 2681686362645798986 }], geometric := [{ slot := .local 8, initial := 2, ratio := 3 }, { slot := .local 9, initial := 4, ratio := 3 }, { slot := .local 10, initial := 6, ratio := 3 }, { slot := .local 11, initial := 2, ratio := 3 }, { slot := .local 12, initial := 1, ratio := 3 }, { slot := .local 13, initial := 2, ratio := 3 }, { slot := .local 14, initial := 3, ratio := 3 }, { slot := .local 15, initial := 1, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow18 : RawRow where
  rowOffset := 18
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 18, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 12432722052283819565 }], geometric := [{ slot := .local 8, initial := 2, ratio := 3 }, { slot := .local 9, initial := 2, ratio := 3 }, { slot := .local 10, initial := 4, ratio := 3 }, { slot := .local 11, initial := 6, ratio := 3 }, { slot := .local 12, initial := 1, ratio := 3 }, { slot := .local 13, initial := 1, ratio := 3 }, { slot := .local 14, initial := 2, ratio := 3 }, { slot := .local 15, initial := 3, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow19 : RawRow where
  rowOffset := 19
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 19, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 2826979200512189741 }], geometric := [{ slot := .local 8, initial := 6, ratio := 3 }, { slot := .local 9, initial := 2, ratio := 3 }, { slot := .local 10, initial := 2, ratio := 3 }, { slot := .local 11, initial := 4, ratio := 3 }, { slot := .local 12, initial := 3, ratio := 3 }, { slot := .local 13, initial := 1, ratio := 3 }, { slot := .local 14, initial := 1, ratio := 3 }, { slot := .local 15, initial := 2, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow20 : RawRow where
  rowOffset := 20
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 20, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 5080805286413226676 }], geometric := [{ slot := .local 8, initial := 2, ratio := 3 }, { slot := .local 9, initial := 3, ratio := 3 }, { slot := .local 10, initial := 1, ratio := 3 }, { slot := .local 11, initial := 1, ratio := 3 }, { slot := .local 12, initial := 4, ratio := 3 }, { slot := .local 13, initial := 6, ratio := 3 }, { slot := .local 14, initial := 2, ratio := 3 }, { slot := .local 15, initial := 2, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow21 : RawRow where
  rowOffset := 21
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 21, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 16827966425431695029 }], geometric := [{ slot := .local 8, initial := 1, ratio := 3 }, { slot := .local 9, initial := 2, ratio := 3 }, { slot := .local 10, initial := 3, ratio := 3 }, { slot := .local 11, initial := 1, ratio := 3 }, { slot := .local 12, initial := 2, ratio := 3 }, { slot := .local 13, initial := 4, ratio := 3 }, { slot := .local 14, initial := 6, ratio := 3 }, { slot := .local 15, initial := 2, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow22 : RawRow where
  rowOffset := 22
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 22, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 9196241087337510154 }], geometric := [{ slot := .local 8, initial := 1, ratio := 3 }, { slot := .local 9, initial := 1, ratio := 3 }, { slot := .local 10, initial := 2, ratio := 3 }, { slot := .local 11, initial := 3, ratio := 3 }, { slot := .local 12, initial := 2, ratio := 3 }, { slot := .local 13, initial := 2, ratio := 3 }, { slot := .local 14, initial := 4, ratio := 3 }, { slot := .local 15, initial := 6, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow23 : RawRow where
  rowOffset := 23
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 23, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 2350771591198563053 }], geometric := [{ slot := .local 8, initial := 3, ratio := 3 }, { slot := .local 9, initial := 1, ratio := 3 }, { slot := .local 10, initial := 1, ratio := 3 }, { slot := .local 11, initial := 2, ratio := 3 }, { slot := .local 12, initial := 6, ratio := 3 }, { slot := .local 13, initial := 2, ratio := 3 }, { slot := .local 14, initial := 2, ratio := 3 }, { slot := .local 15, initial := 4, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow24 : RawRow where
  rowOffset := 24
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 24, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 2989012136977041732 }], geometric := [{ slot := .local 16, initial := 4, ratio := 3 }, { slot := .local 17, initial := 6, ratio := 3 }, { slot := .local 18, initial := 2, ratio := 3 }, { slot := .local 19, initial := 2, ratio := 3 }, { slot := .local 20, initial := 2, ratio := 3 }, { slot := .local 21, initial := 3, ratio := 3 }, { slot := .local 22, initial := 1, ratio := 3 }, { slot := .local 23, initial := 1, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow25 : RawRow where
  rowOffset := 25
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 25, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 4359939046747977080 }], geometric := [{ slot := .local 16, initial := 2, ratio := 3 }, { slot := .local 17, initial := 4, ratio := 3 }, { slot := .local 18, initial := 6, ratio := 3 }, { slot := .local 19, initial := 2, ratio := 3 }, { slot := .local 20, initial := 1, ratio := 3 }, { slot := .local 21, initial := 2, ratio := 3 }, { slot := .local 22, initial := 3, ratio := 3 }, { slot := .local 23, initial := 1, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow26 : RawRow where
  rowOffset := 26
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 26, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 16089932437481530267 }], geometric := [{ slot := .local 16, initial := 2, ratio := 3 }, { slot := .local 17, initial := 2, ratio := 3 }, { slot := .local 18, initial := 4, ratio := 3 }, { slot := .local 19, initial := 6, ratio := 3 }, { slot := .local 20, initial := 1, ratio := 3 }, { slot := .local 21, initial := 1, ratio := 3 }, { slot := .local 22, initial := 2, ratio := 3 }, { slot := .local 23, initial := 3, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow27 : RawRow where
  rowOffset := 27
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 27, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 6601984573273403484 }], geometric := [{ slot := .local 16, initial := 6, ratio := 3 }, { slot := .local 17, initial := 2, ratio := 3 }, { slot := .local 18, initial := 2, ratio := 3 }, { slot := .local 19, initial := 4, ratio := 3 }, { slot := .local 20, initial := 3, ratio := 3 }, { slot := .local 21, initial := 1, ratio := 3 }, { slot := .local 22, initial := 1, ratio := 3 }, { slot := .local 23, initial := 2, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow28 : RawRow where
  rowOffset := 28
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 28, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 13005272261058756234 }], geometric := [{ slot := .local 16, initial := 2, ratio := 3 }, { slot := .local 17, initial := 3, ratio := 3 }, { slot := .local 18, initial := 1, ratio := 3 }, { slot := .local 19, initial := 1, ratio := 3 }, { slot := .local 20, initial := 4, ratio := 3 }, { slot := .local 21, initial := 6, ratio := 3 }, { slot := .local 22, initial := 2, ratio := 3 }, { slot := .local 23, initial := 2, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow29 : RawRow where
  rowOffset := 29
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 29, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 17128237926164276121 }], geometric := [{ slot := .local 16, initial := 1, ratio := 3 }, { slot := .local 17, initial := 2, ratio := 3 }, { slot := .local 18, initial := 3, ratio := 3 }, { slot := .local 19, initial := 1, ratio := 3 }, { slot := .local 20, initial := 2, ratio := 3 }, { slot := .local 21, initial := 4, ratio := 3 }, { slot := .local 22, initial := 6, ratio := 3 }, { slot := .local 23, initial := 2, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow30 : RawRow where
  rowOffset := 30
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 30, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 8240789415616872849 }], geometric := [{ slot := .local 16, initial := 1, ratio := 3 }, { slot := .local 17, initial := 1, ratio := 3 }, { slot := .local 18, initial := 2, ratio := 3 }, { slot := .local 19, initial := 3, ratio := 3 }, { slot := .local 20, initial := 2, ratio := 3 }, { slot := .local 21, initial := 2, ratio := 3 }, { slot := .local 22, initial := 4, ratio := 3 }, { slot := .local 23, initial := 6, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow31 : RawRow where
  rowOffset := 31
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 31, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 8676316357341090631 }], geometric := [{ slot := .local 16, initial := 3, ratio := 3 }, { slot := .local 17, initial := 1, ratio := 3 }, { slot := .local 18, initial := 1, ratio := 3 }, { slot := .local 19, initial := 2, ratio := 3 }, { slot := .local 20, initial := 6, ratio := 3 }, { slot := .local 21, initial := 2, ratio := 3 }, { slot := .local 22, initial := 2, ratio := 3 }, { slot := .local 23, initial := 4, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow32 : RawRow where
  rowOffset := 32
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 32, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 7482194551502142718 }], geometric := [{ slot := .local 24, initial := 4, ratio := 3 }, { slot := .local 25, initial := 6, ratio := 3 }, { slot := .local 26, initial := 2, ratio := 3 }, { slot := .local 27, initial := 2, ratio := 3 }, { slot := .local 28, initial := 2, ratio := 3 }, { slot := .local 29, initial := 3, ratio := 3 }, { slot := .local 30, initial := 1, ratio := 3 }, { slot := .local 31, initial := 1, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow33 : RawRow where
  rowOffset := 33
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 33, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 3471957803411196592 }], geometric := [{ slot := .local 24, initial := 17, ratio := 3 }, { slot := .local 25, initial := 15, ratio := 3 }, { slot := .local 26, initial := 19, ratio := 3 }, { slot := .local 27, initial := 19, ratio := 3 }, { slot := .local 28, initial := 19, ratio := 3 }, { slot := .local 29, initial := 18, ratio := 3 }, { slot := .local 30, initial := 20, ratio := 3 }, { slot := .local 31, initial := 20, ratio := 3 }, { slot := .local 32, initial := 18446744069414584320, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow34 : RawRow where
  rowOffset := 34
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 34, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 8846669050136897522 }], geometric := [{ slot := .local 24, initial := 9223372034707292279, ratio := 3 }, { slot := .local 25, initial := 115, ratio := 3 }, { slot := .local 26, initial := 9223372034707292300, ratio := 3 }, { slot := .local 27, initial := 9223372034707292295, ratio := 3 }, { slot := .local 28, initial := 9223372034707292279, ratio := 3 }, { slot := .local 29, initial := 9223372034707292293, ratio := 3 }, { slot := .local 30, initial := 9223372034707292291, ratio := 3 }, { slot := .local 31, initial := 119, ratio := 3 }, { slot := .local 32, initial := 7, ratio := 3 }, { slot := .local 33, initial := 18446744069414584320, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow35 : RawRow where
  rowOffset := 35
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 35, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 4431017908497072775 }], geometric := [{ slot := .local 24, initial := 13835058052060939140, ratio := 3 }, { slot := .local 25, initial := 855, ratio := 3 }, { slot := .local 26, initial := 4611686017353647104, ratio := 3 }, { slot := .local 27, initial := 4611686017353647098, ratio := 3 }, { slot := .local 28, initial := 4611686017353647047, ratio := 3 }, { slot := .local 29, initial := 4611686017353647101, ratio := 3 }, { slot := .local 30, initial := 13835058052060939233, ratio := 3 }, { slot := .local 31, initial := 963, ratio := 3 }, { slot := .local 32, initial := 48, ratio := 3 }, { slot := .local 33, initial := 7, ratio := 3 }, { slot := .local 34, initial := 18446744069414584320, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow36 : RawRow where
  rowOffset := 36
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 36, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 14382646627736292998 }], geometric := [{ slot := .local 24, initial := 11529215043384121902, ratio := 3 }, { slot := .local 25, initial := 9223372034707298633, ratio := 3 }, { slot := .local 26, initial := 16140901060737769005, ratio := 3 }, { slot := .local 27, initial := 6917529026030476729, ratio := 3 }, { slot := .local 28, initial := 11529215043384122279, ratio := 3 }, { slot := .local 29, initial := 11529215043384122914, ratio := 3 }, { slot := .local 30, initial := 6917529026030476562, ratio := 3 }, { slot := .local 31, initial := 7073, ratio := 3 }, { slot := .local 32, initial := 9223372034707292529, ratio := 3 }, { slot := .local 33, initial := 48, ratio := 3 }, { slot := .local 34, initial := 7, ratio := 3 }, { slot := .local 35, initial := 18446744069414584320, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow37 : RawRow where
  rowOffset := 37
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 37, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 15636596632746594248 }], geometric := [{ slot := .local 24, initial := 8070450530368931442, ratio := 3 }, { slot := .local 25, initial := 13835058052060987027, ratio := 3 }, { slot := .local 26, initial := 5764607521692115680, ratio := 3 }, { slot := .local 27, initial := 5764607521692115045, ratio := 3 }, { slot := .local 28, initial := 1152921504338465777, ratio := 3 }, { slot := .local 29, initial := 1152921504338469915, ratio := 3 }, { slot := .local 30, initial := 12682136547722582908, ratio := 3 }, { slot := .local 31, initial := 4611686017353699950, ratio := 3 }, { slot := .local 32, initial := 2753, ratio := 3 }, { slot := .local 33, initial := 9223372034707292529, ratio := 3 }, { slot := .local 34, initial := 48, ratio := 3 }, { slot := .local 35, initial := 7, ratio := 3 }, { slot := .local 36, initial := 18446744069414584320, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow38 : RawRow where
  rowOffset := 38
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 38, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 14521990061611210983 }], geometric := [{ slot := .local 24, initial := 5188146769523234387, ratio := 3 }, { slot := .local 25, initial := 6917529026030837656, ratio := 3 }, { slot := .local 26, initial := 6341068273861702094, ratio := 3 }, { slot := .local 27, initial := 13258597299892165716, ratio := 3 }, { slot := .local 28, initial := 5188146769523258872, ratio := 3 }, { slot := .local 29, initial := 9799832786876938794, ratio := 3 }, { slot := .local 30, initial := 13258597299892156602, ratio := 3 }, { slot := .local 31, initial := 13835058052061343481, ratio := 3 }, { slot := .local 32, initial := 6917529026030489969, ratio := 3 }, { slot := .local 33, initial := 2753, ratio := 3 }, { slot := .local 34, initial := 9223372034707292529, ratio := 3 }, { slot := .local 35, initial := 48, ratio := 3 }, { slot := .local 36, initial := 7, ratio := 3 }, { slot := .local 37, initial := 18446744069414584320, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow39 : RawRow where
  rowOffset := 39
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 39, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 4351091752509404379 }], geometric := [{ slot := .local 24, initial := 10088063162963992392, ratio := 3 }, { slot := .local 25, initial := 12682136547725308641, ratio := 3 }, { slot := .local 26, initial := 7205759402118382269, ratio := 3 }, { slot := .local 27, initial := 16429131436825635117, ratio := 3 }, { slot := .local 28, initial := 17582052941163858819, ratio := 3 }, { slot := .local 29, initial := 8358680906456813683, ratio := 3 }, { slot := .local 30, initial := 18158513693333183134, ratio := 3 }, { slot := .local 31, initial := 10376293539048767280, ratio := 3 }, { slot := .local 32, initial := 157158, ratio := 3 }, { slot := .local 33, initial := 6917529026030489969, ratio := 3 }, { slot := .local 34, initial := 2753, ratio := 3 }, { slot := .local 35, initial := 9223372034707292529, ratio := 3 }, { slot := .local 36, initial := 48, ratio := 3 }, { slot := .local 37, initial := 7, ratio := 3 }, { slot := .local 38, initial := 18446744069414584320, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow40 : RawRow where
  rowOffset := 40
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 40, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 14119848206371842921 }], geometric := [{ slot := .local 24, initial := 4179340453248561109, ratio := 3 }, { slot := .local 25, initial := 13258597299912739059, ratio := 3 }, { slot := .local 26, initial := 5620492333674754551, ratio := 3 }, { slot := .local 27, initial := 2738188572828421357, ratio := 3 }, { slot := .local 28, initial := 14555633992295690398, ratio := 3 }, { slot := .local 29, initial := 6485183461928707528, ratio := 3 }, { slot := .local 30, initial := 15420325120550429845, ratio := 3 }, { slot := .local 31, initial := 1152921504361529894, ratio := 3 }, { slot := .local 32, initial := 10952754291216096742, ratio := 3 }, { slot := .local 33, initial := 157158, ratio := 3 }, { slot := .local 34, initial := 6917529026030489969, ratio := 3 }, { slot := .local 35, initial := 2753, ratio := 3 }, { slot := .local 36, initial := 9223372034707292529, ratio := 3 }, { slot := .local 37, initial := 48, ratio := 3 }, { slot := .local 38, initial := 7, ratio := 3 }, { slot := .local 39, initial := 18446744069414584320, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow41 : RawRow where
  rowOffset := 41
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 41, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 528205008764728916 }], geometric := [{ slot := .local 24, initial := 5692549927835698344, ratio := 3 }, { slot := .local 25, initial := 13546827676134937103, ratio := 3 }, { slot := .local 26, initial := 11313042261509384786, ratio := 3 }, { slot := .local 27, initial := 17077649783199168880, ratio := 3 }, { slot := .local 28, initial := 6413125868058051042, ratio := 3 }, { slot := .local 29, initial := 15636497902779558421, ratio := 3 }, { slot := .local 30, initial := 9439544816953269319, ratio := 3 }, { slot := .local 31, initial := 7205759402289663444, ratio := 3 }, { slot := .local 32, initial := 12682136547731488632, ratio := 3 }, { slot := .local 33, initial := 10952754291216096742, ratio := 3 }, { slot := .local 34, initial := 157158, ratio := 3 }, { slot := .local 35, initial := 6917529026030489969, ratio := 3 }, { slot := .local 36, initial := 2753, ratio := 3 }, { slot := .local 37, initial := 9223372034707292529, ratio := 3 }, { slot := .local 38, initial := 48, ratio := 3 }, { slot := .local 39, initial := 7, ratio := 3 }, { slot := .local 40, initial := 18446744069414584320, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRow42 : RawRow where
  rowOffset := 42
  ports := [
    { explicit := [], geometric := [] }
  , { explicit := [{ column := .selector, coefficient := 1 }], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [{ slot := .local 42, initial := 1, ratio := 3 }] }
  , { explicit := [{ column := .one, coefficient := 15379406877060454284 }], geometric := [{ slot := .local 24, initial := 5800636319946698146, ratio := 3 }, { slot := .local 25, initial := 1585267069662848642, ratio := 3 }, { slot := .local 26, initial := 17978369709702105566, ratio := 3 }, { slot := .local 27, initial := 9187343239104462052, ratio := 3 }, { slot := .local 28, initial := 16753390611243219645, ratio := 3 }, { slot := .local 29, initial := 7818248952728291443, ratio := 3 }, { slot := .local 30, initial := 13222568504259340283, ratio := 3 }, { slot := .local 31, initial := 7782220155602383499, ratio := 3 }, { slot := .local 32, initial := 8502796094563459664, ratio := 3 }, { slot := .local 33, initial := 12682136547731488632, ratio := 3 }, { slot := .local 34, initial := 10952754291216096742, ratio := 3 }, { slot := .local 35, initial := 157158, ratio := 3 }, { slot := .local 36, initial := 6917529026030489969, ratio := 3 }, { slot := .local 37, initial := 2753, ratio := 3 }, { slot := .local 38, initial := 9223372034707292529, ratio := 3 }, { slot := .local 39, initial := 48, ratio := 3 }, { slot := .local 40, initial := 7, ratio := 3 }, { slot := .local 41, initial := 18446744069414584320, ratio := 3 }] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  , { explicit := [], geometric := [] }
  ]

def rawRows0 : List RawRow := [
  rawRow00
, rawRow01
, rawRow02
, rawRow03
, rawRow04
, rawRow05
, rawRow06
, rawRow07
, rawRow08
, rawRow09
, rawRow10
, rawRow11
, rawRow12
, rawRow13
, rawRow14
, rawRow15
, rawRow16
, rawRow17
, rawRow18
, rawRow19
, rawRow20
, rawRow21
, rawRow22
, rawRow23
, rawRow24
, rawRow25
, rawRow26
, rawRow27
, rawRow28
, rawRow29
, rawRow30
, rawRow31
, rawRow32
, rawRow33
, rawRow34
, rawRow35
, rawRow36
, rawRow37
, rawRow38
, rawRow39
, rawRow40
, rawRow41
, rawRow42
]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeaf
