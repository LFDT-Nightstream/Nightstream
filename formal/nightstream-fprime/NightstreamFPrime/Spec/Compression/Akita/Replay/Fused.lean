import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightTrace

/-! The end check of the fused sum-check of one fold (Akita revision `d50f700`:
`AkitaStage2Verifier::expected_output_claim` and `validate_stage2_replay` of
`akita-verifier/src/stages/stage2.rs`; ePrint 2026/1983, (125) and Section 7.4; the final check of
`fusedCheck` in `Akita/FoldCheck.lean`).

`FusedEnd` has the values of the end of the fused sum-check of one level: the end claim of the
rounds, the message `v_W`, the coefficients `γ` (`stage2_batch`), `ζ_bin` and `η_norm`, and the
four evaluations at `r_2`: the relation weight `m_{τ_1}(r_2)`, the norm weight, `ẽq(r_virt, r_2)`
and the binary-support weight. `FusedEnd.holds` is the check of the code:

    end = v·(m(r_2) + η·K(r_2)) + (γ·ẽq(r_virt, r_2) + ζ·B(r_2))·v·(v + 1).

The code sums the same terms in another order (the virtual, relation, trace and physical L2
oracles); the field has one sum. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

/-- The values of the end check of the fused sum-check of one level. -/
structure FusedEnd where
  output : Ext
  value : Ext
  batch : Ext
  binary : Ext
  norm : Ext
  weight : Ext
  normWeight : Ext
  virt : Ext
  support : Ext
deriving Repr

namespace FusedEnd

/-- The expected end claim (`expected_output_claim`). -/
def expected (f : FusedEnd) : Ext :=
  f.value * (f.weight + f.norm * f.normWeight) +
    (f.batch * f.virt + f.binary * f.support) * (f.value * (f.value + Ext.one))

/-- The check `output == expected_output_claim(r_2)` (`stage2.rs:442`). -/
def holds (f : FusedEnd) : Bool := f.output.equals f.expected

/-- `f` when the check holds. -/
def passing (f : FusedEnd) : Option FusedEnd := if f.holds then some f else none

end FusedEnd

/-- The values of the replay at the end of the fused sum-check of a trace level. -/
structure TraceValues where
  euclidean : Bool
  compressed : Bool
  tail : List Ext
  point : List Ext
  eta : Ext
  tau1 : List Ext
  r2 : List Ext
  virtual : List Ext
  alpha : Ext
  output : Ext
  value : Ext
  batch : Ext
  binary : Ext
  norm : Ext

/-- The values of the end check of the fused sum-check at a trace level with the shapes `O` and `fold`: the
weight of `traceTerms` and `traceScalar` with the code sign of the rows (104), with the factor
`θ = tensorFactor tail η ρ` of the tensor reduction of the level, the norm weight at a Euclidean
level and the binary-support weight at a level with compression digits. -/
def traceFused {S : Shape} (O : OpeningShape S) (fold : Descriptor) (setup : Array Fq)
    (challenges : List (Array Fq)) (V : TraceValues) : FusedEnd :=
  let I : TraceInputs := ⟨setup, challenges, V.tau1, V.point, tensorFactor V.tail V.eta V.point⟩
  let eq := (CellEq.half V.r2).get
  let virt := (CellEq.half V.virtual).get
  { output := V.output
    value := V.value
    batch := V.batch
    binary := if V.compressed then V.binary else Ext.zero
    norm := if V.euclidean then V.norm else Ext.zero
    weight := termsValue V.alpha eq (traceTerms (O := O) (fold := fold) I) +
      traceScalar (O := O) (fold := fold) I eq
    normWeight := if V.euclidean then normValue S fold virt eq else Ext.zero
    virt := eqPoints V.virtual V.r2
    support := if V.compressed then supportValue O fold virt eq else Ext.zero }

end NightstreamFPrime.Spec.Compression.Akita.Replay
