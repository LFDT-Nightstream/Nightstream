import NightstreamFPrime.Export.BoundedDepth.FinalKey20All
import NightstreamFPrime.Spec.Compression.Akita.Key20Instances

/-! The instances of Module-SIS of the final verifier over the Akita opening of `key20.aks`
(SPEC.md, Algorithm 10; Akita, ePrint 2026/1983, revision of 2026-11-23, Table 3, Lemma 10.20
and Appendix G).

`Opening.reduced` states that the list `Key20.instances` is complete for the event `Solved` of
`FinalAkita.breakChance`: each output of the extractor that is a solution of Module-SIS gives a
solution of an instance of the list. `Opening.reduced_shared` states that for matrices that are the
views of one setup vector `S`, each such output gives a solution for the view of `S` of an
instance of `Key20.views`. Lean has no statement on the hardness of an instance. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.FinalKey20

open NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Export.BoundedDepth.FinalVerifier (Mode)
open NightstreamFPrime.Export.BoundedDepth.FinalAkita
open Key20 (root)

attribute [local instance low] Classical.propDecidable

attribute [local instance] NightstreamFPrime.Export.BoundedDepth.FinalKey20.instAlgebraZqK
  NightstreamFPrime.Export.BoundedDepth.FinalKey20.instDegree2K

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  {mode : Mode}
  {setup : FinalVerifier.Setup mode logicalWidth publicFits} {packageWidths : Fin 2 → ℕ}
  {step : AppState → AppWitness → AppState} {cap : ℕ} {control : AppState → AppWitness → Fin 2}
  {Wire Coord : Type} {total : ℕ} {read : Reader Wire}
  {head : Wire → First logicalWidth publicFits}
  [Fintype Coord] [DecidableEq Coord]
  {o : Opening logicalWidth publicFits Wire Coord total}

/-- The list `Key20.instances` is complete for the solutions of Module-SIS of the final verifier
over the Akita opening of `key20.aks` (Akita Lemma 10.20 and Table 3). `out` is an output of the
extractor with `Solved`, the event of `FinalAkita.breakChance`: a solution for the key commitment
from `tableCollision`, or a witness of `R_bind` of the opening. Then a matrix of the opening has a
solution that gives a solution of an instance of the list (`Key20.Reduced`). `mass`, `weight₁`
and `weight₂`: the opening has the norms `‖c‖_1` of the fold challenges of the root and of the
levels 1 and 2 that `Opening.Drawn` gives. The radius of the reduction for the key commitment is
`η_A` of the key group. -/
theorem Opening.reduced (mass : o.mass = Key20.rootMass) (weight₁ : o.weight₁ = 51)
    (weight₂ : o.weight₂ = 31)
    {out : _ ⊕ _}
    (solved : Solved (o.tower setup packageWidths step cap control read head) o.key o.honest
      ((root.fold 0).innerBound (root.shape 0)) out) :
    Key20.Reduced Key20.instances o.key o.openKey o.key₁ o.openKey₁ o.key₂ o.openKey₂ o.key₃
      o.openKey₃ o.key₄ o.openKey₄ o.key₅ o.openKey₅ o.terminal.matrix := by
  rcases out with certificate | brk
  · obtain ⟨ω, -, found⟩ := solved
    obtain ⟨z, -⟩ := Option.isSome_iff_exists.mp found
    exact Or.inl (Or.inl ⟨0, Key20.reduced_key le_rfl z⟩)
  · exact Key20.reduced mass weight₁ weight₂ brk solved

/-- `Opening.reduced` for an opening whose matrices are the views of one flat setup vector `S`
(`Key20.Shared`): each output of the extractor with `Solved` gives a solution of Module-SIS for
the view `M_I(S)` of an instance `I` of `Key20.views`, with the width of its matrix and the bound
of the witness of its level (`Setup.Solved`). -/
theorem Opening.reduced_shared (mass : o.mass = Key20.rootMass) (weight₁ : o.weight₁ = 51)
    (weight₂ : o.weight₂ = 31) {N : ℕ} {S : Fin N → Zq}
    (keys : Key20.Shared S o.key o.openKey o.key₁ o.openKey₁ o.key₂ o.openKey₂ o.key₃ o.openKey₃
      o.key₄ o.openKey₄ o.key₅ o.openKey₅ o.terminal.matrix)
    {out : _ ⊕ _}
    (solved : Solved (o.tower setup packageWidths step cap control read head) o.key o.honest
      ((root.fold 0).innerBound (root.shape 0)) out) :
    Setup.Solved Key20.views S := by
  rcases out with certificate | brk
  · obtain ⟨ω, -, found⟩ := solved
    obtain ⟨z, -⟩ := Option.isSome_iff_exists.mp found
    exact Key20.reduced_key_shared (keys.1.1 0) le_rfl z
  · exact Key20.reduced_shared mass weight₁ weight₂ keys brk solved

end NightstreamFPrime.Export.BoundedDepth.FinalKey20
