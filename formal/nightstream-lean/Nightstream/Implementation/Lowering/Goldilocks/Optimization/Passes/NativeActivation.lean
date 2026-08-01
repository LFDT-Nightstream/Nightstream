import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.NativeCcsActivatedBridge
import Nightstream.Implementation.Lowering.Goldilocks.Optimization.Boundary
import Nightstream.Implementation.Lowering.Goldilocks.Optimization.NativeCcs

/-!
Contract: replace one residual-activated R1CS occurrence with one native
selected-CCS occurrence.

Assurance tier: model-level.

Owns: explicit residual recovery, exact acceptance in both directions under
the enclosing selector-bit fact, exact protected-boundary preservation, and
the degree-two to degree-three transition.

Does not own: the enclosing selector constraint, receipt placement, a
protocol call, a selected application, a manifest, Rust, or a security
reduction.

Emits constraints: one native CCS row per intrinsic source row. It removes
the activation residual rows and residual allocations from this occurrence.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.Optimization.Passes.NativeActivation

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector
open Nightstream.Implementation.Lowering.Goldilocks.Optimization
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

private abbrev Field := Nightstream.SuperNeo.Concrete.F
abbrev Assignment := ColumnId -> Field

def boundaryIds (columns : Boundary.Columns) : List ColumnId :=
  columns.committedColumns ++ columns.publicColumns ++
    columns.outputColumns ++ columns.transcriptColumns

/-- The enclosing branch owns this fact. It is not a new emitted row. -/
def SelectorBit (active : ColumnId) (assignment : Assignment) : Prop :=
  assignment active = 0 \/ assignment active = 1

def sourceSystem
    (owner : PhysicalOwner)
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (boundary : Boundary.Columns) :
    Optimization.System Assignment Boundary.Values where
  Accepts := fun assignment =>
    SelectorBit active assignment /\
      Goldilocks.Satisfies
        (ActivatedRawProgram.rows owner active source residuals)
        assignment
  observe := Boundary.values boundary
  degree := 2

def targetSystem
    (owner : PhysicalOwner)
    (active : ColumnId)
    (source : List Row)
    (boundary : Boundary.Columns) :
    Optimization.System Assignment Boundary.Values where
  Accepts := fun assignment =>
    SelectorBit active assignment /\
      NativeCcsSelector.Satisfies
        (NativeCcsSelector.select active (ownRows owner source))
        assignment
  observe := Boundary.values boundary
  degree := NativeCcsSelector.polynomialDegree

private theorem recovered_active
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (assignment : Assignment)
    (activeFresh : active ∉ residuals) :
    ActivatedRawProgram.complete assignment source residuals active =
      assignment active :=
  ActivatedRawProgram.complete_changesOnly assignment source residuals
    active activeFresh

private theorem recovered_observes
    (source : List Row)
    (residuals : List ColumnId)
    (boundary : Boundary.Columns)
    (assignment : Assignment)
    (boundaryFresh :
      IdsDisjoint residuals (boundaryIds boundary)) :
    Boundary.values boundary
        (ActivatedRawProgram.complete assignment source residuals) =
      Boundary.values boundary assignment := by
  apply Boundary.values_eq_of_agrees
  intro column member
  exact
    ActivatedRawProgram.complete_agreesOn assignment source residuals
      (boundaryIds boundary) boundaryFresh column (by
        simpa [boundaryIds] using member)

/-- Certified local replacement of the residual activation wrapper.

The selector-bit premise is explicit because native selection and residual
activation are equivalent only at selector values zero and one. The enclosing
branch program already emits the selector constraint. -/
def replacement
    (owner : PhysicalOwner)
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (boundary : Boundary.Columns)
    (lengthEqual : source.length = residuals.length)
    (residualsNodup : residuals.Nodup)
    (residualsFresh :
      IdsDisjoint residuals
        (source.flatMap fun row => row.columnIds))
    (activeFresh : active ∉ residuals)
    (boundaryFresh :
      IdsDisjoint residuals (boundaryIds boundary)) :
    Optimization.Replacement
      (sourceSystem owner active source residuals boundary)
      (targetSystem owner active source boundary)
      3 where
  recover := fun assignment =>
    ActivatedRawProgram.complete assignment source residuals
  derive := fun assignment => assignment
  sound := by
    intro assignment accepted
    have activePreserved :=
      recovered_active active source residuals assignment activeFresh
    constructor
    · rcases accepted.1 with activeZero | activeOne
      · exact Or.inl (by simpa [activePreserved] using activeZero)
      · exact Or.inr (by simpa [activePreserved] using activeOne)
    · rcases accepted.1 with activeZero | activeOne
      · apply
          (satisfies_ownRows_iff owner
            (ActivatedRawProgram.rawRows active source residuals)
            (ActivatedRawProgram.complete assignment source residuals)).2
        exact
          ActivatedRawProgram.inactive_complete active source residuals
            assignment lengthEqual residualsNodup residualsFresh
            activeZero activeFresh
      · have sourceOwned :
            Goldilocks.Satisfies (ownRows owner source) assignment :=
          NativeCcsSelector.active_sound active (ownRows owner source)
            assignment activeOne accepted.2
        have sourceRaw : RawSatisfies source assignment :=
          (satisfies_ownRows_iff owner source assignment).1 sourceOwned
        apply
          (satisfies_ownRows_iff owner
            (ActivatedRawProgram.rawRows active source residuals)
            (ActivatedRawProgram.complete assignment source residuals)).2
        exact
          ActivatedRawProgram.active_complete active source residuals
            assignment lengthEqual residualsNodup residualsFresh sourceRaw
  complete := by
    intro assignment accepted
    exact ⟨accepted.1,
      NativeCcsActivatedBridge.selected_of_activated
        owner active source residuals assignment lengthEqual accepted.2⟩
  recover_observes := by
    intro assignment _
    exact
      recovered_observes source residuals boundary assignment boundaryFresh
  derive_observes := fun _ _ => rfl
  source_degree := by
    change 2 <= 3
    omega
  target_degree := by
    change NativeCcsSelector.polynomialDegree <= 3
    rw [NativeCcsSelector.polynomialDegree_exact]
    exact Nat.le_refl 3

end Nightstream.Implementation.Lowering.Goldilocks.Optimization.Passes.NativeActivation
