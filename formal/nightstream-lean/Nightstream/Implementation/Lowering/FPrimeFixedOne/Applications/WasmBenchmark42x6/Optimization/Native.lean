import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Cost
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4FootprintAudit
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Deployment
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStepCost
import Nightstream.Implementation.Lowering.Goldilocks.Optimization.NativeCcs
import Nightstream.Implementation.Lowering.Goldilocks.Optimization.NativeCounts
import Nightstream.Implementation.Lowering.Goldilocks.Optimization.Passes.NativeActivation

/-!
Contract: selected native-CCS optimization for the 42-times-6 WASM
benchmark.

Assurance tier: model-level.

Owns: the selected occurrence replacement, the complete native Step program,
its proof-free manifest, and exact optimized rows and column counts.

Does not own: a concrete Ajtai setup value, JSON, Rust emission, a general
WASM compiler, or a security reduction.

Emits constraints: the native four-matrix Step program already constructed
by the selected fixed-one deployment.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Optimization.Native

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.Implementation.Lowering.Goldilocks.Optimization
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Cost
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev Setup :=
  RelationSetup CurrentM4Cost.dimensions CurrentM4Cost.commitmentRows

noncomputable def nifsCertificate (setup : Setup) :=
  ConcreteNifsCanonicalCertification.nifs
    setup
    (defaultRunning CurrentM4Cost.dimensions CurrentM4Cost.commitmentRows)
    (machine benchmarkHashPlan)
    (terminalRelations
      CurrentM4Cost.dimensions CurrentM4Cost.commitmentRows)
    (terminalChecks
      CurrentM4Cost.dimensions CurrentM4Cost.commitmentRows)
    (widths setup)
    (selectedFootprints setup)
    (deployment setup)

noncomputable def application (setup : Setup) :=
  (deployment setup).application.phase4

noncomputable def profile (setup : Setup) :=
  (application setup).profile

noncomputable def stepRecipe (setup : Setup) :=
  (deployment setup).step

noncomputable def defaultAdmissible (setup : Setup) :=
  (deployment setup).defaultRunningAdmissible

theorem application_eq (setup : Setup) :
    application setup = (deployment setup).application.phase4 :=
  rfl

theorem profile_eq (setup : Setup) :
    profile setup = (deployment setup).application.phase4.profile :=
  rfl

theorem nifsCertificate_operational_eq (setup : Setup) :
    (nifsCertificate setup).operational =
      ConcreteNifsCanonicalOperationalProfile.operational
        setup
        (defaultRunning CurrentM4Cost.dimensions
          CurrentM4Cost.commitmentRows)
        (machine benchmarkHashPlan)
        (terminalRelations
          CurrentM4Cost.dimensions CurrentM4Cost.commitmentRows)
        (terminalChecks
          CurrentM4Cost.dimensions CurrentM4Cost.commitmentRows)
        (widths setup)
        (selectedFootprints setup)
        (deployment setup).application :=
  rfl

noncomputable def frame (setup : Setup) :=
  (ConcreteNifsNativeCcsStep.invokePlan
    (application setup)
    (nifsCertificate setup)
    (stepRecipe setup)
    (defaultAdmissible setup)).frame

noncomputable def intrinsicRows (setup : Setup) : List Row :=
  ConcreteNifsRawProgram.rawRows
    (profile setup)
    (nifsCertificate setup).operational
    (frame setup)

noncomputable def residuals (setup : Setup) : List ColumnId :=
  ConcreteNifsActivatedProgram.residuals
    (profile setup)
    (nifsCertificate setup).operational
    (frame setup)

/-- The local pass preserves every visible value of the recursive NIFS
occurrence. Outputs keep their output role. The constant, selector, and
caller context are protected as transcript-visible input state. -/
noncomputable def occurrenceBoundary (setup : Setup) : Boundary.Columns where
  committedColumns := []
  publicColumns := []
  outputColumns := (frame setup).outputs.ids
  transcriptColumns :=
    [(frame setup).one, (frame setup).active] ++
      (frame setup).contextBundles.ids

private theorem idsDisjoint_append_swap
    {left right residualColumns : List ColumnId}
    (disjoint : IdsDisjoint residualColumns (left ++ right)) :
    IdsDisjoint residualColumns (right ++ left) := by
  intro column residualMember member
  rcases List.mem_append.1 member with inRight | inLeft
  · exact disjoint column residualMember
      (List.mem_append_right left inRight)
  · exact disjoint column residualMember
      (List.mem_append_left right inLeft)

/-- Exact selected occurrence replacement used by the complete native Step. -/
noncomputable def occurrenceReplacement (setup : Setup) :
    Replacement
      (Passes.NativeActivation.sourceSystem
        (frame setup).owner (frame setup).active
        (intrinsicRows setup) (residuals setup)
        (occurrenceBoundary setup))
      (Passes.NativeActivation.targetSystem
        (frame setup).owner (frame setup).active
        (intrinsicRows setup) (occurrenceBoundary setup))
      3 :=
  Passes.NativeActivation.replacement
    (frame setup).owner
    (frame setup).active
    (intrinsicRows setup)
    (residuals setup)
    (occurrenceBoundary setup)
    (by
      exact
        (ConcreteNifsActivatedProgram.residuals_length
          (profile setup)
          (nifsCertificate setup).operational
          (nifsCertificate setup).footprint
          (frame setup)).symm)
    (ConcreteNifsActivatedProgram.residuals_nodup
      (profile setup)
      (nifsCertificate setup).operational
      (frame setup))
    (ConcreteNifsActivatedProgram.residuals_fresh
      (profile setup)
      (nifsCertificate setup).operational
      (nifsCertificate setup).footprint
      (frame setup))
    (ConcreteNifsActivatedProgram.active_not_residual
      (profile setup)
      (nifsCertificate setup).operational
      (frame setup))
    (by
      have visibleDisjoint :
          IdsDisjoint
            (residuals setup)
            (([(frame setup).one, (frame setup).active] ++
                (frame setup).contextBundles.ids) ++
              (frame setup).outputs.ids) := by
        simpa only [CallFrame.visibleIds] using
          (ConcreteNifsActivatedProgram.residuals_disjoint_visible
          (profile setup)
          (nifsCertificate setup).operational
          (frame setup))
      simpa only [Passes.NativeActivation.boundaryIds,
        occurrenceBoundary, List.nil_append] using
          idsDisjoint_append_swap visibleDisjoint)

/-- Complete native Step program for the selected benchmark deployment. -/
noncomputable def program (setup : Setup) : NativeCcsProgram.Program :=
  ConcreteNifsNativeCcsStep.program
    (application setup)
    (nifsCertificate setup)
    (stepRecipe setup)
    (defaultAdmissible setup)

/-- Proof-free native manifest used by later code generation. -/
noncomputable def manifest (setup : Setup) :
    NativeCcsManifest.Program :=
  (WasmBenchmark42x6.nativeManifest setup).stepProgram

/-- The manifest boundary conservatively protects every native allocation.
The explicit result list also records its output role. -/
noncomputable def programBoundary (setup : Setup) : Boundary.Columns :=
  Boundary.ofOwnedColumns
    (program setup).allocations
    ((WasmBenchmark42x6.nativeManifest setup).stepResultColumns.map
      fun column => column.id)
    ((program setup).allocations.map fun column => column.id)

noncomputable def observe (setup : Setup) :=
  Boundary.values (programBoundary setup)

/-- Native program serialization is an exact degree-three replacement. -/
noncomputable def manifestReplacement (setup : Setup) :
    Replacement
      (Goldilocks.Optimization.NativeCcs.system
        (program setup) (observe setup))
      (Goldilocks.Optimization.NativeCcs.manifestSystem
        (manifest setup) (observe setup))
      3 := by
  change
    Replacement
      (Goldilocks.Optimization.NativeCcs.system
        (program setup) (observe setup))
      (Goldilocks.Optimization.NativeCcs.manifestSystem
        (NativeCcsManifest.Program.ofProgram (program setup))
        (observe setup))
      3
  exact
    Goldilocks.Optimization.NativeCcs.ofProgramReplacement
      (program setup) (observe setup) 3 (by
        change 3 <= 3
        exact Nat.le_refl 3)

private theorem cost_ext
    {left right : Cost}
    (rows : left.recurringRows = right.recurringRows)
    (committed : left.committedColumns = right.committedColumns)
    (publicEq : left.publicColumns = right.publicColumns)
    (auxiliary : left.auxiliaryColumns = right.auxiliaryColumns) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem cost_add_right_cancel
    {left right extra : Cost}
    (equal : left + extra = right + extra) :
    left = right := by
  apply cost_ext
  · exact Nat.add_right_cancel (congrArg Cost.recurringRows equal)
  · exact Nat.add_right_cancel (congrArg Cost.committedColumns equal)
  · exact Nat.add_right_cancel (congrArg Cost.publicColumns equal)
  · exact Nat.add_right_cancel (congrArg Cost.auxiliaryColumns equal)

theorem intrinsicCost_exact
    (setup : Setup)
    (polynomialExact :
      setup.system.constraintPolynomial = Semantics.polynomial) :
    ConcreteNifsRawProgram.cost
        (profile setup)
        (nifsCertificate setup).operational
        (frame setup) =
      ⟨9_886_806, 0, 0, 9_833_395⟩ := by
  have static :
      ConcreteNifsRawProgram.cost
          (deployment setup).application.phase4.profile
          (ConcreteNifsCanonicalOperationalProfile.operational
            setup
            (defaultRunning CurrentM4Cost.dimensions
              CurrentM4Cost.commitmentRows)
            (machine benchmarkHashPlan)
            (terminalRelations
              CurrentM4Cost.dimensions CurrentM4Cost.commitmentRows)
            (terminalChecks
              CurrentM4Cost.dimensions CurrentM4Cost.commitmentRows)
            (widths setup)
            (selectedFootprints setup)
            (deployment setup).application)
          (frame setup) =
        ⟨9_886_806, 0, 0, 9_833_395⟩ := by
    calc
      ConcreteNifsRawProgram.cost
            (deployment setup).application.phase4.profile
            (ConcreteNifsCanonicalOperationalProfile.operational
              setup
              (defaultRunning CurrentM4Cost.dimensions
                CurrentM4Cost.commitmentRows)
              (machine benchmarkHashPlan)
              (terminalRelations
                CurrentM4Cost.dimensions CurrentM4Cost.commitmentRows)
              (terminalChecks
                CurrentM4Cost.dimensions CurrentM4Cost.commitmentRows)
              (widths setup)
              (selectedFootprints setup)
              (deployment setup).application)
            (frame setup) =
        ConcreteNifsStaticFootprint.intrinsicCost
          CurrentM4Cost.shape setup.system.constraintPolynomial
          publicRingColumns CurrentM4Cost.commitmentRows
          CurrentM4Cost.publicFits :=
        ConcreteNifsStaticFootprint.intrinsicCost_eq
          setup
          (defaultRunning CurrentM4Cost.dimensions
            CurrentM4Cost.commitmentRows)
          (machine benchmarkHashPlan)
          (terminalRelations
            CurrentM4Cost.dimensions CurrentM4Cost.commitmentRows)
          (terminalChecks
            CurrentM4Cost.dimensions CurrentM4Cost.commitmentRows)
          (widths setup)
          (selectedFootprints setup)
          (deployment setup).application
          (frame setup)
    _ = ⟨9_886_806, 0, 0, 9_833_395⟩ := by
      rw [polynomialExact]
      exact CurrentM4FootprintAudit.intrinsicCost_exact
  change
    ConcreteNifsRawProgram.cost
        (deployment setup).application.phase4.profile
        (ConcreteNifsCanonicalOperationalProfile.operational
          setup
          (defaultRunning CurrentM4Cost.dimensions
            CurrentM4Cost.commitmentRows)
          (machine benchmarkHashPlan)
          (terminalRelations
            CurrentM4Cost.dimensions CurrentM4Cost.commitmentRows)
          (terminalChecks
            CurrentM4Cost.dimensions CurrentM4Cost.commitmentRows)
          (widths setup)
          (selectedFootprints setup)
          (deployment setup).application)
        (frame setup) =
      ⟨9_886_806, 0, 0, 9_833_395⟩
  exact static

theorem overhead_exact
    (setup : Setup)
    (polynomialExact :
      setup.system.constraintPolynomial = Semantics.polynomial) :
    ConcreteNifsNativeCcsStepCost.overhead
        (application setup)
        (nifsCertificate setup)
        (stepRecipe setup)
        (defaultAdmissible setup) =
      ⟨9_886_806, 0, 0, 9_886_806⟩ := by
  unfold ConcreteNifsNativeCcsStepCost.overhead
  change
    ActivatedRawProgram.overheadCost
        (ConcreteNifsRawProgram.cost
          (profile setup)
          (nifsCertificate setup).operational
          (frame setup)).recurringRows =
      ⟨9_886_806, 0, 0, 9_886_806⟩
  rw [intrinsicCost_exact setup polynomialExact]
  exact CurrentM4FootprintAudit.activationOverhead_exact

/-- Exact optimized Step cost. The only reduction is the removed activation
overhead. Committed and public widths are unchanged. -/
theorem program_cost_exact
    (setup : Setup)
    (polynomialExact :
      setup.system.constraintPolynomial = Semantics.polynomial) :
    (program setup).cost =
      ⟨9_972_756, 244_058, 6, 9_838_443⟩ := by
  have sourceSplit :=
    ConcreteNifsNativeCcsStepCost.sourceCost_eq_nativeCost_add_overhead
      (application setup)
      (nifsCertificate setup)
      (stepRecipe setup)
      (defaultAdmissible setup)
  have sourceExact :=
    CurrentM4Cost.stepCost_exact setup polynomialExact
  have overheadExact := overhead_exact setup polynomialExact
  change
    (CurrentM4Cost.certificate setup).stepCost =
      (program setup).cost +
        ConcreteNifsNativeCcsStepCost.overhead
          (application setup)
          (nifsCertificate setup)
          (stepRecipe setup)
          (defaultAdmissible setup) at sourceSplit
  rw [sourceExact, overheadExact] at sourceSplit
  apply cost_add_right_cancel
  exact sourceSplit.symm.trans (by rfl)

/-- Exact physical row count of the optimized benchmark Step program. -/
theorem program_rows_exact
    (setup : Setup)
    (polynomialExact :
      setup.system.constraintPolynomial = Semantics.polynomial) :
    (program setup).rows.length = 9_972_756 := by
  rw [NativeCcsProgram.Program.rows_length,
    program_cost_exact setup polynomialExact]

/-- Exact committed allocation count, derived from allocation ownership. -/
theorem program_committedColumns_exact
    (setup : Setup)
    (polynomialExact :
      setup.system.constraintPolynomial = Semantics.polynomial) :
    NativeCounts.ownershipCount .committedColumn
        (program setup).allocations =
      244_058 := by
  rw [← NativeCounts.program_committedColumns,
    program_cost_exact setup polynomialExact]

/-- Exact public allocation count, derived from allocation ownership. -/
theorem program_publicColumns_exact
    (setup : Setup)
    (polynomialExact :
      setup.system.constraintPolynomial = Semantics.polynomial) :
    NativeCounts.ownershipCount .publicColumn
        (program setup).allocations =
      6 := by
  rw [← NativeCounts.program_publicColumns,
    program_cost_exact setup polynomialExact]

/-- Exact auxiliary allocation count, derived from allocation ownership. -/
theorem program_auxiliaryColumns_exact
    (setup : Setup)
    (polynomialExact :
      setup.system.constraintPolynomial = Semantics.polynomial) :
    NativeCounts.ownershipCount .auxiliaryColumn
        (program setup).allocations =
      9_838_443 := by
  rw [← NativeCounts.program_auxiliaryColumns,
    program_cost_exact setup polynomialExact]

/-- Exact total physical allocation count across all three roles. -/
theorem program_columns_exact
    (setup : Setup)
    (polynomialExact :
      setup.system.constraintPolynomial = Semantics.polynomial) :
    (program setup).allocations.length = 10_082_507 := by
  rw [NativeCounts.program_allocations_length,
    program_cost_exact setup polynomialExact]
  rfl

theorem manifest_cost_exact
    (setup : Setup)
    (polynomialExact :
      setup.system.constraintPolynomial = Semantics.polynomial) :
    (manifest setup).cost =
      ⟨9_972_756, 244_058, 6, 9_838_443⟩ := by
  change
    (NativeCcsManifest.Program.ofProgram (program setup)).cost =
      ⟨9_972_756, 244_058, 6, 9_838_443⟩
  rw [Goldilocks.Optimization.NativeCcs.manifest_cost_exact]
  exact program_cost_exact setup polynomialExact

theorem manifest_rows_exact
    (setup : Setup)
    (polynomialExact :
      setup.system.constraintPolynomial = Semantics.polynomial) :
    (manifest setup).rows.length = 9_972_756 := by
  rw [← NativeCcsManifest.Program.cost_recurringRows]
  exact congrArg Cost.recurringRows
    (manifest_cost_exact setup polynomialExact)

/-- Serialization retains the exact role-tagged allocation stream. -/
theorem manifest_columns_eq_program
    (setup : Setup) :
    (manifest setup).columns = (program setup).allocations := by
  change
    (NativeCcsManifest.Program.ofProgram (program setup)).columns =
      (program setup).allocations
  exact NativeCcsManifest.Program.columns_ofProgram _

/-- Serialization retains the exact total physical allocation count. -/
theorem manifest_columns_exact
    (setup : Setup)
    (polynomialExact :
      setup.system.constraintPolynomial = Semantics.polynomial) :
    (manifest setup).columns.length = 10_082_507 := by
  rw [manifest_columns_eq_program,
    program_columns_exact setup polynomialExact]

/-- The native replacement removes exactly one residual row and one residual
auxiliary column for each intrinsic NIFS row. -/
theorem removed_activation_overhead_exact
    (setup : Setup)
    (polynomialExact :
      setup.system.constraintPolynomial = Semantics.polynomial) :
    (program setup).cost +
        ⟨9_886_806, 0, 0, 9_886_806⟩ =
      ⟨19_859_562, 244_058, 6, 19_725_249⟩ := by
  rw [program_cost_exact setup polynomialExact]
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Optimization.Native
