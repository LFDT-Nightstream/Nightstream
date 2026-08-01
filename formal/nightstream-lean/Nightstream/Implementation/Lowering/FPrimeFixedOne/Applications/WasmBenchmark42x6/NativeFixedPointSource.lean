import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointCost
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsCompiler
import Nightstream.Implementation.R1CS.Core.SeededAjtai

/-!
Contract: package the exact native benchmark Step program as a finite
four-matrix compiler input.

Assurance tier: model-level.

Owns: setup-independent shape facts, the non-authoritative zero-matrix seed,
the exact native Step program, compiler validity, and its Boolean row domain.

Does not own: compiler matrix transport, recursive fixed-point stability,
terminal R1CS lowering, Rust, or a security reduction.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointSource

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointCost
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector
open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- The sole setup-owned input that the emitted Step program cannot derive.
The key is computed from the seed by Lean. It is not an arbitrary function
paired with unrelated seed metadata. -/
structure Template where
  ajtai :
    SeededAjtai.Setup commitmentRows
      (Phi81ColumnLayout.blockCount
        (ConcreteNifsPlain270Profile.Shape dimensions).carrierWidth)

theorem domainCovers :
    PiCcsDomains.production.nc.Covers
      (ConcreteNifsPlain270Profile.Shape dimensions) := by
  constructor <;> decide

theorem rowNonempty :
    0 <
      (ConcreteNifsPlain270Profile.Shape dimensions).rowVariables := by
  decide

namespace Template

def verifierKey (template : Template) :
    VerifierKey
      (ConcreteNifsPlain270Profile.Shape dimensions)
      publicRingColumns
      (ConcreteNifsPlain270Profile.publicFits dimensions)
      commitmentRows :=
  template.ajtai.verifierKey

noncomputable def withSystem
    (template : Template)
    (system : Structure dimensions.shape) :
    RelationSetup dimensions commitmentRows where
  verifierKey := template.verifierKey
  system := system
  domainCovers := domainCovers
  rowNonempty := rowNonempty

end Template

def seedSystem : Structure dimensions.shape where
  matrices := fun _ _ _ => 0
  constraintPolynomial := NativeCcsSelector.constraintPolynomial

noncomputable def seedSetup (template : Template) :
    RelationSetup dimensions commitmentRows :=
  template.withSystem seedSystem

@[simp] theorem seedSetup_polynomial (template : Template) :
    (seedSetup template).system.constraintPolynomial =
      NativeCcsSelector.constraintPolynomial := by
  rfl

noncomputable def program (template : Template) : NativeCcsProgram.Program :=
  NativeFixedPointCost.nativeProgram (seedSetup template)

noncomputable def valid (template : Template) :
    NativeCcsCompiler.Valid (program template) :=
  ConcreteNifsNativeCcsCompiler.valid
    (deployment (seedSetup template)).application.phase4
    (NativeFixedPointCost.nifsCertificate (seedSetup template))
    (deployment (seedSetup template)).step
    (deployment (seedSetup template)).defaultRunningAdmissible

def rowDomain (template : Template) :
    NativeCcsCompiler.RowDomain (program template) where
  rowVariables := dimensions.rowVariables
  rowsCovered := by
    change
      (NativeFixedPointCost.nativeProgram
        (seedSetup template)).rows.length ≤ 2 ^ dimensions.rowVariables
    rw [NativeFixedPointCost.nativeRows_exact (seedSetup template) rfl]
    decide

theorem columnsExact (template : Template) :
    (program template).columnIds.length =
      dimensions.alignedLogicalWidth := by
  exact
    (NativeFixedPointCost.logicalWidth_fixed
      (seedSetup template) rfl).symm

theorem rowsExact (template : Template) :
    (program template).rows.length = 5_299_490 := by
  exact NativeFixedPointCost.nativeRows_exact (seedSetup template) rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointSource
