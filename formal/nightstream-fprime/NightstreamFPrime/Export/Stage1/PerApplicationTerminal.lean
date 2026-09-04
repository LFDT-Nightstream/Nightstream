import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
import NightstreamFPrime.Lifecycle.Stage1.Terminal

/-!
Owns the outer terminal verifier for one verifier-selected application
package. The relation, application, Ajtai key, and verifier-context digest are
the same values used by the recursive step. The full verification-key digest
remains a separate package binding.

The terminal verifier adds no matrix row. It checks all running CE openings
and the fresh CCS opening against the self-derived relation.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationTerminal

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper

abbrev Program := Lifecycle.Stage1.Application.Program

abbrev FitsTwoPow28 (application : Program) :=
  PerApplicationFixedPoint.FitsTwoPow28 application

abbrev CommitmentSetup (application : Program) :=
  PerApplicationCanonicalPackage.CommitmentSetup application

abbrev ProofEnvelope (application : Program) :=
  Lifecycle.Stage1.Terminal.ProofEnvelope
    (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
    (publicFits := PerApplicationFixedPoint.publicFits application)

noncomputable def Holds
    (application : Program) (fits : FitsTwoPow28 application)
    (commitmentSetup : CommitmentSetup application)
    (statement : TerminalStatement AppState)
    (proof : ProofEnvelope application) : Prop :=
  Lifecycle.Stage1.Terminal.HoldsFor
    (PerApplicationFixedPoint.relation application fits)
    (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
    (PerApplicationCanonicalPackage.verifierContextDigest fits
      commitmentSetup)
    application statement proof

theorem holds_bottom_iff
    (application : Program) (fits : FitsTwoPow28 application)
    (commitmentSetup : CommitmentSetup application)
    (statement : TerminalStatement AppState) :
    Holds application fits commitmentSetup statement .bottom ↔
      statement.iteration = 0 ∧ statement.zi = statement.z0 := by
  exact Lifecycle.Stage1.Terminal.holdsFor_bottom_iff
    (PerApplicationFixedPoint.relation application fits)
    (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
    (PerApplicationCanonicalPackage.verifierContextDigest fits
      commitmentSetup)
    application statement

theorem holds_recursive_iff
    (application : Program) (fits : FitsTwoPow28 application)
    (commitmentSetup : CommitmentSetup application)
    (statement : TerminalStatement AppState)
    (payload : TerminalProof
      (Running
        (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Lifecycle.Stage1.Terminal.RunningWitness
        (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Fresh
        (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Lifecycle.Stage1.Terminal.FreshWitness
        (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      slotCount) :
    Holds application fits commitmentSetup statement (.recursive payload) ↔
      RecursiveTerminalTransition
        (setup (PerApplicationFixedPoint.relation application fits)
          (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
          (PerApplicationCanonicalPackage.verifierContextDigest fits
            commitmentSetup))
        (machineFor (PerApplicationFixedPoint.publicFits application)
          application)
        (Lifecycle.Stage1.Terminal.relations
          (PerApplicationFixedPoint.relation application fits)
          (PerApplicationCanonicalPackage.commitmentKey commitmentSetup))
        statement payload := by
  exact Lifecycle.Stage1.Terminal.holdsFor_recursive_iff
    (PerApplicationFixedPoint.relation application fits)
    (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
    (PerApplicationCanonicalPackage.verifierContextDigest fits
      commitmentSetup)
    application statement payload

theorem relations_iff_terminalHolds
    (application : Program) (fits : FitsTwoPow28 application)
    (commitmentSetup : CommitmentSetup application)
    (running : Running
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (runningWitness : Lifecycle.Stage1.Terminal.RunningWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (fresh : Fresh
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (freshWitness : Lifecycle.Stage1.Terminal.FreshWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application)) :
    (Lifecycle.Stage1.Terminal.relations
        (PerApplicationFixedPoint.relation application fits)
        (PerApplicationCanonicalPackage.commitmentKey
          commitmentSetup)).runningHolds
          functionIndex
          (PerApplicationCanonicalPackage.verifierContextDigest fits
            commitmentSetup) running runningWitness ∧
      (Lifecycle.Stage1.Terminal.relations
        (PerApplicationFixedPoint.relation application fits)
        (PerApplicationCanonicalPackage.commitmentKey
          commitmentSetup)).freshHolds
          functionIndex
          (PerApplicationCanonicalPackage.verifierContextDigest fits
            commitmentSetup) fresh freshWitness ↔
      Lifecycle.TerminalHolds
        (PerApplicationFixedPoint.relation application fits)
        (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) running
        runningWitness fresh freshWitness := by
  exact Lifecycle.Stage1.Terminal.relations_iff_terminalHolds
    (PerApplicationFixedPoint.relation application fits)
    (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
    (PerApplicationCanonicalPackage.verifierContextDigest fits
      commitmentSetup)
    running runningWitness fresh freshWitness

end NightstreamFPrime.Export.Stage1.PerApplicationTerminal
