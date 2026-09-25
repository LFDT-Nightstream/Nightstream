import NightstreamFPrime.Layout.Stage1.PiDECProtocolCompleteness

/-!
Owns one environment for the existing pilot and local C/R/D rows. Pilot
witnesses are constructed first. NIFS consumes the same external source
values and preserves those witnesses through the existing phase intervals.
Application, running-transition, next-preimage, and physical lowering remain
with their own constructors.
-/

namespace NightstreamFPrime.Layout.Stage1.PilotNifsCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (prior advertised : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
  (digest : Digest)
  (priorFixed : PilotProduction.FixedPreimage prior)
  (advertisedFixed : PilotProduction.FixedPreimage advertised)
  (digestFixed : digest.length = PilotProduction.digestWords)
  (values : PiCCSProofInputs.ProofValues) (context : VerifierContext.Digest4)

private theorem output_spec
    (interface : OutputHash.Interface) (offset : Nat) (env : Env)
    (preimage : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (result : Digest)
    (preimageRead : OutputHash.RepresentsPreimage interface offset env preimage)
    (digestRead : OutputHash.RepresentsDigest interface offset env result)
    (hashed : result = stateHash preimage) : OutputHash.SpecHolds interface offset env := by
  change List.ofFn (fun lane => (interface.digest offset lane).eval env) =
    Poseidon2.hash (Gadgets.Poseidon2.Hash.evalList env (interface.preimage offset))
  rw [preimageRead]
  exact digestRead.symm.trans hashed

private theorem prior_scope (interface : Lifecycle.Pilot.Interface) (offset : Nat) (env : Env)
    (assumptions : Lifecycle.Pilot.Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (Lifecycle.Pilot.priorCircuit interface).main offset),
      expression.VarsBelow (offset +
        localLength (Circuit.ops (Lifecycle.Pilot.priorCircuit interface).main offset)) :=
  PriorStateHash.flatConstraints_varsBelow interface.prior offset assumptions.1

private theorem output_scope (interface : Lifecycle.Pilot.Interface) (offset : Nat) (env : Env)
    (assumptions : Lifecycle.Pilot.Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (Lifecycle.Pilot.outputCircuit interface).main
        (Pilot.outputOffset interface offset)),
      expression.VarsBelow (Pilot.outputOffset interface offset + localLength
        (Circuit.ops (Lifecycle.Pilot.outputCircuit interface).main (Pilot.outputOffset interface offset))) :=
  OutputHash.flatConstraints_varsBelow interface.output (Pilot.outputOffset interface offset) assumptions.2

private theorem pilot_spec
    (env : Env)
    (agrees : ∀ index, index < PilotProduction.witnessOffset → env index =
      PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior)) advertised digest
        priorFixed advertisedFixed digestFixed values context index)
    (outputHash : digest = stateHash advertised) :
    Lifecycle.Pilot.SpecHolds PilotProduction.interface PilotProduction.witnessOffset env := by
  have represented := PilotProduction.protocolEnv_represents_of_agreesBelow prior
    (encHash (stateHash prior)) advertised digest priorFixed advertisedFixed digestFixed env
    (fun index below => (agrees index below).trans
      (PiCCSProtocolCompleteness.pilot_word prior (encHash (stateHash prior)) advertised digest
        priorFixed advertisedFixed digestFixed values context index below))
  constructor
  · change (fun column => (PilotProduction.priorInterface.publicInput
        PilotProduction.witnessOffset column).eval env) =
      PriorStateHash.encodedHash (Poseidon2.hash (Gadgets.Poseidon2.Hash.evalList env
        (PilotProduction.priorInterface.preimage PilotProduction.witnessOffset)))
    rw [represented.1]
    funext column
    exact represented.2.1 column
  · exact output_spec PilotProduction.outputInterface
      (Pilot.outputOffset PilotProduction.interface PilotProduction.witnessOffset) env advertised digest
      represented.2.2.1 represented.2.2.2 outputHash

private theorem pilot_constraints (interface : Lifecycle.Pilot.Interface) (offset : Nat) :
    flatConstraints [
      Sequence.childOp "stage1.prior_state_hash" (Lifecycle.Pilot.priorCircuit interface) offset,
      Sequence.childOp "stage1.output_hash" (Lifecycle.Pilot.outputCircuit interface)
        (Pilot.outputOffset interface offset)] = Pilot.logicalConstraints interface offset := by
  change flatConstraints (
    [Sequence.childOp "stage1.prior_state_hash" (Lifecycle.Pilot.priorCircuit interface) offset] ++
    [Sequence.childOp "stage1.output_hash" (Lifecycle.Pilot.outputCircuit interface)
      (Pilot.outputOffset interface offset)]) = _
  rw [flatConstraints_append, flatConstraints_singleton, flatConstraints_singleton]
  rfl

/-- Construct both pilot hash children from the canonical protocol source and
the actual output hash. The returned prefix carries its rows and exact scope. -/
theorem pilot_prefix
    (outputHash : digest = stateHash advertised) :
    ∃ p : Sequence.Prefix
        (PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior)) advertised digest
          priorFixed advertisedFixed digestFixed values context) PilotProduction.witnessOffset,
      PilotProduction.witnessOffset + localLength p.operations =
        Pilot.logicalColumnCount PilotProduction.interface PilotProduction.witnessOffset ∧
      flatConstraints p.operations =
        Pilot.logicalConstraints PilotProduction.interface PilotProduction.witnessOffset := by
  let initial := PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior)) advertised digest
    priorFixed advertisedFixed digestFixed values context
  let p0 := Sequence.empty initial PilotProduction.witnessOffset
  have specification := pilot_spec prior advertised digest priorFixed advertisedFixed digestFixed
    values context initial (fun _ _ => rfl) outputHash
  have assumptions := PilotProduction.assumptions initial
  obtain ⟨p1, p1Operations, p1End, _, _⟩ := Sequence.appendAt p0 "stage1.prior_state_hash"
    (Lifecycle.Pilot.priorCircuit PilotProduction.interface) PilotProduction.witnessOffset rfl
    (prior_scope PilotProduction.interface PilotProduction.witnessOffset initial assumptions)
    assumptions.1 specification.1
  have outputStart : PilotProduction.witnessOffset + localLength p1.operations =
      Pilot.outputOffset PilotProduction.interface PilotProduction.witnessOffset :=
    p1End.trans (Pilot.outputOffset_eq_add PilotProduction.interface PilotProduction.witnessOffset).symm
  have outputSpec := pilot_spec prior advertised digest priorFixed advertisedFixed digestFixed
    values context p1.current (fun index below => p1.agrees index (Or.inl below)) outputHash
  have outputAssumptions := PilotProduction.assumptions p1.current
  obtain ⟨p2, p2Operations, p2End, _, _⟩ := Sequence.appendAt p1 "stage1.output_hash"
    (Lifecycle.Pilot.outputCircuit PilotProduction.interface)
    (Pilot.outputOffset PilotProduction.interface PilotProduction.witnessOffset) outputStart
    (output_scope PilotProduction.interface PilotProduction.witnessOffset p1.current outputAssumptions)
    outputAssumptions.2 outputSpec.2
  refine ⟨p2, p2End.trans
    (Pilot.logicalColumnCount_eq_add PilotProduction.interface PilotProduction.witnessOffset).symm, ?_⟩
  rw [p2Operations, p1Operations]
  simpa only [p0, Sequence.empty, List.nil_append, List.singleton_append] using
    pilot_constraints PilotProduction.interface PilotProduction.witnessOffset

private theorem pilot_end_before_context :
    Pilot.logicalColumnCount PilotProduction.interface PilotProduction.witnessOffset ≤
      PiCCSInputs.expectedContextStart := by
  rw [PiCCSInputs.expectedContextStart_matches_pilot, Pilot.physicalColumnCount_eq]
  omega

private theorem pilot_end_before_c :
    Pilot.logicalColumnCount PilotProduction.interface PilotProduction.witnessOffset ≤
      PiCCSInputs.phaseOffset := by
  have bound := pilot_end_before_context
  unfold PiCCSInputs.phaseOffset PiCCSInputs.proofInputStart
  omega

private theorem c_before_r : PiCCSInputs.phaseOffset ≤ PiRLCInputs.phaseOffset := by
  have bound := PiRLCInputs.piCcsLogicalFreshBase_le_phaseOffset
  unfold PiCCSStarts.logicalFreshBase at bound
  exact Nat.le_trans (Nat.le_add_right _ _) bound

private theorem external_outside_pilot (index : Nat)
    (external : PiCCSOrdinarySourceSupport.External index) :
    index < PilotProduction.witnessOffset ∨
      Pilot.logicalColumnCount PilotProduction.interface PilotProduction.witnessOffset ≤ index := by
  rcases external with priorRange | publicRange | outputRange | contextRange | proofRange
  · apply Or.inl
    rcases priorRange with ⟨_, upper⟩
    unfold PilotProduction.witnessOffset PilotProduction.externalColumnCount PilotProduction.outputDigestStart
      PilotProduction.outputPreimageStart PilotProduction.priorPublicInputStart
    omega
  · apply Or.inl
    rcases publicRange with ⟨_, upper⟩
    change index < PilotProduction.priorPublicInputStart + PriorStateHash.publicWidth at upper
    unfold PilotProduction.witnessOffset PilotProduction.externalColumnCount PilotProduction.outputDigestStart
      PilotProduction.outputPreimageStart
    omega
  · apply Or.inl
    rcases outputRange with ⟨_, upper⟩
    unfold PilotProduction.witnessOffset PilotProduction.externalColumnCount PilotProduction.outputDigestStart
    omega
  · exact Or.inr (Nat.le_trans pilot_end_before_context contextRange.1)
  · apply Or.inr
    have lower := proofRange.1
    unfold PiCCSInputs.proofInputStart at lower
    exact Nat.le_trans pilot_end_before_context (by omega)

variable
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (template : Proof 9)

/-- Actual hash and NIFS acceptance construct the existing pilot and C/R/D
rows in one final environment. The initial source values and accepted NIFS
output remain exact. No generated pilot/NIFS phase, row, or output value is
assumed. -/
theorem completePrefix
    (result : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPc : prior.pc = 1) (advertisedPc : advertised.pc = 1)
    (priorContext : prior.verifierKeys functionIndex = context.toList)
    (advertisedContext : advertised.verifierKeys functionIndex = context.toList)
    (outputHash : digest = stateHash advertised)
    (accepted : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
      (prior.running functionIndex)
      (PiCCSProofInputs.protocolFresh logicalWidth publicFits (encHash (stateHash prior)) values)
      (PiCCSProofInputs.relationProof relation values template) = some result) :
    ∃ p : Sequence.Prefix
        (PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior)) advertised digest
          priorFixed advertisedFixed digestFixed values context) PilotProduction.witnessOffset,
      ∃ c : Sequence.Prefix p.current PiCCSInputs.phaseOffset,
        ∃ r : Sequence.Prefix c.current PiRLCInputs.phaseOffset,
          ∃ d : Sequence.Prefix
              (PiDECProofInputs.load r.current (PiCCSProofInputs.relationProof relation values template)
                (PiRLC.v1_1.Semantics.evalOutput relation
                  (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
                  PiRLCInputs.phaseOffset r.current).publicInput) PiDECInputs.phaseOffset,
            PilotProduction.witnessOffset + localLength p.operations =
              Pilot.logicalColumnCount PilotProduction.interface PilotProduction.witnessOffset ∧
            flatConstraints p.operations =
              Pilot.logicalConstraints PilotProduction.interface PilotProduction.witnessOffset ∧
            c.operations = PiCCS.v1_1.Formal.opsAt relation (PiCCSProofInputs.relationInterface relation)
              PiCCSInputs.phaseOffset ∧
            r.operations = PiRLC.v1_1.Formal.opsAt relation
              (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
              PiRLCInputs.phaseOffset ∧
            d.operations = PiDEC.v1_1.Formal.opsAt relation (PiDECInputs.interface logicalWidth publicFits)
              PiDECInputs.phaseOffset ∧
            holdsFlat d.current p.operations ∧ holdsFlat d.current c.operations ∧
            holdsFlat d.current r.operations ∧
            Lifecycle.Pilot.SpecHolds PilotProduction.interface PilotProduction.witnessOffset d.current ∧
            PiDEC.v1_1.Semantics.PhaseHolds relation ajtai (PiDECInputs.interface logicalWidth publicFits)
              PiDECInputs.phaseOffset d.current ∧
            RunningTransitionInputs.piDecRunningOutput relation d.current = result := by
  obtain ⟨p, pilotEnd, pilotConstraints⟩ := pilot_prefix prior advertised digest priorFixed advertisedFixed
    digestFixed values context outputHash
  have source : ∀ index, PiCCSOrdinarySourceSupport.External index → p.current index =
      PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior)) advertised digest
        priorFixed advertisedFixed digestFixed values context index := by
    intro index external
    apply p.agrees
    rcases external_outside_pilot index external with below | after
    · exact Or.inl below
    · exact Or.inr (by rw [pilotEnd]; exact after)
  obtain ⟨c, r, d, cOperations, rOperations, dOperations, cRows, rRows, dPhase, dOutput⟩ :=
    PiDECProtocolCompleteness.completePrefix_from relation ajtai prior (encHash (stateHash prior))
      advertised digest priorFixed advertisedFixed digestFixed values context template result
      priorPc advertisedPc priorContext advertisedContext accepted p.current source
  have preserved : ∀ index, index < Pilot.logicalColumnCount PilotProduction.interface
      PilotProduction.witnessOffset → d.current index = p.current index := by
    intro index below
    have beforeC := Nat.lt_of_lt_of_le below pilot_end_before_c
    have beforeR := Nat.lt_of_lt_of_le beforeC c_before_r
    have beforeDInputs := Nat.lt_of_lt_of_le beforeR
      (Nat.le_trans (Nat.le_add_right _ _) PiDECProtocolCompleteness.rEnd_before_dInputs)
    have beforeD : index < PiDECInputs.phaseOffset :=
      Nat.lt_of_lt_of_le beforeDInputs (Nat.le_add_right _ _)
    exact (d.agrees index (Or.inl beforeD)).trans
      ((PiDECProofInputs.load_agreesOutside _ _ _ index (Or.inl beforeDInputs)).trans
        ((r.agrees index (Or.inl beforeR)).trans (c.agrees index (Or.inl beforeC))))
  have pilotRows : holdsFlat d.current p.operations := by
    intro expression member
    have scope := p.scope expression member
    rw [pilotEnd] at scope
    exact (expression.eval_eq_of_agree_below _ d.current p.current scope preserved).trans
      (p.rows expression member)
  have selectedPilotRows : ∀ expression ∈ Pilot.logicalConstraints PilotProduction.interface
      PilotProduction.witnessOffset, expression.eval d.current = 0 := by
    rw [← pilotConstraints]
    exact pilotRows
  have priorRows : holdsFlat d.current (Circuit.ops
      (Lifecycle.Pilot.priorCircuit PilotProduction.interface).main PilotProduction.witnessOffset) := by
    intro expression member
    exact selectedPilotRows expression (List.mem_append_left _ member)
  have outputRows : holdsFlat d.current (Circuit.ops
      (Lifecycle.Pilot.outputCircuit PilotProduction.interface).main
      (Pilot.outputOffset PilotProduction.interface PilotProduction.witnessOffset)) := by
    intro expression member
    exact selectedPilotRows expression (List.mem_append_right _ member)
  have pilotPhase := Lifecycle.Pilot.phase_soundness PilotProduction.interface PilotProduction.witnessOffset
    d.current (PilotProduction.assumptions d.current)
    (holdsFlat_implies_holds _ _ priorRows) (holdsFlat_implies_holds _ _ outputRows)
  exact ⟨p, c, r, d, pilotEnd, pilotConstraints, cOperations, rOperations, dOperations,
    pilotRows, cRows, rRows, pilotPhase, dPhase, dOutput⟩

end NightstreamFPrime.Layout.Stage1.PilotNifsCompleteness
