/-!
Paper-owned semantic contract for HyperNova Definition 12.

Source: HyperNova Section 6.2 and Appendix H.2.

Owns: NP-complete full encoding, independent structure/instance encoders,
left-inverse decoding, structure compatibility, monotonicity, and one universal
default instance-witness pair.

Does not own: a concrete circuit compiler, an asymptotic cost proof,
Construction 2, SuperNeo, Rust, R1CS, or constraints.

Emits constraints: no.

Polynomial-time and succinctness judgments are a separate algorithmic
security boundary.  This module freezes the extensional equations those
algorithms must satisfy; it does not turn an arbitrary supplied encoder into a
proved compatible scheme.
-/

namespace Nightstream.HyperNova.NIVCCompatibility

universe uParameters uCircuit uInput uAdvice uOutput
  uRunningStructure uFreshStructure uInstance uWitness

/-- Independent function and relation semantics used by Definition 12. -/
structure Semantics
    (Parameters : Type uParameters)
    (Circuit : Type uCircuit)
    (Input : Type uInput)
    (Advice : Type uAdvice)
    (Output : Type uOutput)
    (RunningStructure : Type uRunningStructure)
    (FreshStructure : Type uFreshStructure)
    (Instance : Type uInstance)
    (Witness : Type uWitness) where
  execute : Circuit -> Input -> Advice -> Output
  runningHolds : Parameters -> RunningStructure -> Instance -> Witness -> Prop
  freshUnderlyingHolds :
    Parameters -> FreshStructure -> Instance -> Witness -> Prop
  structuresCompatible : RunningStructure -> FreshStructure -> Prop
  circuitSize : Circuit -> Nat
  structureSize : FreshStructure -> Nat

/-- Deterministic full and partial encoders with explicit left inverses. -/
structure Encoding
    (Circuit : Type uCircuit)
    (Input : Type uInput)
    (Advice : Type uAdvice)
    (Output : Type uOutput)
    (RunningStructure : Type uRunningStructure)
    (FreshStructure : Type uFreshStructure)
    (Instance : Type uInstance)
    (Witness : Type uWitness) where
  encode : Circuit -> Input -> Advice -> Output ->
    FreshStructure × Instance × Witness
  encodeStructures : Circuit -> RunningStructure × FreshStructure
  encodeInstance : Input -> Output -> Instance
  decode : FreshStructure × Instance × Witness ->
    Option (Circuit × Input × Advice × Output)
  decodeStructures : RunningStructure × FreshStructure -> Option Circuit
  decodeInstance : Instance -> Option (Input × Output)

/-- NP-completeness and efficient invertibility, stated extensionally. -/
def NPComplete
    {Parameters : Type uParameters}
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Advice : Type uAdvice}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {Instance : Type uInstance}
    {Witness : Type uWitness}
    (semantics : Semantics Parameters Circuit Input Advice Output
      RunningStructure FreshStructure Instance Witness)
    (encoding : Encoding Circuit Input Advice Output
      RunningStructure FreshStructure Instance Witness) : Prop :=
  (forall parameters circuit input advice output,
    let encoded := encoding.encode circuit input advice output
    semantics.freshUnderlyingHolds parameters encoded.1 encoded.2.1
        encoded.2.2 <->
      semantics.execute circuit input advice = output) /\
  (forall circuit input advice output,
    encoding.decode (encoding.encode circuit input advice output) =
      some (circuit, input, advice, output))

/-- Structure and instance encodings are witness-independent and agree with
the corresponding projections of the full encoder. -/
def PartialFunctions
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Advice : Type uAdvice}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {Instance : Type uInstance}
    {Witness : Type uWitness}
    (encoding : Encoding Circuit Input Advice Output
      RunningStructure FreshStructure Instance Witness)
    (structuresCompatible : RunningStructure -> FreshStructure -> Prop) : Prop :=
  (forall circuit input advice output,
    let structures := encoding.encodeStructures circuit
    let encoded := encoding.encode circuit input advice output
    encoded.1 = structures.2 /\
      encoded.2.1 = encoding.encodeInstance input output /\
      structuresCompatible structures.1 structures.2) /\
  (forall circuit,
    encoding.decodeStructures (encoding.encodeStructures circuit) = some circuit) /\
  (forall input output,
    encoding.decodeInstance (encoding.encodeInstance input output) =
      some (input, output))

/-- Encoding larger circuits never yields a smaller relation structure. -/
def Monotone
    {Parameters : Type uParameters}
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Advice : Type uAdvice}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {Instance : Type uInstance}
    {Witness : Type uWitness}
    (semantics : Semantics Parameters Circuit Input Advice Output
      RunningStructure FreshStructure Instance Witness)
    (encoding : Encoding Circuit Input Advice Output
      RunningStructure FreshStructure Instance Witness) : Prop :=
  forall left right,
    semantics.circuitSize left <= semantics.circuitSize right ->
      semantics.structureSize (encoding.encodeStructures left).2 <=
        semantics.structureSize (encoding.encodeStructures right).2

/-- One paper default pair, valid for every public parameter and every running
structure. -/
structure UniversalDefault
    {Parameters : Type uParameters}
    {RunningStructure : Type uRunningStructure}
    {Instance : Type uInstance}
    {Witness : Type uWitness}
    (runningHolds : Parameters -> RunningStructure -> Instance -> Witness -> Prop)
    where
  defaultInstance : Instance
  defaultWitness : Witness
  valid : forall parameters relationStructure,
    runningHolds parameters relationStructure defaultInstance defaultWitness

/-- Exact extensional part of HyperNova NIVC compatibility. -/
def Holds
    {Parameters : Type uParameters}
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Advice : Type uAdvice}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {Instance : Type uInstance}
    {Witness : Type uWitness}
    (semantics : Semantics Parameters Circuit Input Advice Output
      RunningStructure FreshStructure Instance Witness)
    (encoding : Encoding Circuit Input Advice Output
      RunningStructure FreshStructure Instance Witness)
    (default : UniversalDefault semantics.runningHolds) : Prop :=
  NPComplete semantics encoding /\
  PartialFunctions encoding semantics.structuresCompatible /\
  Monotone semantics encoding /\
  (forall parameters relationStructure,
    semantics.runningHolds parameters relationStructure
      default.defaultInstance default.defaultWitness)

end Nightstream.HyperNova.NIVCCompatibility
