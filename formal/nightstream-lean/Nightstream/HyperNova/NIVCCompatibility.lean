/-!
Contract: HyperNova Definition 12 NIVC compatibility.

Source: corrected HyperNova Section 6.2, Construction 3, and Appendix H.2.

Owns: canonical full and partial compiler inverses, decoding of every
satisfying canonical tuple, common rectangular padding, parameter-owned
defaults, and the compact recursive verifier-key and statement-identifier
boundary.

Does not own: a concrete circuit compiler, polynomial-time proofs, a
Fiat--Shamir knowledge theorem, collision resistance, Construction 2,
SuperNeo, Rust, R1CS rows, or constraint counts.

Emits constraints: no.

This is an extensional model-level contract. In particular, identifier
equality is never used as authority. A disagreement with equal identifiers is
returned as an explicit collision event.
-/

namespace Nightstream.HyperNova.NIVCCompatibility

universe uValue uWord uParameters uCircuit uInput uAdvice uOutput
  uRunningStructure uFreshStructure uRunningInstance uRunningWitness
  uFreshInstance uFreshWitness uVerifierKey
  uVerifierProjection uStatementId uVerifierInput uVerifierOutput

/-! ## Canonical encodings -/

/-- No codeword is a strict prefix of another codeword. -/
def PrefixFree
    {Value : Type uValue}
    {Word : Type uWord}
    (encode : Value -> List Word) : Prop :=
  forall left right suffix,
    encode right = encode left ++ suffix ->
      left = right

/-- Data of one serializer and its partial canonical decoder. -/
structure Codec
    (Value : Type uValue)
    (Word : Type uWord) where
  encode : Value -> List Word
  decode : List Word -> Option Value

namespace Codec

/-- Both inverse directions and prefix freedom required of a canonical codec. -/
def Canonical
    {Value : Type uValue}
    {Word : Type uWord}
    (codec : Codec Value Word) : Prop :=
  (forall value, codec.decode (codec.encode value) = some value) /\
  (forall words value, codec.decode words = some value ->
    codec.encode value = words) /\
  PrefixFree codec.encode

/-- A canonical encoder is injective. -/
theorem encode_injective
    {Value : Type uValue}
    {Word : Type uWord}
    (codec : Codec Value Word)
    (canonical : codec.Canonical) :
    Function.Injective codec.encode := by
  intro left right sameEncoding
  have leftDecoded := canonical.1 left
  have rightDecoded := canonical.1 right
  rw [sameEncoding, rightDecoded] at leftDecoded
  exact Option.some.inj leftDecoded.symm

/-- Construct a partial decoder from an explicit encoding. -/
noncomputable def withClassicalDecoder
    {Value : Type uValue}
    {Word : Type uWord}
    (encode : Value -> List Word) :
    Codec Value Word where
  encode := encode
  decode := fun words =>
    letI : Decidable (exists value, encode value = words) :=
      Classical.propDecidable _
    if existsValue : exists value, encode value = words then
      some (Classical.choose existsValue)
    else
      none

/-- Fixed width plus injectivity gives both inverse directions and prefix
freedom for `withClassicalDecoder`. -/
theorem fixedWidthInjective_canonical
    {Value : Type uValue}
    {Word : Type uWord}
    (width : Nat)
    (encode : Value -> List Word)
    (encodeLength : forall value, (encode value).length = width)
    (encodeInjective : Function.Injective encode) :
    (withClassicalDecoder encode).Canonical := by
  classical
  refine ⟨?_, ?_, ?_⟩
  · intro value
    let existsValue : exists candidate, encode candidate = encode value :=
      ⟨value, rfl⟩
    change
      (if found : exists candidate, encode candidate = encode value then
          some (Classical.choose found)
        else none) = some value
    rw [dif_pos existsValue]
    apply congrArg some
    exact encodeInjective (Classical.choose_spec existsValue)
  · intro words value decoded
    by_cases existsValue : exists candidate, encode candidate = words
    · change
        (if found : exists candidate, encode candidate = words then
            some (Classical.choose found)
          else none) = some value at decoded
      rw [dif_pos existsValue] at decoded
      have chosenEq : Classical.choose existsValue = value :=
        Option.some.inj decoded
      rw [← chosenEq]
      exact Classical.choose_spec existsValue
    · change
        (if found : exists candidate, encode candidate = words then
            some (Classical.choose found)
          else none) = some value at decoded
      rw [dif_neg existsValue] at decoded
      contradiction
  · intro left right suffix isPrefix
    change encode right = encode left ++ suffix at isPrefix
    have lengthEquality := congrArg List.length isPrefix
    have adjusted : width + 0 = width + suffix.length := by
      simpa [encodeLength] using lengthEquality
    have suffixLength : suffix.length = 0 :=
      (Nat.add_left_cancel adjusted).symm
    have suffixEmpty : suffix = [] :=
      List.eq_nil_of_length_eq_zero suffixLength
    subst suffix
    simp only [List.append_nil] at isPrefix
    exact encodeInjective isPrefix.symm

end Codec

/-- Canonical serializers for every compiler object used by Definition 12.

The `Parameters` value owns the declared field, dimensions, capacity, version,
and layout. The product codecs include their type and length information; the
`Canonical` predicate below makes each resulting language prefix-free. -/
structure CanonicalLayouts
    (Word : Type uWord)
    (Parameters : Type uParameters)
    (Circuit : Type uCircuit)
    (Input : Type uInput)
    (Advice : Type uAdvice)
    (Output : Type uOutput)
    (RunningStructure : Type uRunningStructure)
    (FreshStructure : Type uFreshStructure)
    (RunningInstance : Type uRunningInstance)
    (RunningWitness : Type uRunningWitness)
    (FreshInstance : Type uFreshInstance)
    (FreshWitness : Type uFreshWitness) where
  parameters : Codec Parameters Word
  sourceTuple : Codec (Circuit × Input × Advice × Output) Word
  structures : Codec (RunningStructure × FreshStructure) Word
  inputOutput : Codec (Input × Output) Word
  runningInstance : Codec RunningInstance Word
  runningWitness : Codec RunningWitness Word
  freshInstance : Codec FreshInstance Word
  encodedTuple : Codec (FreshStructure × FreshInstance × FreshWitness) Word

namespace CanonicalLayouts

/-- Every compiler-facing serialization has canonical inverse equations and
is prefix-free. -/
def Holds
    {Word : Type uWord}
    {Parameters : Type uParameters}
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Advice : Type uAdvice}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {RunningInstance : Type uRunningInstance}
    {RunningWitness : Type uRunningWitness}
    {FreshInstance : Type uFreshInstance}
    {FreshWitness : Type uFreshWitness}
    (layouts : CanonicalLayouts Word Parameters Circuit Input Advice Output
      RunningStructure FreshStructure RunningInstance RunningWitness
      FreshInstance FreshWitness) : Prop :=
  layouts.parameters.Canonical /\
  layouts.sourceTuple.Canonical /\
  layouts.structures.Canonical /\
  layouts.inputOutput.Canonical /\
  layouts.runningInstance.Canonical /\
  layouts.runningWitness.Canonical /\
  layouts.freshInstance.Canonical /\
  layouts.encodedTuple.Canonical

end CanonicalLayouts

/-! ## Compiler semantics and inverse laws -/

/-- Independent function and relation semantics used by Definition 12. -/
structure Semantics
    (Parameters : Type uParameters)
    (Circuit : Type uCircuit)
    (Input : Type uInput)
    (Advice : Type uAdvice)
    (Output : Type uOutput)
    (RunningStructure : Type uRunningStructure)
    (FreshStructure : Type uFreshStructure)
    (RunningInstance : Type uRunningInstance)
    (RunningWitness : Type uRunningWitness)
    (FreshInstance : Type uFreshInstance)
    (FreshWitness : Type uFreshWitness) where
  execute : Circuit -> Input -> Advice -> Output
  runningHolds :
    Parameters -> RunningStructure -> RunningInstance -> RunningWitness -> Prop
  freshUnderlyingHolds :
    Parameters -> FreshStructure -> FreshInstance -> FreshWitness -> Prop
  runningStructureAdmissible : Parameters -> RunningStructure -> Prop
  structuresCompatible : RunningStructure -> FreshStructure -> Prop
  circuitSize : Circuit -> Nat
  structureSize : FreshStructure -> Nat
  structureRows : FreshStructure -> Nat
  structureColumns : FreshStructure -> Nat

/-- Deterministic full and partial compiler functions. Their inverse laws are
stated separately so data is not confused with proof authority. -/
structure Encoding
    (Circuit : Type uCircuit)
    (Input : Type uInput)
    (Advice : Type uAdvice)
    (Output : Type uOutput)
    (RunningStructure : Type uRunningStructure)
    (FreshStructure : Type uFreshStructure)
    (FreshInstance : Type uFreshInstance)
    (FreshWitness : Type uFreshWitness) where
  encode : Circuit -> Input -> Advice -> Output ->
    FreshStructure × FreshInstance × FreshWitness
  encodeStructures : Circuit -> RunningStructure × FreshStructure
  encodeInstance : Input -> Output -> FreshInstance
  decode : FreshStructure × FreshInstance × FreshWitness ->
    Option (Circuit × Input × Advice × Output)
  decodeStructures : RunningStructure × FreshStructure -> Option Circuit
  decodeInstance : FreshInstance -> Option (Input × Output)

/-- NP-completeness and the corrected witness-decoding requirements.

The final clause covers every satisfying tuple whose structure is in the
structure encoder's image. It is stronger than a round trip for honest
encodings: a satisfying canonical assignment must decode to the exact source
tuple and re-encode without change. -/
def NPComplete
    {Parameters : Type uParameters}
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Advice : Type uAdvice}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {RunningInstance : Type uRunningInstance}
    {RunningWitness : Type uRunningWitness}
    {FreshInstance : Type uFreshInstance}
    {FreshWitness : Type uFreshWitness}
    (semantics : Semantics Parameters Circuit Input Advice Output
      RunningStructure FreshStructure RunningInstance RunningWitness
      FreshInstance FreshWitness)
    (encoding : Encoding Circuit Input Advice Output
      RunningStructure FreshStructure FreshInstance FreshWitness) : Prop :=
  (forall parameters circuit input advice output,
    let encoded := encoding.encode circuit input advice output
    semantics.freshUnderlyingHolds parameters encoded.1 encoded.2.1
        encoded.2.2 <->
      semantics.execute circuit input advice = output) /\
  (forall circuit input advice output,
    encoding.decode (encoding.encode circuit input advice output) =
      some (circuit, input, advice, output)) /\
  (forall encoded circuit input advice output,
    encoding.decode encoded = some (circuit, input, advice, output) ->
      encoding.encode circuit input advice output = encoded) /\
  (forall parameters circuit freshStructure encodedInstance witness,
    freshStructure = (encoding.encodeStructures circuit).2 ->
    semantics.freshUnderlyingHolds parameters freshStructure encodedInstance witness ->
      exists input advice output,
        encoding.decode (freshStructure, encodedInstance, witness) =
            some (circuit, input, advice, output) /\
        encoding.encode circuit input advice output =
            (freshStructure, encodedInstance, witness) /\
        semantics.execute circuit input advice = output)

/-- Structure and instance encodings are witness-independent, compatible, and
canonical in both inverse directions. -/
def PartialFunctions
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Advice : Type uAdvice}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {FreshInstance : Type uFreshInstance}
    {FreshWitness : Type uFreshWitness}
    (encoding : Encoding Circuit Input Advice Output
      RunningStructure FreshStructure FreshInstance FreshWitness)
    (structuresCompatible : RunningStructure -> FreshStructure -> Prop) : Prop :=
  (forall circuit input advice output,
    let structures := encoding.encodeStructures circuit
    let encoded := encoding.encode circuit input advice output
    encoded.1 = structures.2 /\
      encoded.2.1 = encoding.encodeInstance input output /\
      structuresCompatible structures.1 structures.2) /\
  (forall circuit,
    encoding.decodeStructures (encoding.encodeStructures circuit) = some circuit) /\
  (forall structures circuit,
    encoding.decodeStructures structures = some circuit ->
      encoding.encodeStructures circuit = structures) /\
  (forall input output,
    encoding.decodeInstance (encoding.encodeInstance input output) =
      some (input, output)) /\
  (forall encodedInstance input output,
    encoding.decodeInstance encodedInstance = some (input, output) ->
      encoding.encodeInstance input output = encodedInstance)

/-- Encoding larger canonical circuits never yields a smaller padded
relation structure. -/
def Monotone
    {Parameters : Type uParameters}
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Advice : Type uAdvice}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {RunningInstance : Type uRunningInstance}
    {RunningWitness : Type uRunningWitness}
    {FreshInstance : Type uFreshInstance}
    {FreshWitness : Type uFreshWitness}
    (semantics : Semantics Parameters Circuit Input Advice Output
      RunningStructure FreshStructure RunningInstance RunningWitness
      FreshInstance FreshWitness)
    (encoding : Encoding Circuit Input Advice Output
      RunningStructure FreshStructure FreshInstance FreshWitness) : Prop :=
  forall left right,
    semantics.circuitSize left <= semantics.circuitSize right ->
      semantics.structureSize (encoding.encodeStructures left).2 <=
        semantics.structureSize (encoding.encodeStructures right).2

/-- Public-parameter-owned rectangular capacities and canonical zero-padding
domain. The row capacity also contains every assignment column, so a protocol
can prepend the padded identity matrix and use one row-domain SumCheck. -/
structure CompilerLayout
    {Parameters : Type uParameters}
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Advice : Type uAdvice}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {RunningInstance : Type uRunningInstance}
    {RunningWitness : Type uRunningWitness}
    {FreshInstance : Type uFreshInstance}
    {FreshWitness : Type uFreshWitness}
    (semantics : Semantics Parameters Circuit Input Advice Output
      RunningStructure FreshStructure RunningInstance RunningWitness
      FreshInstance FreshWitness)
    (encoding : Encoding Circuit Input Advice Output
      RunningStructure FreshStructure FreshInstance FreshWitness) where
  Fits : Parameters -> Circuit -> Prop
  rowCapacity : Parameters -> Nat
  columnCapacity : Parameters -> Nat
  columnsFitRows : forall parameters,
    columnCapacity parameters <= rowCapacity parameters
  paddedCanonical : Parameters -> FreshStructure -> Prop

namespace CompilerLayout

/-- Every circuit admitted by one public parameter string has the same
rectangular row and column capacities and a canonical padded representation. -/
def Holds
    {Parameters : Type uParameters}
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Advice : Type uAdvice}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {RunningInstance : Type uRunningInstance}
    {RunningWitness : Type uRunningWitness}
    {FreshInstance : Type uFreshInstance}
    {FreshWitness : Type uFreshWitness}
    {semantics : Semantics Parameters Circuit Input Advice Output
      RunningStructure FreshStructure RunningInstance RunningWitness
      FreshInstance FreshWitness}
    {encoding : Encoding Circuit Input Advice Output
      RunningStructure FreshStructure FreshInstance FreshWitness}
    (layout : CompilerLayout semantics encoding) : Prop :=
  forall parameters circuit,
    layout.Fits parameters circuit ->
      semantics.structureRows (encoding.encodeStructures circuit).2 =
          layout.rowCapacity parameters /\
      semantics.structureColumns (encoding.encodeStructures circuit).2 =
          layout.columnCapacity parameters /\
      layout.paddedCanonical parameters
        (encoding.encodeStructures circuit).2

/-- A fitted compiler output has the exact parameter-owned rectangular
capacities. -/
theorem capacities_of_fits
    {Parameters : Type uParameters}
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Advice : Type uAdvice}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {RunningInstance : Type uRunningInstance}
    {RunningWitness : Type uRunningWitness}
    {FreshInstance : Type uFreshInstance}
    {FreshWitness : Type uFreshWitness}
    {semantics : Semantics Parameters Circuit Input Advice Output
      RunningStructure FreshStructure RunningInstance RunningWitness
      FreshInstance FreshWitness}
    {encoding : Encoding Circuit Input Advice Output
      RunningStructure FreshStructure FreshInstance FreshWitness}
    (layout : CompilerLayout semantics encoding)
    (holds : layout.Holds)
    {parameters : Parameters}
    {circuit : Circuit}
    (fits : layout.Fits parameters circuit) :
    semantics.structureRows (encoding.encodeStructures circuit).2 =
        layout.rowCapacity parameters /\
      semantics.structureColumns (encoding.encodeStructures circuit).2 =
        layout.columnCapacity parameters :=
  ⟨(holds parameters circuit fits).1,
    (holds parameters circuit fits).2.1⟩

/-- Every fitted assignment column lies in the common row-domain capacity.
This is the shape fact needed by the padded identity matrix; it does not add a
second SumCheck domain. -/
theorem columns_fit_row_domain
    {Parameters : Type uParameters}
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Advice : Type uAdvice}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {RunningInstance : Type uRunningInstance}
    {RunningWitness : Type uRunningWitness}
    {FreshInstance : Type uFreshInstance}
    {FreshWitness : Type uFreshWitness}
    {semantics : Semantics Parameters Circuit Input Advice Output
      RunningStructure FreshStructure RunningInstance RunningWitness
      FreshInstance FreshWitness}
    {encoding : Encoding Circuit Input Advice Output
      RunningStructure FreshStructure FreshInstance FreshWitness}
    (layout : CompilerLayout semantics encoding)
    (holds : layout.Holds)
    {parameters : Parameters}
    {circuit : Circuit}
    (fits : layout.Fits parameters circuit) :
    semantics.structureColumns (encoding.encodeStructures circuit).2 <=
      semantics.structureRows (encoding.encodeStructures circuit).2 := by
  rw [(holds parameters circuit fits).1,
    (holds parameters circuit fits).2.1]
  exact layout.columnsFitRows parameters

end CompilerLayout

/-- Deterministic parameter-dependent default pair. The witness type must
contain all commitment-opening data, including opening randomness. -/
structure DefaultAlgorithm
    (Parameters : Type uParameters)
    (RunningInstance : Type uRunningInstance)
    (RunningWitness : Type uRunningWitness) where
  choose : Parameters -> RunningInstance × RunningWitness

namespace DefaultAlgorithm

/-- The selected pair satisfies every admissible running structure for its
own public parameters. -/
def Holds
    {Parameters : Type uParameters}
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Advice : Type uAdvice}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {RunningInstance : Type uRunningInstance}
    {RunningWitness : Type uRunningWitness}
    {FreshInstance : Type uFreshInstance}
    {FreshWitness : Type uFreshWitness}
    (semantics : Semantics Parameters Circuit Input Advice Output
      RunningStructure FreshStructure RunningInstance RunningWitness
      FreshInstance FreshWitness)
    (default : DefaultAlgorithm Parameters RunningInstance RunningWitness) : Prop :=
  forall parameters relationStructure,
    semantics.runningStructureAdmissible parameters relationStructure ->
      semantics.runningHolds parameters relationStructure
        (default.choose parameters).1 (default.choose parameters).2

end DefaultAlgorithm

/-! ## Compact recursive verifier and statement binding -/

/-- Complete statement committed by the fixed-length statement identifier. -/
structure FullStatement
    (Parameters : Type uParameters)
    (RunningStructure : Type uRunningStructure)
    (FreshStructure : Type uFreshStructure)
    (VerifierKey : Type uVerifierKey) where
  parameters : Parameters
  runningStructure : RunningStructure
  freshStructure : FreshStructure
  verifierKey : VerifierKey

/-- Canonical, domain-separated statement encoding and fixed-width digest
representation. `hash` remains an abstract compression function. -/
structure StatementIdentifierScheme
    (FullStatement : Type uValue)
    (StatementId : Type uStatementId)
    (Word : Type uWord) where
  statementCodec : Codec FullStatement Word
  domainLabel : List Word
  hash : List Word -> StatementId
  identifierCodec : Codec StatementId Word
  identifierWidth : Nat

namespace StatementIdentifierScheme

def identifier
    {FullStatement : Type uValue}
    {StatementId : Type uStatementId}
    {Word : Type uWord}
    (scheme : StatementIdentifierScheme FullStatement StatementId Word)
    (statement : FullStatement) : StatementId :=
  scheme.hash (scheme.statementCodec.encode statement)

/-- Canonical statement syntax, a nonempty domain label, and a fixed-width
identifier representation. -/
def Holds
    {FullStatement : Type uValue}
    {StatementId : Type uStatementId}
    {Word : Type uWord}
    (scheme : StatementIdentifierScheme FullStatement StatementId Word) : Prop :=
  scheme.statementCodec.Canonical /\
  scheme.identifierCodec.Canonical /\
  scheme.domainLabel ≠ [] /\
  (forall statement, exists body,
    scheme.statementCodec.encode statement = scheme.domainLabel ++ body) /\
  (forall statementId,
    (scheme.identifierCodec.encode statementId).length =
      scheme.identifierWidth)

/-- Two distinct complete statements compressed to one identifier. -/
def Collision
    {FullStatement : Type uValue}
    {StatementId : Type uStatementId}
    {Word : Type uWord}
    (scheme : StatementIdentifierScheme FullStatement StatementId Word) : Prop :=
  exists left right,
    left ≠ right /\
    scheme.identifier left = scheme.identifier right

/-- Equal identifiers bind equal complete statements or expose the exact
statement-identifier collision. -/
theorem eq_or_collision
    {FullStatement : Type uValue}
    {StatementId : Type uStatementId}
    {Word : Type uWord}
    (scheme : StatementIdentifierScheme FullStatement StatementId Word)
    (left right : FullStatement)
    (sameIdentifier : scheme.identifier left = scheme.identifier right) :
    left = right \/ scheme.Collision := by
  classical
  by_cases sameStatement : left = right
  · exact Or.inl sameStatement
  · exact Or.inr ⟨left, right, sameStatement, sameIdentifier⟩

end StatementIdentifierScheme

/-- The only key data carried inside the recursive verifier circuit. Full
structures remain in the prover key and outer NIVC verifier key. -/
structure RecursiveVerifierKey
    (VerifierProjection : Type uVerifierProjection)
    (StatementId : Type uStatementId) where
  verifierProjection : VerifierProjection
  statementId : StatementId

/-- Data for one fixed augmented circuit's compact recursive verifier
projection. The full statement remains available to the outer verifier; only
its projection and identifier enter this fixed recursive circuit. -/
structure CompactVerifierInterface
    (Word : Type uWord)
    (Parameters : Type uParameters)
    (RunningStructure : Type uRunningStructure)
    (FreshStructure : Type uFreshStructure)
    (VerifierKey : Type uVerifierKey)
    (VerifierProjection : Type uVerifierProjection)
    (StatementId : Type uStatementId)
    (VerifierInput : Type uVerifierInput)
    (VerifierOutput : Type uVerifierOutput) where
  fixedStatement :
    FullStatement Parameters RunningStructure FreshStructure VerifierKey
  declaredSizeBound : Parameters -> Nat
  project : Parameters -> VerifierKey -> VerifierProjection
  projectionCodec : Codec VerifierProjection Word
  projectionWidth : Nat -> Nat
  projectionCompact : Parameters -> VerifierProjection -> Prop
  statementIdentifier : StatementIdentifierScheme
    (FullStatement Parameters RunningStructure FreshStructure VerifierKey)
    StatementId Word
  verifyFull :
    FullStatement Parameters RunningStructure FreshStructure VerifierKey ->
      VerifierInput -> VerifierOutput
  verifyRecursive :
    RecursiveVerifierKey VerifierProjection StatementId ->
      VerifierInput -> VerifierOutput

namespace CompactVerifierInterface

/-- Compute the fixed circuit's recursive key without copying either complete
structure. -/
def recursiveKey
    {Word : Type uWord}
    {Parameters : Type uParameters}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {VerifierKey : Type uVerifierKey}
    {VerifierProjection : Type uVerifierProjection}
    {StatementId : Type uStatementId}
    {VerifierInput : Type uVerifierInput}
    {VerifierOutput : Type uVerifierOutput}
    (interface : CompactVerifierInterface Word Parameters RunningStructure
      FreshStructure VerifierKey VerifierProjection StatementId VerifierInput
      VerifierOutput) :
    RecursiveVerifierKey VerifierProjection StatementId where
  verifierProjection :=
    interface.project interface.fixedStatement.parameters
      interface.fixedStatement.verifierKey
  statementId :=
    interface.statementIdentifier.identifier interface.fixedStatement

/-- The projection has a size-bound-fixed canonical layout, is declared
compact, and computes the same verifier result for the fixed full statement
when paired with its identifier. -/
def Holds
    {Word : Type uWord}
    {Parameters : Type uParameters}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {VerifierKey : Type uVerifierKey}
    {VerifierProjection : Type uVerifierProjection}
    {StatementId : Type uStatementId}
    {VerifierInput : Type uVerifierInput}
    {VerifierOutput : Type uVerifierOutput}
    (interface : CompactVerifierInterface Word Parameters RunningStructure
      FreshStructure VerifierKey VerifierProjection StatementId VerifierInput
      VerifierOutput) : Prop :=
  interface.projectionCodec.Canonical /\
  interface.statementIdentifier.Holds /\
  ((interface.projectionCodec.encode
      (interface.project interface.fixedStatement.parameters
        interface.fixedStatement.verifierKey)).length =
        interface.projectionWidth
          (interface.declaredSizeBound
            interface.fixedStatement.parameters) /\
    interface.projectionCompact interface.fixedStatement.parameters
      (interface.project interface.fixedStatement.parameters
        interface.fixedStatement.verifierKey)) /\
  (forall input : VerifierInput,
    interface.verifyRecursive interface.recursiveKey input =
      interface.verifyFull interface.fixedStatement input)

end CompactVerifierInterface

/-! ## Complete corrected Definition 12 contract -/

/-- Exact model-level part of corrected HyperNova Definition 12. -/
def Holds
    {Word : Type uWord}
    {Parameters : Type uParameters}
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Advice : Type uAdvice}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {RunningInstance : Type uRunningInstance}
    {RunningWitness : Type uRunningWitness}
    {FreshInstance : Type uFreshInstance}
    {FreshWitness : Type uFreshWitness}
    {VerifierKey : Type uVerifierKey}
    {VerifierProjection : Type uVerifierProjection}
    {StatementId : Type uStatementId}
    {VerifierInput : Type uVerifierInput}
    {VerifierOutput : Type uVerifierOutput}
    (semantics : Semantics Parameters Circuit Input Advice Output
      RunningStructure FreshStructure RunningInstance RunningWitness
      FreshInstance FreshWitness)
    (encoding : Encoding Circuit Input Advice Output
      RunningStructure FreshStructure FreshInstance FreshWitness)
    (layouts : CanonicalLayouts Word Parameters Circuit Input Advice Output
      RunningStructure FreshStructure RunningInstance RunningWitness
      FreshInstance FreshWitness)
    (compilerLayout : CompilerLayout semantics encoding)
    (default : DefaultAlgorithm Parameters RunningInstance RunningWitness)
    (compactVerifier : CompactVerifierInterface Word Parameters RunningStructure
      FreshStructure VerifierKey VerifierProjection StatementId VerifierInput
      VerifierOutput) : Prop :=
  NPComplete semantics encoding /\
  PartialFunctions encoding semantics.structuresCompatible /\
  Monotone semantics encoding /\
  layouts.Holds /\
  compilerLayout.Holds /\
  default.Holds semantics /\
  compactVerifier.Holds

end Nightstream.HyperNova.NIVCCompatibility
