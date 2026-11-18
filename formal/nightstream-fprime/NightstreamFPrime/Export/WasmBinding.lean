import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Spec.Nebula.Batch
import NightstreamFPrime.Spec.Nebula.SegmentGate

/-! Owns the check of the Rust export of the WASM catalog binding, one file per WASM size
(`export_wasm_rocq --binding`): the decoder, the checker, its soundness (`check_sound`), and the
binding of each active event of a row to a slot of that row under SPEC.md section 7.6 rules 1, 3
and 4 (`binds_of_accepted`). It does not own the text parser (`Export.WasmBindingMain`), the Rust
rows of the slot equations, or the relation between the logical and the physical catalog. -/

namespace NightstreamFPrime.Export

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Spec.Nebula.MemoryTrace

/-- A port of the built binding: its rank in the batched catalog, the physical position of its
memory, the namespace of that memory, and its step-witness columns. -/
structure WasmBinding.BoundPort where
  rank : ℕ
  memory : ℕ
  isRam : Bool
  spec : PortSpec

/-- The step-witness column of one-row column `column` in row `i` is `base + i * stride`. -/
structure WasmBinding.ColumnShift where
  column : Column
  base : ℕ
  stride : ℕ

/-- A decoded export of the catalog binding of one WASM size. -/
structure WasmBinding where
  size : WasmSize
  geometry : Geometry
  rows : ℕ
  logical : List MemorySpec
  table : List ℕ
  catalog : List MemorySpec
  regions : List Region
  columns : List WasmBinding.ColumnShift
  slots : List (List WasmBinding.BoundPort)

namespace WasmBinding

/-- The physical catalog of a logical catalog under a table map (Rust
`build_wasm_relation_layout`): memory `t` has the kind of the first logical memory of table
`t`, and the ports of each logical memory of table `t`, in logical order. -/
def overlay (logical : List MemorySpec) (table : List ℕ) : List MemorySpec :=
  (List.range (table.foldr (fun t count => max (t + 1) count) 0)).map fun t =>
    let members := (logical.zip table).filterMap fun entry =>
      if entry.2 = t then some entry.1 else none
    { memoryId := t, memoryKind := (members.head?.map MemorySpec.memoryKind).getD .rom,
      ports := members.flatMap MemorySpec.ports }

/-- The row of each rank of the batched catalog of `k` rows. -/
def rankRows (catalog : List MemorySpec) (k : ℕ) : List ℕ :=
  catalog.flatMap fun m => (List.range k).flatMap fun i => m.ports.map fun _ => i

/-- The ports of the batched catalog in rank order, each with its row and its memory id. -/
def rankedRows (catalog : List MemorySpec) (k : ℕ) (shift : ℕ → Column → Column) :
    List (ℕ × ℕ × PortSpec) :=
  catalog.flatMap fun m => (List.range k).flatMap fun i =>
    m.ports.map fun port => (i, m.memoryId, port.shift (shift i))

/-- The WASM sizes in the order of their export numbers. -/
def sizes : List WasmSize := [.micro, .small, .medium, .large]

/-! ### Decoder -/

private def atom : Value → Except String ℕ
  | .atom n => .ok n
  | .array _ => .error "expected a number"

private def items : Value → Except String (List Value)
  | .array values => .ok values
  | .atom _ => .error "expected an array"

private def listOf {α : Type} (item : Value → Except String α) (value : Value) :
    Except String (List α) := do
  (← items value).mapM item

private def portKindOf : Value → Except String PortKind
  | .array [.atom 0] => .ok .read
  | .array [.atom 1] => .ok (.write none)
  | .array [.atom 2, .atom c] => .ok (.write (some c))
  | _ => .error "expected a port kind [0], [1] or [2, c]"

private def activationOf : Value → Except String Activation
  | .array [.atom 0] => .ok .always
  | .array [.atom 1, .atom c] => .ok (.when c)
  | .array [.atom 2, .atom c] => .ok (.«unless» c)
  | _ => .error "expected an activation [0], [1, c] or [2, c]"

private def specOf (address : Value) (value : ℕ) (kind activation : Value) :
    Except String PortSpec :=
  return ⟨← listOf atom address, value, ← portKindOf kind, ← activationOf activation⟩

private def portOf : Value → Except String PortSpec
  | .array [address, .atom value, kind, activation] => specOf address value kind activation
  | _ => .error "expected a port [[a, ...], v, portKind, activation]"

private def memoryOf (id : ℕ) : Value → Except String MemorySpec
  | .array [.atom 0, ports] => return ⟨id, .rom, ← listOf portOf ports⟩
  | .array [.atom 1, ports] => return ⟨id, .ram, ← listOf portOf ports⟩
  | _ => .error "expected a memory [kind, [port, ...]]"

private def catalogOf (value : Value) : Except String (List MemorySpec) := do
  (← items value).zipIdx.mapM fun entry => memoryOf entry.2 entry.1

private def regionOf : Value → Except String Region
  | .array [.atom base, widths] => return ⟨base, ← listOf atom widths⟩
  | _ => .error "expected a region [base, [w, ...]]"

private def columnShiftOf : Value → Except String ColumnShift
  | .array [.atom column, .atom base, .atom stride] => .ok ⟨column, base, stride⟩
  | _ => .error "expected a column map entry [c, base, stride]"

private def boundPortOf : Value → Except String BoundPort
  | .array [.atom rank, .atom memory, .atom 0, address, .atom value, kind, activation] =>
    return ⟨rank, memory, false, ← specOf address value kind activation⟩
  | .array [.atom rank, .atom memory, .atom 1, address, .atom value, kind, activation] =>
    return ⟨rank, memory, true, ← specOf address value kind activation⟩
  | _ => .error "expected a bound port [rank, memory, isRam, [a, ...], v, portKind, activation]"

private def geometryOf : Value → Except String Geometry
  | .array [.atom r, .atom μ, .atom wTs, .atom bOps, .atom bScan, .atom n, .atom sMax] =>
    .ok ⟨r, μ, wTs, bOps, bScan, n, sMax⟩
  | _ => .error "expected a geometry [r, mu, wTs, bOps, bScan, n, sMax]"

/-- The export of schema 1 from its value. -/
def decode : Value → Except String WasmBinding
  | .array [.atom 1, .atom size, geometry, .atom rows, logical, table, catalog, regions, columns,
      slots] => do
    let some wasmSize := sizes[size]? | throw s!"size {size} is not 0, 1, 2 or 3"
    return ⟨wasmSize, ← geometryOf geometry, rows, ← catalogOf logical, ← listOf atom table,
      ← catalogOf catalog, ← listOf regionOf regions, ← listOf columnShiftOf columns,
      ← listOf (listOf boundPortOf) slots⟩
  | _ => .error
    "expected [1, size, geometry, rows, logical, table, catalog, regions, columns, slots]"

variable (x : WasmBinding)

/-- The step-witness column of one-row column `c` in row `i`, from the column map. -/
def shiftFn (i : ℕ) (c : Column) : Column :=
  match x.columns.find? (·.column == c) with
  | some entry => entry.base + i * entry.stride
  | none => 0

/-- Memory `m` has the region at position `m`. -/
def binding : Binding := ⟨fun m => x.regions[m]?⟩

/-- The catalog of a step: `rows` rows of the physical catalog. -/
def batched : List MemorySpec := batchCatalog x.catalog x.rows x.shiftFn

/-- The ranks of each slot. -/
def plan : List (List ℕ) := x.slots.map (·.map BoundPort.rank)

/-- Row `i` of a step witness. -/
def rowOf (step : Witness) (i : ℕ) : Witness := ⟨fun c => step.entries (x.shiftFn i c)⟩

/-! ### Checks -/

/-- Each logical memory has a table of its kind. -/
def tablesOk : Bool :=
  x.logical.length == x.table.length && (x.logical.zip x.table).all fun entry =>
    decide ((x.catalog[entry.2]?).map MemorySpec.memoryKind = some entry.1.memoryKind)

/-- The physical catalog is the overlay of the logical catalog. -/
def overlayOk : Bool := decide (x.catalog = overlay x.logical x.table)

/-- The logical memories have the ids `0, 1, …`. -/
def logicalIdsOk : Bool := decide (x.logical.map MemorySpec.memoryId = List.range x.logical.length)

/-- No other logical memory has the table of a logical memory with a write port. -/
def writesAloneOk : Bool :=
  (x.logical.zip x.table).all fun a => !(a.1.ports.any fun port => port.portKind != .read) ||
    (x.logical.zip x.table).all fun b => b.2 != a.2 || b.1.memoryId == a.1.memoryId

/-- One region per physical memory. -/
def regionCountOk : Bool := x.regions.length == x.catalog.length

/-- Each region is aligned to its size. -/
def alignedOk : Bool := x.regions.all fun r => decide (r.size ∣ r.base)

/-- Each region is inside its space. -/
def inSpaceOk : Bool :=
  x.regions.zipIdx.all fun entry => decide (entry.1.base + entry.1.size ≤
    if isRam x.catalog entry.2 then x.geometry.ramSize else x.geometry.romSize)

/-- The regions of one space are disjoint. -/
def disjointOk : Bool :=
  x.regions.zipIdx.all fun a => x.regions.zipIdx.all fun b =>
    decide (a.2 = b.2 ∨ isRam x.catalog a.2 ≠ isRam x.catalog b.2 ∨
      a.1.base + a.1.size ≤ b.1.base ∨ b.1.base + b.1.size ≤ a.1.base)

/-- Each port has one address column per width of its region. -/
def widthsOk : Bool :=
  x.catalog.all fun m => m.ports.all fun port =>
    decide ((x.regions[m.memoryId]?).map (·.widths.length) = some port.addressColumns.length)

/-- Each column that a port reads has an entry in the column map. -/
def columnsOk : Bool :=
  x.catalog.all fun m => m.ports.all fun port => port.columns.all fun c =>
    x.columns.any fun entry => decide (entry.column = c)

/-- `bOps` slots, in equal groups of the rows. -/
def slotCountOk : Bool :=
  decide (0 < x.rows ∧ x.rows ∣ x.geometry.bOps ∧ x.slots.length = x.geometry.bOps)

/-- The slots name each rank once. -/
def namesOk : Bool :=
  decide (x.plan.flatten.mergeSort (fun a b => decide (a ≤ b)) =
    List.range (rankRows x.catalog x.rows).length)

/-- Each bound port is the port of its rank, with the namespace of its memory, in a slot of its
row. -/
def wiredOk : Bool :=
  let ports := (rankedRows x.catalog x.rows x.shiftFn).toArray
  x.slots.zipIdx.all fun entry => entry.1.all fun q =>
    decide (ports[q.rank]? = some (entry.2 / (x.geometry.bOps / x.rows), q.memory, q.spec) ∧
      q.isRam = isRam x.catalog q.memory)

/-- The checks of an export in order, each with its rejection reason. -/
def checks : List (String × (WasmBinding → Bool)) :=
  [("a logical memory has no table of its kind", tablesOk),
   ("the physical catalog is not the overlay of the logical catalog", overlayOk),
   ("the logical memories do not have the ids 0, 1, ...", logicalIdsOk),
   ("a written logical memory shares its table", writesAloneOk),
   ("the regions are not one per physical memory", regionCountOk),
   ("a region is not aligned to its size", alignedOk),
   ("a region is not inside its space", inSpaceOk),
   ("two regions of one space overlap", disjointOk),
   ("a port has not one address column per region width", widthsOk),
   ("a port reads a column that the column map does not have", columnsOk),
   ("the slots are not bOps slots in equal groups of the rows", slotCountOk),
   ("the slots do not name each rank once", namesOk),
   ("a bound port is not the port of its rank in a slot of its row", wiredOk)]

/-- The checker. -/
def check : Bool := checks.all fun entry => entry.2 x

/-- The reason of the first check that fails. -/
def reason : Option String := (checks.find? fun entry => !entry.2 x).map Prod.fst

/-- The facts of an accepted export. -/
structure Accepted : Prop where
  isOverlay : x.catalog = overlay x.logical x.table
  logicalIds : x.logical.map MemorySpec.memoryId = List.range x.logical.length
  writesAlone : ∀ a ∈ x.logical.zip x.table, (∃ port ∈ a.1.ports, port.portKind ≠ .read) →
    ∀ b ∈ x.logical.zip x.table, b.2 = a.2 → b.1.memoryId = a.1.memoryId
  distinct : (x.catalog.map MemorySpec.memoryId).Nodup
  tableKinds : List.Forall₂ (fun m t => (x.catalog[t]?).map MemorySpec.memoryKind =
    some m.memoryKind) x.logical x.table
  regionCount : x.regions.length = x.catalog.length
  disjoint : x.binding.Disjoint x.catalog
  aligned : ∀ m r, x.binding.region m = some r → r.size ∣ r.base
  inSpace : ∀ m r, x.binding.region m = some r →
    r.base + r.size ≤ if isRam x.catalog m then x.geometry.ramSize else x.geometry.romSize
  widths : ∀ m ∈ x.catalog, ∀ port ∈ m.ports,
    ∃ r, x.binding.region m.memoryId = some r ∧ port.addressColumns.length = r.widths.length
  mapped : ∀ m ∈ x.catalog, ∀ port ∈ m.ports, ∀ c ∈ port.columns, ∃ entry ∈ x.columns,
    entry.column = c
  rowsPos : 0 < x.rows
  rowsDvd : x.rows ∣ x.geometry.bOps
  slotCount : x.slots.length = x.geometry.bOps
  names : x.plan.flatten.Perm (List.range (rankedPorts x.batched).length)
  wired : ∀ slot ∈ x.slots, ∀ q ∈ slot, (rankedPorts x.batched)[q.rank]? = some (q.memory, q.spec) ∧
    q.isRam = isRam x.catalog q.memory
  rowSlots : ∀ j (hj : j < x.slots.length), ∀ q ∈ x.slots[j],
    (rankRows x.catalog x.rows)[q.rank]? = some (j / (x.geometry.bOps / x.rows))

/-- SPEC.md section 7.6 rules 1, 3 and 4 for the active bound port `q` of a slot with access
`a`: the namespace and the write flag of the port, the packed address parts in the region of
its memory, and the value columns. -/
def SlotEquations (q : BoundPort) (step : Witness) (a : PortAccess) : Prop :=
  a.isRam = q.isRam ∧
    (∃ r, x.binding.region q.memory = some r ∧ r.Fits (q.spec.addressColumns.map step) ∧
      a.addr = r.addr (q.spec.addressColumns.map step)) ∧
    match q.spec.portKind with
    | .read => a.isWrite = false ∧ (a.vr : ℤ) = step q.spec.valueColumn
    | .write prior => a.isWrite = true ∧ (a.vw : ℤ) = step q.spec.valueColumn ∧
        ∀ c ∈ prior, (a.vr : ℤ) = step c

/-! ### Soundness -/

variable {catalog : List MemorySpec} {k : ℕ} {shift : ℕ → Column → Column}

/-- The ports of `rankedRows` are the ranked ports of the batched catalog. -/
theorem map_snd_rankedRows :
    (rankedRows catalog k shift).map Prod.snd = rankedPorts (batchCatalog catalog k shift) := by
  simp [rankedRows, rankedPorts, batchCatalog, List.map_flatMap, List.flatMap_map,
    Function.comp_def]

/-- The rows of `rankedRows` are `rankRows`. -/
theorem map_fst_rankedRows : (rankedRows catalog k shift).map Prod.fst = rankRows catalog k := by
  simp [rankedRows, rankRows, List.map_flatMap, Function.comp_def]

/-- Each port of row `i < k` of the catalog has a rank in `rankedRows`. -/
theorem exists_rankedRows {m : MemorySpec} (inCatalog : m ∈ catalog) {i : ℕ} (row : i < k)
    {port : PortSpec} (inPorts : port ∈ m.ports) :
    ∃ r : ℕ, (rankedRows catalog k shift)[r]? = some (i, m.memoryId, port.shift (shift i)) :=
  List.mem_iff_getElem?.mp (List.mem_flatMap.mpr ⟨m, inCatalog,
    List.mem_flatMap.mpr ⟨i, List.mem_range.mpr row, List.mem_map.mpr ⟨port, inPorts, rfl⟩⟩⟩)

/-- The port of a rank of row `i` is the row-`i` shift of a port of the catalog. -/
theorem rankedPorts_of_rankRows {r i : ℕ} (row : (rankRows catalog k)[r]? = some i) :
    ∃ m ∈ catalog, ∃ port ∈ m.ports, (rankedPorts (batchCatalog catalog k shift))[r]? =
      some (m.memoryId, port.shift (shift i)) := by
  rw [← map_fst_rankedRows (shift := shift), List.getElem?_map, Option.map_eq_some_iff] at row
  obtain ⟨entry, found, rfl⟩ := row
  obtain ⟨m, inCatalog, member⟩ := List.mem_flatMap.mp (List.mem_iff_getElem?.mpr ⟨r, found⟩)
  obtain ⟨i, -, member⟩ := List.mem_flatMap.mp member
  obtain ⟨port, inPorts, rfl⟩ := List.mem_map.mp member
  refine ⟨m, inCatalog, port, inPorts, ?_⟩
  rw [← map_snd_rankedRows, List.getElem?_map, found]
  rfl

/-- The overlay has the memory ids `0, 1, …`. -/
theorem distinct_overlay (logical : List MemorySpec) (table : List ℕ) :
    ((overlay logical table).map MemorySpec.memoryId).Nodup := by
  simp [overlay, List.map_map, Function.comp_def, List.nodup_range]

/-- The wiring check at a bound port of slot `j`. -/
private theorem wired_of_check {x : WasmBinding} (wired : x.wiredOk = true) {j : ℕ}
    (hj : j < x.slots.length) {q : BoundPort} (inSlot : q ∈ x.slots[j]) :
    (rankedRows x.catalog x.rows x.shiftFn)[q.rank]? =
      some (j / (x.geometry.bOps / x.rows), q.memory, q.spec) ∧
      q.isRam = isRam x.catalog q.memory := by
  have entry : (x.slots[j], j) ∈ x.slots.zipIdx :=
    List.mem_zipIdx_iff_getElem?.mpr (List.getElem?_eq_getElem hj)
  have := List.all_eq_true.mp (List.all_eq_true.mp wired _ entry) q inSlot
  simpa using this

/-- An export that the checker accepts has the facts of `Accepted`. -/
theorem check_sound {x : WasmBinding} (accepted : x.check = true) : x.Accepted := by
  simp only [check, checks, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true]
    at accepted
  obtain ⟨tables, overlaid, logicalIds, writesAlone, regionCount, aligned, inSpace, disjoint,
    widths, columns, slotCount, names, wired⟩ := accepted
  have overlaid : x.catalog = overlay x.logical x.table := of_decide_eq_true overlaid
  obtain ⟨tableCount, tables⟩ := Bool.and_eq_true _ _ ▸ tables
  obtain ⟨rowsPos, rowsDvd, slotCount⟩ := of_decide_eq_true slotCount
  refine
    { isOverlay := overlaid
      distinct := overlaid ▸ distinct_overlay _ _
      logicalIds := of_decide_eq_true logicalIds
      writesAlone := fun a inA written b inB same => by
        have := List.all_eq_true.mp writesAlone a inA
        simp only [Bool.or_eq_true, Bool.not_eq_true', List.any_eq_false, bne_iff_ne, ne_eq,
          Decidable.not_not] at this
        rcases this with none | alone
        · obtain ⟨port, inPorts, write⟩ := written
          exact absurd (none port inPorts) write
        · have := List.all_eq_true.mp alone b inB
          simpa [same] using this
      tableKinds := List.forall₂_iff_zip.mpr ⟨beq_iff_eq.mp tableCount, fun member =>
        of_decide_eq_true (List.all_eq_true.mp tables _ member)⟩
      regionCount := beq_iff_eq.mp regionCount
      disjoint := fun i j ri rj hi hj apart same => ?_
      aligned := fun m r found =>
        of_decide_eq_true (List.all_eq_true.mp aligned r (List.mem_iff_getElem?.mpr ⟨m, found⟩))
      inSpace := fun m r found =>
        of_decide_eq_true (List.all_eq_true.mp inSpace (r, m)
          (List.mem_zipIdx_iff_getElem?.mpr found))
      widths := fun m inCatalog port inPorts => by
        have := of_decide_eq_true
          (List.all_eq_true.mp (List.all_eq_true.mp widths m inCatalog) port inPorts)
        obtain ⟨r, found, same⟩ := Option.map_eq_some_iff.mp this
        exact ⟨r, found, same.symm⟩
      mapped := fun m inCatalog port inPorts c read => by
        obtain ⟨entry, member, same⟩ := List.any_eq_true.mp (List.all_eq_true.mp
          (List.all_eq_true.mp (List.all_eq_true.mp columns m inCatalog) port inPorts) c read)
        exact ⟨entry, member, of_decide_eq_true same⟩
      rowsPos
      rowsDvd
      slotCount
      names := ?_
      wired := fun slot inSlots q inSlot => ?_
      rowSlots := fun j hj q inSlot => ?_ }
  · have := of_decide_eq_true (List.all_eq_true.mp (List.all_eq_true.mp disjoint (ri, i)
      (List.mem_zipIdx_iff_getElem?.mpr hi)) (rj, j) (List.mem_zipIdx_iff_getElem?.mpr hj))
    rcases this with same' | differ | below
    · exact absurd same' apart
    · exact absurd same differ
    · exact below
  · have sorted := of_decide_eq_true names
    rw [batched, ← map_snd_rankedRows, List.length_map, ← List.length_map Prod.fst,
      map_fst_rankedRows, ← sorted]
    exact (List.mergeSort_perm _ _).symm
  · obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem inSlots
    obtain ⟨found, ram⟩ := wired_of_check wired hj inSlot
    refine ⟨?_, ram⟩
    rw [batched, ← map_snd_rankedRows, List.getElem?_map, found]
    rfl
  · rw [← map_fst_rankedRows (shift := x.shiftFn), List.getElem?_map,
      (wired_of_check wired hj inSlot).1]
    rfl

/-- The slot equations of an active bound port that shows event `e` bind `e` in range. -/
theorem binds_of_slotEquations {x : WasmBinding} {q : BoundPort} {step : Witness} {e : Event}
    {a : PortAccess} (ram : q.isRam = isRam x.catalog q.memory)
    (shown : portEvent q.memory step q.spec = some e) (equations : x.SlotEquations q step a) :
    x.binding.Binds x.catalog e a ∧ x.binding.InRange e.memory e.address := by
  obtain ⟨rank, memory, isRamq, address, value, kind, activation⟩ := q
  obtain ⟨sameRam, ⟨r, found, fits, addr⟩, values⟩ := equations
  simp only at ram sameRam found fits addr values
  unfold portEvent at shown
  split_ifs at shown
  have at_ : x.binding.At x.catalog memory (address.map step) a :=
    ⟨r, found, sameRam.trans ram, addr⟩
  cases kind with
  | read =>
    cases shown
    exact ⟨⟨at_, values⟩, r, found, fits⟩
  | write prior =>
    cases shown
    obtain ⟨write, written, before⟩ := values
    refine ⟨⟨at_, write, written, fun v member => ?_⟩, r, found, fits⟩
    obtain ⟨c, inPrior, rfl⟩ := Option.mem_map.mp member
    exact before c inPrior

/-- Row `i` of a step witness holds the columns of row `i` at their shifted columns. -/
theorem holdsRows_rowOf (x : WasmBinding) (step : Witness) :
    HoldsRows x.catalog x.rows x.shiftFn step ((List.range x.rows).map (x.rowOf step)) :=
  ⟨by simp, fun i _ _ _ _ _ _ _ => by simp only [List.getElem_map, List.getElem_range]; rfl⟩

/-- In an accepted export, an access that obeys the slot equations of a bound port binds the
event of the port's rank in range. -/
theorem binds_of_rankEvent {x : WasmBinding} (accepted : Accepted x) {slot : List BoundPort}
    (inSlots : slot ∈ x.slots) {q : BoundPort} (inSlot : q ∈ slot) {step : Witness} {e : Event}
    (shown : rankEvent x.batched step q.rank = some e) {a : PortAccess}
    (equations : x.SlotEquations q step a) :
    x.binding.Binds x.catalog e a ∧ x.binding.InRange e.memory e.address := by
  obtain ⟨wire, ram⟩ := accepted.wired slot inSlots q inSlot
  rw [rankEvent, wire] at shown
  exact binds_of_slotEquations ram shown equations

/-- In an accepted export, each bound port of slot `j` is the shift of a port of the catalog
to row `j / (bOps / rows)`. -/
theorem port_of_accepted {x : WasmBinding} (accepted : Accepted x) {j : ℕ}
    (hj : j < x.slots.length) {q : BoundPort} (inSlot : q ∈ x.slots[j]) :
    ∃ m ∈ x.catalog, ∃ port ∈ m.ports, q.memory = m.memoryId ∧
      q.spec = port.shift (x.shiftFn (j / (x.geometry.bOps / x.rows))) := by
  obtain ⟨m, inCatalog, port, inPorts, found⟩ :=
    rankedPorts_of_rankRows (shift := x.shiftFn) (accepted.rowSlots j hj q inSlot)
  have wire := (accepted.wired _ (List.getElem_mem hj) q inSlot).1
  rw [batched, found, Option.some.injEq, Prod.mk.injEq] at wire
  exact ⟨m, inCatalog, port, inPorts, wire.1.symm, wire.2.symm⟩

/-- In an accepted export, each active event of row `i` of a step is the event of a bound port
of a slot of row `i`, and an access that obeys the slot equations of that port binds the event
in range. -/
theorem binds_of_accepted {x : WasmBinding} (accepted : Accepted x) (step : Witness)
    {i : ℕ} (row : i < x.rows) {m : MemorySpec} (inCatalog : m ∈ x.catalog) {port : PortSpec}
    (inPorts : port ∈ m.ports) {e : Event}
    (shown : portEvent m.memoryId (x.rowOf step i) port = some e) :
    ∃ j, ∃ hj : j < x.slots.length, j / (x.geometry.bOps / x.rows) = i ∧
      ∃ q ∈ x.slots[j], rankEvent x.batched step q.rank = some e ∧
        ∀ a, SlotEquations x q step a →
          x.binding.Binds x.catalog e a ∧ x.binding.InRange e.memory e.address := by
  obtain ⟨r, found⟩ := exists_rankedRows (shift := x.shiftFn) inCatalog row inPorts
  have small : r < (rankedPorts x.batched).length := by
    rw [batched, ← map_snd_rankedRows, List.length_map]
    exact (List.getElem?_eq_some_iff.mp found).1
  obtain ⟨ranks, inPlan, named⟩ :=
    List.mem_flatten.mp (accepted.names.symm.subset (List.mem_range.mpr small))
  obtain ⟨slot, inSlots, rfl⟩ := List.mem_map.mp inPlan
  obtain ⟨q, inSlot, rfl⟩ := List.mem_map.mp named
  obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem inSlots
  have rowEq := accepted.rowSlots j hj q inSlot
  rw [← map_fst_rankedRows (shift := x.shiftFn), List.getElem?_map, found, Option.map_some,
    Option.some.injEq] at rowEq
  have event : rankEvent x.batched step q.rank = some e := by
    rw [rankEvent, batched, ← map_snd_rankedRows, List.getElem?_map, found]
    exact (portEvent_shift (f := x.shiftFn i) (step := step) (w := x.rowOf step i) m.memoryId
      fun _ _ => rfl).trans shown
  exact ⟨j, hj, rowEq.symm, q, inSlot, event, fun a equations =>
    binds_of_rankEvent accepted inSlots inSlot event equations⟩

end WasmBinding

end NightstreamFPrime.Export
