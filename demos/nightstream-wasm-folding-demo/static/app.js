const source = document.querySelector("#source");
const exampleSelect = document.querySelector("#example");
const runButton = document.querySelector("#run");
const prepareButton = document.querySelector("#prepare");
const proveButton = document.querySelector("#prove");
const proofMode = document.querySelector("#proof-mode");
const status = document.querySelector("#status");
const traceBody = document.querySelector("#trace-body");
const rowDialog = document.querySelector("#row-dialog");
const rowDialogClose = document.querySelector("#row-dialog-close");
const rowJson = document.querySelector("#row-json");
let traceId = null;
let preparationId = null;
let customSource = null;
const programs = new Map();

const proofModeCopy = {
  folding_audit: "Fast folding sanity check over the normalized WASM relation. NIFS.V is not constrained inside F′; verification replays the full audit history and is not succinct.",
  ivc_no_memory: "Proves the normalized WASM transition relation with constrained NIFS.V. ROM and RAM consistency are not included.",
  nebula_memory: "Adds the Nebula ROM/RAM consistency relation and complete initial/final memory scan. This mode is substantially slower.",
};

const auxiliaryOpcode = {
  "aux::CallParamInit": "ParamInit",
  "aux::HostCallArg": "HostArg",
  "aux::HostCallResult": "HostResult",
  "aux::Padding": "Padding",
};

const text = (selector, value) => { document.querySelector(selector).textContent = value; };
const stackValue = (value) => value.hi == null
  ? String(value.lo)
  : `${BigInt.asIntN(64, (BigInt(value.hi) << 32n) | BigInt(value.lo))}i64`;
const access = (value) => value ? `s[${value.address / 2}] = ${stackValue(value)}` : "—";
const duration = (milliseconds) => milliseconds < 1000
  ? `${milliseconds} ms`
  : `${(milliseconds / 1000).toFixed(2)} s`;

function setButtonLoading(button, loading) {
  button.classList.toggle("is-loading", loading);
  button.setAttribute("aria-busy", String(loading));
}

function invalidatePreparation() {
  preparationId = null;
  prepareButton.disabled = traceId == null;
  proveButton.disabled = true;
  text("#proof-title", "Ready to preprocess");
  text("#proof-details", "");
  text("#preprocess-time", "—");
  text("#prove-time", "—");
  text("#verify-time", "—");
  text("#verifier-key-digest", "—");
  text("#memory-plan-digest", "—");
  text("#initial-ram-digest", "—");
  document.querySelector("#memory-plan-material").hidden = true;
  document.querySelector("#initial-ram-material").hidden = true;
}

function invalidateTrace() {
  traceId = null;
  invalidatePreparation();
}

function resetExecution() {
  text("#result", "—");
  text("#row-count", "—");
  text("#wasm-bytes", "—");
  text("#instructions", "—");
  text("#edges", "—");
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.colSpan = 7;
  cell.className = "empty";
  cell.textContent = "Trace the program to inspect the normalized VM rows.";
  row.append(cell);
  traceBody.replaceChildren(row);
}

function selectProgram(id) {
  const program = id === "custom" ? { source: customSource } : programs.get(id);
  if (!program || program.source == null) return;
  exampleSelect.value = id;
  source.value = program.source;
  invalidateTrace();
  resetExecution();
  status.classList.remove("error");
  status.textContent = "Ready";
}

function opcodeLabel(row) {
  const nesting = "│ ".repeat(row.call_depth_before);
  const transition = row.call_depth_after > row.call_depth_before
    ? "↳ "
    : row.call_depth_after < row.call_depth_before ? "↰ " : "";
  return `${nesting}${transition}${auxiliaryOpcode[row.kind] ?? row.opcode}`;
}

function variableAccess(row) {
  if (row.local_index != null) {
    return `l${row.local_index} ${row.local_read == null ? "←" : "→"} ${row.local_read ?? row.local_write}`;
  }
  if (row.global_index != null) {
    return `g${row.global_index} ${row.global_read == null ? "←" : "→"} ${row.global_read ?? row.global_write}`;
  }
  return "—";
}

function inspectRow(row) {
  text("#row-dialog-title", `Trace row ${row.cycle}`);
  rowJson.textContent = JSON.stringify(row, null, 2);
  rowDialog.showModal();
}

async function loadPrograms() {
  const response = await fetch("/api/program");
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "could not load examples");

  for (const program of payload.programs) {
    programs.set(program.id, program);
    const option = document.createElement("option");
    option.value = program.id;
    option.textContent = program.label;
    exampleSelect.append(option);
  }
  selectProgram(payload.default_id);
}

async function runTrace() {
  const tracedSource = source.value;
  invalidateTrace();
  resetExecution();
  runButton.disabled = true;
  status.classList.remove("error");
  status.textContent = "Tracing in native Wasmtime…";
  const started = performance.now();
  try {
    const response = await fetch("/api/trace", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ source: tracedSource }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "trace failed");
    if (source.value !== tracedSource) throw new Error("The program changed; trace it again.");
    traceId = payload.trace_id;
    prepareButton.disabled = false;
    render(payload);
    status.textContent = `Traced in ${Math.round(performance.now() - started)} ms`;
  } catch (error) {
    status.textContent = error.message;
    status.classList.add("error");
  } finally {
    runButton.disabled = false;
  }
}

async function preprocess() {
  if (traceId == null) return;
  const preparedTraceId = traceId;
  const preparedMode = proofMode.value;
  let traceExpired = false;
  invalidatePreparation();
  prepareButton.disabled = true;
  proveButton.disabled = true;
  setButtonLoading(prepareButton, true);
  text("#proof-title", "Preprocessing…");
  text("#proof-details", "Constructing the fixed relation and verifier material.");
  try {
    const response = await fetch("/api/prepare", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ trace_id: preparedTraceId, mode: preparedMode }),
    });
    const payload = await response.json();
    if (!response.ok) {
      traceExpired = response.status === 409;
      throw new Error(payload.error || "preprocessing failed");
    }
    if (traceId !== preparedTraceId || proofMode.value !== preparedMode) {
      throw new Error("The trace or proof mode changed; preprocess it again.");
    }
    preparationId = payload.preparation_id;
    proveButton.disabled = false;
    text("#proof-title", "Prepared · ready to fold");
    text("#preprocess-time", duration(payload.preprocess_ms));
    const workUnit = payload.mode === "folding_audit" ? "chunk(s)" : "fold(s)";
    text("#proof-details", `${payload.normalized_rows} rows · batch ${payload.batch_size} · ${payload.folds} ${workUnit} · ${payload.padded_rows} padded row(s). ${payload.security}`);
    text("#verifier-key-digest", payload.verifier_key_digest);
    if (payload.memory_plan_digest) {
      document.querySelector("#memory-plan-material").hidden = false;
      text("#memory-plan-digest", payload.memory_plan_digest.join(":"));
    }
    if (payload.initial_ram_digest) {
      document.querySelector("#initial-ram-material").hidden = false;
      text("#initial-ram-digest", payload.initial_ram_digest.join(":"));
    }
  } catch (error) {
    if (traceExpired) traceId = null;
    preparationId = null;
    proveButton.disabled = true;
    text("#proof-title", "Preprocessing unavailable");
    text("#proof-details", error.message);
  } finally {
    setButtonLoading(prepareButton, false);
    prepareButton.disabled = traceId == null;
  }
}

async function prove() {
  if (preparationId == null) return;
  const requestedPreparationId = preparationId;
  let preparationExpired = false;
  prepareButton.disabled = true;
  proveButton.disabled = true;
  setButtonLoading(proveButton, true);
  text("#proof-title", "Folding…");
  text("#prove-time", "running");
  text("#verify-time", "waiting");
  try {
    const response = await fetch("/api/prove", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ preparation_id: requestedPreparationId }),
    });
    const payload = await response.json();
    if (!response.ok) {
      preparationExpired = response.status === 409;
      throw new Error(payload.error || "proof failed");
    }
    if (preparationId !== requestedPreparationId) return;
    text("#proof-title", payload.mode === "folding_audit"
      ? "Verified folding audit · replay checked"
      : payload.memory_consistency
        ? "Verified proof with memory consistency"
        : "Verified recursive proof · memory unchecked");
    text("#prove-time", duration(payload.prove_ms));
    text("#verify-time", duration(payload.verify_ms));
    text("#result", payload.result[0] ?? (payload.trapped ? "trap" : "—"));
  } catch (error) {
    if (preparationId === requestedPreparationId || preparationExpired) {
      if (preparationExpired) preparationId = null;
      text("#proof-title", "Proof unavailable");
      text("#proof-details", error.message);
      text("#prove-time", "—");
      text("#verify-time", "—");
    }
  } finally {
    setButtonLoading(proveButton, false);
    prepareButton.disabled = traceId == null;
    proveButton.disabled = preparationId == null;
  }
}

function render(payload) {
  status.classList.remove("error");
  text("#result", payload.execution.results[0] ?? (payload.execution.trapped ? "trap" : "—"));
  text("#row-count", payload.execution.normalized_rows.toLocaleString());
  text("#wasm-bytes", payload.program.wasm_bytes.toLocaleString());
  text("#instructions", payload.program.decoded_instructions.toLocaleString());
  text("#edges", payload.program.control_edges.toLocaleString());
  traceBody.replaceChildren(...payload.rows.map((row) => {
    const tr = document.createElement("tr");
    tr.className = [
      row.kind === "program" ? "row-program" : "row-aux",
      row.trapped ? "row-trapped" : row.halted ? "row-terminal" : "",
    ].filter(Boolean).join(" ");
    tr.title = row.kind === "program" ? "program row" : row.kind;
    tr.tabIndex = 0;
    tr.setAttribute("aria-label", `Inspect trace row ${row.cycle} as JSON`);
    tr.addEventListener("click", () => inspectRow(row));
    tr.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        inspectRow(row);
      }
    });
    const cells = [row.cycle, opcodeLabel(row), `${row.pc_before} → ${row.pc_after}`, `${row.sp_before} → ${row.sp_after}`, row.stack_reads.map(access).join(", ") || "—", access(row.stack_write), variableAccess(row)];
    cells.forEach((value, index) => {
      const td = document.createElement("td");
      td.textContent = value;
      if (index === 1) td.className = "opcode";
      tr.append(td);
    });
    return tr;
  }));
}

runButton.addEventListener("click", runTrace);
prepareButton.addEventListener("click", preprocess);
proveButton.addEventListener("click", prove);
rowDialogClose.addEventListener("click", () => rowDialog.close());
proofMode.addEventListener("change", () => {
  text("#proof-copy", proofModeCopy[proofMode.value]);
  invalidatePreparation();
});
exampleSelect.addEventListener("change", () => selectProgram(exampleSelect.value));
source.addEventListener("input", () => {
  customSource = source.value;
  if (!exampleSelect.querySelector('option[value="custom"]')) {
    const option = document.createElement("option");
    option.value = "custom";
    option.textContent = "Scratch";
    exampleSelect.append(option);
  }
  exampleSelect.value = "custom";
  invalidateTrace();
  resetExecution();
  status.classList.remove("error");
  status.textContent = "Edited · trace required";
});
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") runTrace();
});
loadPrograms().catch((error) => { status.textContent = error.message; });
