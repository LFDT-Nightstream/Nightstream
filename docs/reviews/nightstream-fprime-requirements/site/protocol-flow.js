/* Protocol operations, exact handoffs and phase results use the same requirement records. */
function replayExecutionPassed(record) {
  return record.replay_execution === 'passed';
}

function replayProofComplete(record) {
  return record.proof === 'proved' && record.connection === 'connected';
}

function createReplayProgress({data, byId, el}) {
  const replay = data.prover_replay;
  const all = replay.phases.flatMap(phase => phase.records);
  const passed = all.filter(id => replayExecutionPassed(byId.get(id))).length;
  const panel = el('div', undefined, 'replay-panel');
  const title = el('div', undefined, 'replay-heading');
  title.append(el('h2', replay.title), el('span', passed + '/' + all.length + ' execution checks passed',
    'status ' + (passed === all.length ? 'status-good' : 'status-open')));
  panel.append(title, el('p', replay.summary));
  const stages = el('ol', undefined, 'replay-stages');
  for (const phase of replay.phases) {
    const done = phase.records.filter(id => replayExecutionPassed(byId.get(id))).length;
    const proofComplete = (phase.proof_record ? [phase.proof_record] : phase.records)
      .every(id => replayProofComplete(byId.get(id)));
    const row = el('li');
    const link = el('a', undefined, 'replay-phase' + (phase.records.includes(replay.next) ? ' replay-current' : ''));
    link.href = '#req-' + phase.id + '.replay';
    link.append(el('strong', phase.label), el('span', done + '/' + phase.records.length + ' execution checks passed',
      done === phase.records.length ? 'axis-rust' : 'replay-open'),
      el('span', (phase.proof_label || 'Formal proof connection') + ': ' + (proofComplete ? 'complete' : 'open'),
        proofComplete ? 'axis-proof' : 'replay-open'), el('span', phase.detail, 'flow-short'));
    row.append(link); stages.append(row);
  }
  panel.append(stages);
  for (const phase of replay.phases.filter(item => item.proof_record)) {
    const source = el('a', phase.label + ' ' + phase.proof_label.toLowerCase());
    source.href = '#req-' + phase.proof_record;
    const reference = el('p'); reference.append(source); panel.append(reference);
  }
  const next = el('p', undefined, 'replay-next'); next.append(el('strong', 'Next recorded step: '));
  const link = el('a', replay.next_step || byId.get(replay.next).label); link.href = '#req-' + replay.next; next.append(link); panel.append(next);
  panel.append(el('p', replay.boundary, 'replay-boundary'));
  const source = el('a', 'Goal and committed replay reports', 'replay-source');
  source.href = data.provenance.repository + '/blob/' + replay.source_commit + '/' + replay.goal_path;
  panel.append(source);
  return panel;
}

// Orthogonal routes keep long connections in the spaces around the cards.
function protocolFlowPath(edge, boxes, bounds) {
  const s = boxes.get(edge.from), t = boxes.get(edge.to);
  const sx = s.x + s.width / 2, sy = s.y + s.height / 2;
  const tx = t.x + t.width / 2, ty = t.y + t.height / 2;
  let points;
  if (edge.kind === 'feedback') {
    points = [[s.x + s.width, sy], [bounds.right, sy], [bounds.right, bounds.top],
      [bounds.left, bounds.top], [bounds.left, ty], [t.x, ty]];
  } else if (edge.kind === 'encode') {
    const rail = bounds.right - 12;
    points = [[s.x + s.width, sy], [rail, sy], [rail, ty], [t.x + t.width, ty]];
  } else if (edge.kind === 'base') {
    points = [[sx, s.y + s.height], [sx, bounds.bottom], [tx, bounds.bottom], [tx, t.y + t.height]];
  } else if (edge.kind === 'check') {
    const rail = s.x - 14;
    points = [[s.x, sy], [rail, sy], [rail, t.y - 14], [tx, t.y - 14], [tx, t.y]];
  } else if (Math.abs(sx - tx) < 1) {
    const between = [...boxes.values()].some(b => b !== s && b !== t &&
      Math.abs(b.x - s.x) < 1 && b.y > s.y && b.y < t.y);
    points = between ? [[s.x, sy], [s.x - 8, sy], [s.x - 8, ty], [t.x, ty]] :
      [[sx, s.y + s.height], [tx, t.y]];
  } else {
    const rightward = tx > sx;
    const start = rightward ? s.x + s.width : s.x;
    const end = rightward ? t.x : t.x + t.width;
    const mid = (start + end) / 2;
    points = [[start, sy], [mid, sy], [mid, ty], [end, ty]];
  }
  return points.map(([x, y], i) => (i ? 'L' : 'M') + ' ' + x + ' ' + y).join(' ');
}

function createProtocolFlow({flow, byId, el, names, sourceUrl, premiseLink, replay}) {
  const items = new Map(flow.items.map(item => [item.id, item]));
  const replayIds = new Set(replay?.phases.flatMap(phase => phase.records) || []);
  const axes = [['proof', 'Proof'], ['connection', 'Link'], ['rust', 'Rust']];
  let built = false, dialog, detail, diagram, wires, observer;
  const cards = new Map();
  const badges = records => {
    const row = el('span', undefined, 'axis-counts');
    for (const [axis, label] of axes) {
      const counts = RequirementAssurance.counts(records.map(key => byId.get(key)), axis);
      const {finished, total} = RequirementAssurance.progress(axis, counts);
      const badge = el('span', undefined, 'axis-count axis-' + axis);
      badge.append(el('span', label), el('strong', total ? finished + '/' + total : 'n/a'));
      badge.title = RequirementAssurance.categories[axis].map(([key, name]) => name + ' ' + counts[key]).join(' · ');
      row.append(badge);
    }
    return row;
  };
  function card(key) {
    const item = items.get(key);
    const node = el('a', undefined, 'flow-card flow-' + item.kind);
    node.href = '#protocol-flow:' + key;
    node.dataset.flowItem = key;
    cards.set(key, node);
    if (item.actor) node.append(el('span', item.actor, 'flow-actor'));
    node.append(el('strong', item.label), el('span', item.short, 'flow-short'));
    node.append(badges(item.records));
    const replayRecords = item.records.filter(id => replayIds.has(id));
    if (replayRecords.length) {
      const done = replayRecords.filter(id => replayExecutionPassed(byId.get(id))).length;
      node.append(el('span', 'Replay execution: ' + done + '/' + replayRecords.length + ' passed',
        'flow-replay ' + (done === replayRecords.length ? 'axis-rust' : 'replay-open')));
    }
    return node;
  }
  function inspect(item) {
    detail.replaceChildren();
    detail.append(el('span', item.kind === 'handoff' ? 'Exact-value connection' : item.kind === 'assurance' ? 'Result for the whole phase' : item.actor || 'Protocol operation', 'flow-actor'));
    const heading = el('h2', item.label); heading.id = 'flow-detail-title';
    detail.append(heading, el('p', item.detail));
    if (item.formula) detail.append(el('p', item.formula, 'flow-formula'));
    detail.append(badges(item.records));
    const link = el('a', 'Permanent link to this step'); link.href = '#protocol-flow:' + item.id;
    const permalink = el('p'); permalink.append(link); detail.append(permalink);
    const guide = el('dl', undefined, 'flow-status-guide');
    for (const [label, meaning] of [['Proof', 'The stated local result in Lean.'], ['Link', 'Its connection to the required values, circuit or consumer.'], ['Rust', 'Code and the exact scope of executed checks.']]) {
      guide.append(el('dt', label), el('dd', meaning));
    }
    detail.append(guide);
    const connections = flow.edges.filter(edge => edge.from === item.id || edge.to === item.id);
    if (connections.length) {
      const adjacent = el('div', undefined, 'uses'); adjacent.append(el('strong', 'Flow connections'));
      for (const edge of connections) {
        const other = edge.from === item.id ? edge.to : edge.from;
        const label = edge.label || (edge.kind === 'circuit' ? 'Circuit flow' : 'Value flow');
        const a = el('a', (edge.from === item.id ? 'To: ' : 'From: ') + items.get(other).label + ' · ' + label);
        a.href = '#protocol-flow:' + other; adjacent.append(a);
      }
      detail.append(adjacent);
    }
    if (item.uses?.length) {
      const uses = el('div', undefined, 'uses'); uses.append(el('strong', 'Shared support'));
      for (const key of item.uses) { const a = el('a', byId.get(key).label); a.href = '#req-' + key; uses.append(a); }
      detail.append(uses);
    }
    detail.append(el('h3', 'Requirements and evidence'));
    for (const key of item.records) {
      const record = byId.get(key), entry = el('details', undefined, 'flow-record');
      const summary = el('summary'); summary.append(el('strong', record.label));
      const statuses = el('span', undefined, 'flow-record-status');
      for (const [axis, label] of axes) statuses.append(el('span', label + ': ' + names[record[axis]], 'axis-' + axis));
      summary.append(statuses); entry.append(summary, el('p', record.requirement));
      if (record.remaining) entry.append(el('p', record.remaining, 'flow-scope'));
      const a = el('a', 'Full requirement: ' + key); a.href = '#req-' + key; entry.append(a);
      if (record.depends_on?.length) {
        const uses = el('div', undefined, 'uses'); uses.append(el('strong', 'Uses'));
        for (const id of record.depends_on) {
          const dependency = el('a', byId.get(id).label); dependency.href = '#req-' + id; uses.append(dependency);
        }
        entry.append(uses);
      }
      for (const ref of [...(record.paper || []), ...(record.code || [])]) {
        const url = sourceUrl(ref.path, ref.line);
        const reference = el(url ? 'a' : 'span', ref.symbol || ref.section || ref.path, 'flow-reference');
        if (url) reference.href = url;
        reference.title = ref.path + ':' + ref.line; entry.append(reference);
      }
      for (const id of record.assumption_ids || []) entry.append(premiseLink(id));
      detail.append(entry);
    }
    if (!dialog.open) dialog.showModal();
    dialog.scrollTop = 0;
  }
  function svg(tag, attributes) {
    const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    return element;
  }
  function draw() {
    const rect = diagram.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    wires.replaceChildren();
    wires.setAttribute('viewBox', '0 0 ' + rect.width + ' ' + rect.height);
    const definitions = svg('defs', {});
    for (const type of ['data', 'recursion']) {
      const marker = svg('marker', {id: 'flow-arrow-' + type, viewBox: '0 0 10 10',
        refX: '9', refY: '5', markerWidth: '6', markerHeight: '6', orient: 'auto-start-reverse'});
      marker.append(svg('path', {d: 'M 0 0 L 10 5 L 0 10 z', class: 'flow-arrow-' + type}));
      definitions.append(marker);
    }
    wires.append(definitions);
    const boxes = new Map();
    for (const [key, node] of cards) {
      const box = node.getBoundingClientRect();
      boxes.set(key, {x: box.left - rect.left, y: box.top - rect.top, width: box.width, height: box.height});
    }
    for (const edge of flow.edges) {
      const path = svg('path', {d: protocolFlowPath(edge, boxes,
        {left: 12, right: rect.width - 12, top: 20, bottom: rect.height - 26}),
        class: 'flow-wire flow-wire-' + edge.kind,
        'marker-end': 'url(#flow-arrow-' + (['data', 'backend'].includes(edge.kind) ? 'data' : 'recursion') + ')'});
      wires.append(path);
    }
  }
  function render(container, selected) {
    if (!built) {
      const key = el('div', undefined, 'flow-key');
      key.append(el('strong', '1 fresh + 16 running → 17 aligned → 1 combined → 16 children'));
      key.append(el('span', 'Goldilocks · b = 2 · kρ = 16 · B = 65,536', 'flow-short'));
      const markdown = el('a', 'Read this flow in Markdown'); markdown.href = 'protocol-flow.md'; key.append(markdown);
      container.append(key);
      const legend = el('p', undefined, 'flow-legend');
      legend.append(el('span', '→ Protocol values'), el('span', '→ Circuit and next iteration', 'flow-recursion-key'),
        el('span', '⇢ Same checks and values / base-case bypass', 'flow-recursion-key'));
      container.append(legend);
      const scroll = el('div', undefined, 'flow-scroll'); scroll.tabIndex = 0;
      scroll.setAttribute('aria-label', 'Complete protocol and recursive circuit. Scroll horizontally on narrow screens.');
      diagram = el('div', undefined, 'protocol-diagram');
      const feedback = el('p', flow.feedback_note, 'flow-feedback'); diagram.append(feedback);
      diagram.append(el('p', 'Recursive fold · i > 0', 'flow-fold-label'));
      wires = svg('svg', {class: 'flow-wires', 'aria-hidden': 'true'}); diagram.append(wires);
      const columns = el('div', undefined, 'flow-columns');
      for (const section of flow.sections) {
        const frame = el('section', undefined, 'flow-stage flow-stage-' + section.id);
        frame.id = 'protocol-stage-' + section.id;
        const header = el('header'); header.append(el('h2', section.label), el('p', section.note)); frame.append(header);
        for (const row of section.rows) {
          const line = el('div', undefined, 'flow-row' + (row.length > 1 ? ' flow-parallel' : ''));
          row.forEach(key => line.append(card(key))); frame.append(line);
        }
        columns.append(frame);
      }
      diagram.append(columns);
      const circuit = el('section', undefined, 'flow-circuit');
      const header = el('header'); header.append(el('h2', flow.circuit.label), el('p', flow.circuit.note));
      circuit.append(header);
      const circuitRows = el('div', undefined, 'flow-circuit-grid');
      for (const id of flow.circuit.items) circuitRows.append(card(id));
      circuit.append(circuitRows); diagram.append(circuit);
      diagram.append(el('p', flow.circuit.base_note, 'flow-base-note'));
      scroll.append(diagram); container.append(scroll);
      container.append(el('p', 'A fold produces claims. PiDEC checks public digits and recombination; later folds or terminal opening checks establish private validity. Formulas and exact evidence are inside each box.', 'flow-reading-note'));
      const support = el('section', undefined, 'flow-support');
      support.append(el('h2', 'Proofs and implementation links'));
      const grid = el('div', undefined, 'flow-support-grid');
      for (const key of flow.support) grid.append(card(key));
      support.append(grid, el('p', 'Each requirement is counted once in this view. Shared results remain linked from the operations that use them. Counts keep their local scopes; a Rust count includes code that has not had a complete execution check.'));
      container.append(support);
      const conditions = el('section', undefined, 'flow-conditions'); conditions.append(el('h2', 'Explicit conditions'));
      for (const condition of flow.conditions) {
        const row = el('div'); row.append(premiseLink(condition.id), el('p', condition.detail)); conditions.append(row);
      }
      container.append(conditions);
      dialog = el('dialog', undefined, 'flow-dialog'); dialog.setAttribute('aria-labelledby', 'flow-detail-title');
      const close = el('button', 'Close', 'copy-link flow-close'); close.type = 'button'; close.addEventListener('click', () => dialog.close());
      detail = el('div', undefined, 'flow-detail'); dialog.append(close, detail); container.append(dialog);
      dialog.addEventListener('close', () => { if (location.hash.startsWith('#protocol-flow:')) location.hash = 'protocol-flow'; });
      observer = new ResizeObserver(draw); observer.observe(diagram);
      for (const node of cards.values()) observer.observe(node);
      built = true;
    }
    for (const card of container.querySelectorAll('[data-flow-item]')) {
      if (card.dataset.flowItem === selected) card.setAttribute('aria-current', 'true');
      else card.removeAttribute('aria-current');
    }
    if (items.has(selected)) inspect(items.get(selected));
    else if (dialog.open) dialog.close();
    requestAnimationFrame(draw);
  }
  return {render, close() { if (dialog?.open) dialog.close(); }};
}

if (typeof module !== 'undefined') module.exports = {protocolFlowPath};
