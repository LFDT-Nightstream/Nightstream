// Uses only recorded requirement edges. Subject groups do not create dependencies.
function buildProofGraph(nodes, scopeId, includeExternal = false) {
  const byId = new Map(nodes.map(node => [node.id, node]));
  const members = new Set(nodes.filter(node => {
    if (node.kind !== 'leaf') return false;
    for (let owner = node; owner; owner = byId.get(owner.parent)) if (owner.id === scopeId) return true;
    return false;
  }).map(node => node.id));
  const boundary = nodes.flatMap(node => (node.depends_on || []).map(source => ({source, target: node.id})))
    .filter(edge => members.has(edge.source) || members.has(edge.target));
  const inputs = new Set(boundary.filter(edge => !members.has(edge.source)).map(edge => edge.source));
  const outputs = new Set(boundary.filter(edge => !members.has(edge.target)).map(edge => edge.target));
  const edges = boundary.filter(edge => includeExternal || (members.has(edge.source) && members.has(edge.target)));
  const visible = new Set([...members, ...edges.flatMap(edge => [edge.source, edge.target])]);
  const usedBy = new Map([...visible].map(id => [id, []]));
  const uses = new Map([...visible].map(id => [id, []]));
  for (const edge of edges) {
    usedBy.get(edge.source).push(edge.target);
    uses.get(edge.target).push(edge.source);
  }
  const connected = [...visible].filter(id => usedBy.get(id).length || uses.get(id).length);
  const isolated = [...visible].filter(id => !usedBy.get(id).length && !uses.get(id).length);
  // Collapse mutual references for layout only; retain every node and edge.
  const indices = new Map(), low = new Map(), stack = [], onStack = new Set(), components = [];
  function visit(id) {
    indices.set(id, indices.size);
    low.set(id, indices.get(id));
    stack.push(id); onStack.add(id);
    for (const target of usedBy.get(id)) {
      if (!indices.has(target)) { visit(target); low.set(id, Math.min(low.get(id), low.get(target))); }
      else if (onStack.has(target)) low.set(id, Math.min(low.get(id), indices.get(target)));
    }
    if (low.get(id) === indices.get(id)) {
      const component = [];
      let member;
      do { member = stack.pop(); onStack.delete(member); component.push(member); } while (member !== id);
      components.push(component);
    }
  }
  connected.forEach(id => { if (!indices.has(id)) visit(id); });
  const componentOf = new Map(components.flatMap((ids, index) => ids.map(id => [id, index])));
  const levels = new Map();
  function level(index) {
    if (levels.has(index)) return levels.get(index);
    const consumers = new Set(components[index].flatMap(id => usedBy.get(id)).map(id => componentOf.get(id)));
    consumers.delete(index);
    const result = consumers.size ? 1 + Math.max(...[...consumers].map(level)) : 0;
    levels.set(index, result);
    return result;
  }
  const rank = new Map(connected.map(id => [id, level(componentOf.get(id))]));
  const rows = Array.from({length: connected.length ? Math.max(...rank.values()) + 1 : 0}, () => []);
  connected.forEach(id => rows[rank.get(id)].push(id));
  const positions = new Map(), width = Math.max(0, ...rows.map(row => row.length));
  for (const row of rows) {
    const center = id => {
      const above = usedBy.get(id).filter(target => positions.has(target));
      return above.length ? above.reduce((sum, target) => sum + positions.get(target), 0) / above.length : width / 2;
    };
    row.sort((a, b) => center(a) - center(b));
    row.forEach((id, index) => positions.set(id, (width - row.length) / 2 + index));
  }
  const cycles = components.filter(ids => ids.length > 1 || usedBy.get(ids[0]).includes(ids[0]));
  return {members, inputs, outputs, edges, visible, usedBy, uses, rows, rank, isolated, cycles};
}

function proofTrace(model, selected) {
  const walk = adjacency => {
    const found = new Set([selected]);
    const pending = [selected];
    while (pending.length) for (const id of adjacency.get(pending.pop()) || []) {
      if (!found.has(id)) { found.add(id); pending.push(id); }
    }
    return found;
  };
  return {needed: walk(model.uses), consumers: walk(model.usedBy)};
}

function createProofGraph({nodes, byId, navNames, el, names, premiseLink, sourceUrl}) {
  let observer, frame;
  function destroy() { observer?.disconnect(); cancelAnimationFrame(frame); }
  function ownerName(id) {
    let owner = byId.get(id);
    while (owner.parent && owner.parent !== 'root') owner = byId.get(owner.parent);
    return navNames[owner.id] || owner.label;
  }
  const short = {proved: 'proved', connected: 'connected', assumption: 'assumed', not_required: 'n/a', not_reviewed: 'unreviewed'};
  function statuses(node) {
    const result = el('span', undefined, 'proof-node-status');
    for (const [key, label] of [['proof', 'Proof'], ['connection', 'Link']]) {
      const badge = el('span', undefined, 'axis-count axis-' + key);
      badge.append(el('span', label), el('strong', short[node[key]] || node[key]));
      badge.setAttribute('aria-label', label + ': ' + names[node[key]]);
      result.append(badge);
    }
    return result;
  }
  function render(container, scopeId, initialSelection) {
    destroy();
    const scope = byId.get(scopeId);
    let model = buildProofGraph(nodes, scopeId);
    let selected = initialSelection || null, zoom = 1;
    const controls = el('div', undefined, 'proof-controls');
    const choice = el('select');
    choice.setAttribute('aria-label', 'Find a requirement in this proof graph');
    const external = el('input');
    external.type = 'checkbox';
    external.checked = !!selected && !model.members.has(selected);
    const externalLabel = el('label', undefined, 'proof-external-toggle');
    externalLabel.append(external, el('span', 'Include outside requirements'));
    const fit = el('button', 'Fit graph', 'copy-link'), actual = el('button', '100%', 'copy-link');
    const clear = el('button', 'Clear highlight', 'copy-link');
    for (const button of [fit, actual, clear]) button.type = 'button';
    controls.append(choice, externalLabel, fit, actual, clear);
    const stats = el('p', undefined, 'tech-meta');
    stats.setAttribute('aria-live', 'polite');
    const intro = el('p', 'Each box is one requirement. Proof is its local result; Link is its connection to the required consumer. An arrow means the upper requirement uses the lower one. The side panel includes scope notes for those connections. Drag or scroll to move; select a box to trace its chain.', 'tech-note');
    const workspace = el('div', undefined, 'proof-workspace');
    const viewport = el('div', undefined, 'proof-viewport');
    viewport.tabIndex = 0;
    viewport.setAttribute('aria-label', 'Proof dependency graph. Drag or scroll to move.');
    const stage = el('div', undefined, 'proof-stage'), canvas = el('div', undefined, 'proof-canvas');
    stage.append(canvas); viewport.append(stage);
    const inspector = el('aside', undefined, 'proof-inspector');
    inspector.setAttribute('aria-label', 'Selected requirement and proof evidence');
    const separate = el('details', undefined, 'proof-unconnected');
    workspace.append(viewport, inspector);
    container.append(intro, stats, controls, workspace, separate);
    let cards = new Map(), lines = [], svg;
    function svgNode(tag, attrs) {
      const item = document.createElementNS('http://www.w3.org/2000/svg', tag);
      for (const [key, value] of Object.entries(attrs)) item.setAttribute(key, value);
      return item;
    }
    function nodeCard(id) {
      const node = byId.get(id);
      const button = el('button', undefined, 'proof-node');
      button.type = 'button';
      const outside = !model.members.has(id);
      button.classList.toggle('proof-outside', outside);
      button.append(el('span', outside ? 'Outside · ' + ownerName(id) : byId.get(node.parent).label, 'tech-meta'));
      button.append(el('strong', node.label), statuses(node));
      button.addEventListener('click', () => select(id));
      cards.set(id, button);
      return button;
    }
    function reference(id) {
      const button = el('button', byId.get(id).label, 'proof-reference');
      button.type = 'button';
      button.addEventListener('click', () => {
        if (model.visible.has(id)) select(id, true);
        else if (model.inputs.has(id) || model.outputs.has(id)) {
          external.checked = true; selected = id; rebuild(); select(id, true);
        } else {
          let owner = byId.get(id);
          while (owner.parent && owner.parent !== 'root') owner = byId.get(owner.parent);
          location.hash = 'proof-' + owner.id + ':' + id;
        }
      });
      return button;
    }
    function describe() {
      inspector.replaceChildren();
      if (!selected) {
        inspector.append(el('h3', 'Follow a proof'), el('p', 'Select a box to see the facts it needs, the results that use it, and the cited theorems.'));
        inspector.append(el('p', 'Missing edges mean no dependency was recorded. They do not prove that a requirement is independent.', 'tech-meta'));
        return;
      }
      const node = byId.get(selected);
      inspector.append(el('span', ownerName(selected), 'tech-meta'), el('h3', node.label), statuses(node), el('p', node.requirement));
      if (node.scope?.length) inspector.append(el('p', 'Scope: ' + node.scope.join(' · ').replaceAll('_', ' '), 'tech-meta'));
      if (node.assumption_ids?.length && premiseLink) {
        inspector.append(el('h4', 'Explicit assumptions'));
        for (const id of node.assumption_ids) inspector.append(premiseLink(id), el('br'));
      }
      const requirementLink = el('a', 'Full requirement & evidence');
      requirementLink.href = '#req-' + selected;
      inspector.append(requirementLink);
      for (const [label, ids] of [
        ['Needs', node.depends_on || []],
        ['Used by', nodes.filter(other => (other.depends_on || []).includes(selected)).map(other => other.id)]
      ]) {
        inspector.append(el('h4', label + ' (' + ids.length + ')'));
        if (!ids.length) inspector.append(el('p', 'None recorded.', 'tech-meta'));
        for (const id of ids) {
          inspector.append(reference(id));
          const note = label === 'Needs' ? node.dependency_notes?.[id] : byId.get(id).dependency_notes?.[selected];
          if (note) inspector.append(el('p', note, 'tech-meta'));
        }
      }
      const evidence = el('details', undefined, 'proof-evidence');
      evidence.append(el('summary', 'Scope & theorem evidence'));
      if (node.remaining) evidence.append(el('p', node.remaining));
      evidence.append(el('p', 'Rust: ' + names[node.rust], 'tech-meta'));
      for (const item of node.code || []) {
        const url = sourceUrl?.(item.path, item.line);
        const location = el(url ? 'a' : 'div', item.path + ':' + item.line, 'source-path');
        if (url) location.href = url;
        evidence.append(el('div', item.symbol || item.path, 'source-name'), location);
      }
      for (const item of node.paper || []) evidence.append(el('p', (item.section || '') + ' · ' + item.path + ':' + item.line, 'source-path'));
      inspector.append(evidence);
    }
    function highlight() {
      const trace = selected ? proofTrace(model, selected) : null;
      for (const [id, card] of cards) {
        card.setAttribute('aria-pressed', String(id === selected));
        card.classList.toggle('proof-muted', !!trace && !trace.needed.has(id) && !trace.consumers.has(id));
        card.classList.toggle('proof-needed', !!trace && id !== selected && trace.needed.has(id));
        card.classList.toggle('proof-consumer', !!trace && id !== selected && trace.consumers.has(id));
      }
      for (const {edge, line} of lines) {
        const active = trace && ((trace.needed.has(edge.source) && trace.needed.has(edge.target)) ||
          (trace.consumers.has(edge.source) && trace.consumers.has(edge.target)));
        line.classList.toggle('on-path', !!active);
        line.classList.toggle('proof-muted', !!trace && !active);
      }
    }
    function centerCard(id) {
      const card = cards.get(id);
      if (!card) return;
      if (!canvas.contains(card)) { separate.open = true; card.scrollIntoView({block: 'nearest'}); return; }
      const rect = card.getBoundingClientRect(), view = viewport.getBoundingClientRect();
      viewport.scrollLeft += rect.left + rect.width / 2 - view.left - viewport.clientWidth / 2;
      viewport.scrollTop += rect.top + rect.height / 2 - view.top - viewport.clientHeight / 2;
    }
    function select(id, center = false) {
      selected = id;
      choice.value = id || '';
      history.replaceState(null, '', '#proof-' + scopeId + (id ? ':' + id : ''));
      describe(); highlight();
      if (center && id) centerCard(id);
    }
    function draw() {
      stage.style.width = canvas.offsetWidth * zoom + 'px';
      stage.style.height = canvas.offsetHeight * zoom + 'px';
      const bounds = canvas.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return;
      svg.setAttribute('viewBox', '0 0 ' + bounds.width + ' ' + bounds.height);
      for (const {edge, line} of lines) {
        const source = cards.get(edge.source).getBoundingClientRect(), target = cards.get(edge.target).getBoundingClientRect();
        const x1 = source.left + source.width / 2 - bounds.left, y1 = source.top - bounds.top;
        const x2 = target.left + target.width / 2 - bounds.left, y2 = target.bottom - bounds.top;
        const mid = (y1 + y2) / 2;
        line.setAttribute('d', 'M ' + x1 + ' ' + y1 + ' C ' + x1 + ' ' + mid + ', ' + x2 + ' ' + mid + ', ' + x2 + ' ' + (y2 + 2));
      }
    }
    function setZoom(value) {
      const centerX = (viewport.scrollLeft + viewport.clientWidth / 2) / zoom;
      const centerY = (viewport.scrollTop + viewport.clientHeight / 2) / zoom;
      zoom = value;
      canvas.style.transform = 'scale(' + zoom + ')';
      draw();
      viewport.scrollLeft = centerX * zoom - viewport.clientWidth / 2;
      viewport.scrollTop = centerY * zoom - viewport.clientHeight / 2;
    }
    function rebuild() {
      observer?.disconnect(); cancelAnimationFrame(frame);
      model = buildProofGraph(nodes, scopeId, external.checked);
      if (selected && !model.visible.has(selected)) selected = null;
      stats.textContent = model.members.size + ' requirements in ' + (navNames[scopeId] || scope.label) + ' · ' +
        model.edges.length + ' connections shown · ' + model.inputs.size + ' outside inputs · ' + model.outputs.size + ' outside consumers';
      if (model.cycles.length) stats.textContent += ' · Mutual references are kept on the same row.';
      choice.replaceChildren(el('option', 'Find a requirement…'));
      choice.firstChild.value = '';
      for (const id of model.visible) {
        const option = el('option', byId.get(id).label + (!model.members.has(id) ? ' — ' + ownerName(id) : ''));
        option.value = id; choice.append(option);
      }
      choice.value = selected || '';
      canvas.replaceChildren(); cards = new Map(); lines = [];
      svg = svgNode('svg', {class: 'proof-wires', 'aria-hidden': 'true'});
      const defs = svgNode('defs', {}), marker = svgNode('marker', {id: 'proof-arrow', viewBox: '0 0 10 10', refX: '9', refY: '5', markerWidth: '5', markerHeight: '5', orient: 'auto'});
      marker.append(svgNode('path', {d: 'M 0 0 L 10 5 L 0 10 z', fill: 'var(--accent)'}));
      defs.append(marker); svg.append(defs); canvas.append(svg);
      for (const row of model.rows) {
        const layer = el('div', undefined, 'proof-layer');
        for (const id of row) layer.append(nodeCard(id));
        canvas.append(layer);
      }
      for (const edge of model.edges) {
        const line = svgNode('path', {class: 'proof-wire' + (!model.members.has(edge.source) || !model.members.has(edge.target) ? ' proof-boundary-wire' : ''), 'marker-end': 'url(#proof-arrow)'});
        svg.append(line); lines.push({edge, line});
      }
      viewport.hidden = model.rows.length === 0;
      workspace.classList.toggle('proof-empty', viewport.hidden);
      fit.disabled = actual.disabled = viewport.hidden;
      separate.replaceChildren(el('summary', model.isolated.length + ' requirements with no connections shown'));
      separate.hidden = model.isolated.length === 0;
      separate.open = viewport.hidden;
      separate.append(el('p', external.checked ? 'No dependencies are recorded for these items.' : 'These items have no connections inside this view. Some use outside requirements.', 'tech-meta'));
      const grid = el('div', undefined, 'proof-isolated-grid');
      for (const id of model.isolated) grid.append(nodeCard(id));
      separate.append(grid);
      describe(); highlight();
      observer = new ResizeObserver(draw); observer.observe(canvas);
      frame = requestAnimationFrame(() => {
        zoom = 1; canvas.style.transform = 'scale(1)'; draw();
        viewport.scrollLeft = (canvas.offsetWidth - viewport.clientWidth) / 2;
        viewport.scrollTop = 0;
        if (selected) centerCard(selected);
      });
    }
    let drag;
    viewport.addEventListener('pointerdown', event => {
      if (event.pointerType !== 'mouse' || event.button !== 0 || event.target.closest('button')) return;
      drag = {x: event.clientX, y: event.clientY, left: viewport.scrollLeft, top: viewport.scrollTop};
      viewport.setPointerCapture(event.pointerId); viewport.classList.add('dragging'); event.preventDefault();
    });
    viewport.addEventListener('pointermove', event => {
      if (!drag) return;
      viewport.scrollLeft = drag.left + drag.x - event.clientX;
      viewport.scrollTop = drag.top + drag.y - event.clientY;
    });
    viewport.addEventListener('lostpointercapture', () => { drag = null; viewport.classList.remove('dragging'); });
    viewport.addEventListener('pointerup', event => { if (viewport.hasPointerCapture(event.pointerId)) viewport.releasePointerCapture(event.pointerId); });
    external.addEventListener('change', () => { rebuild(); select(selected); });
    choice.addEventListener('change', () => select(choice.value || null, true));
    fit.addEventListener('click', () => { setZoom(Math.min(viewport.clientWidth / canvas.offsetWidth, viewport.clientHeight / canvas.offsetHeight)); viewport.scrollTo(0, 0); });
    actual.addEventListener('click', () => setZoom(1));
    clear.addEventListener('click', () => select(null));
    rebuild();
  }
  return {render, destroy};
}

if (typeof module !== 'undefined') module.exports = {buildProofGraph, proofTrace};
