// Main assembly from reading-guide.md; exact theorem references stay in the details.
const techAssembly = [
  {id: 'root', row: 0, column: 3, needs: ['P']},
  {id: 'P', row: 1, column: 3, needs: ['H', 'L'], caption: 'Accepted proofs, implementation evidence, and the production path'},
  {id: 'H', row: 2, column: 2, needs: ['N', 'T'], caption: 'Verify the complete chain of computation'},
  {id: 'L', row: 2, column: 4, needs: ['F', 'T'], caption: 'Connect the verifier to its circuit and exported package'},
  {id: 'N', row: 3, column: 2, needs: ['C', 'R', 'D', 'T'], caption: 'Combine the three phases into one folding result'},
  {id: 'C', row: 4, column: 1, needs: ['F'], caption: 'Check inputs and produce evaluation claims'},
  {id: 'R', row: 4, column: 3, needs: ['F'], caption: 'Combine the claims into one'},
  {id: 'D', row: 4, column: 5, needs: ['F'], caption: 'Split the witness into parts with small values'},
  {id: 'F', row: 5, column: 2, needs: [], caption: 'Fields, rings, bounds, commitments, and setup'},
  {id: 'T', row: 5, column: 4, needs: [], caption: 'Poseidon2, canonical words, and transcript primitives'}
];

// Saved requirement and group links open the same tree as the main navigation.
function techTreeRoute(fragment, byId) {
  if (fragment.startsWith('proof-')) {
    const parts = fragment.slice(6).split(':');
    const scope = byId.get(parts[0]);
    const item = byId.get(parts[1] || parts[0]);
    if (scope?.kind === 'group' && item && parts.length <= 2) {
      return {id: item.id, proofScopeId: scope.id};
    }
  }
  let id = fragment.startsWith('req-') ? fragment.slice(4) :
    fragment.startsWith('group-') ? fragment.slice(6) :
    fragment.startsWith('tech-') ? fragment.slice(5) : 'root';
  if (!byId.has(id)) id = 'root';
  const node = byId.get(id);
  if (fragment.startsWith('req-') || node.kind === 'leaf') {
    let owner = node;
    while (owner.parent && owner.parent !== 'root') owner = byId.get(owner.parent);
    return {id, proofScopeId: owner.id, recordId: id};
  }
  return {id};
}

function createTechTree({nodes, byId, children, navNames, el, countBadges, names, premiseLink, sourceUrl, renderRecord, scopeControls}) {
  let observer;
  let frame;
  const proofGraph = createProofGraph({nodes, byId, navNames, el, names, premiseLink, sourceUrl});
  const assemblyById = new Map(techAssembly.map(node => [node.id, node]));
  function link(id, label, className) {
    const anchor = el('a', label || navNames[id] || byId.get(id).label, className);
    anchor.href = id === 'root' ? '#tech-tree' : '#tech-' + id;
    return anchor;
  }
  function svgElement(tag, attributes) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    return node;
  }
  return function renderTechTree(container, id, proofScopeId) {
    observer?.disconnect();
    cancelAnimationFrame(frame);
    proofGraph.destroy();
    container.replaceChildren();
    const selected = byId.get(id);
    let owner = byId.get(proofScopeId || id);
    while (owner.parent && owner.parent !== 'root') owner = byId.get(owner.parent);
    const active = new Set([owner.id]);
    function markParents(child) {
      for (const node of techAssembly) {
        if (node.needs.includes(child) && !active.has(node.id)) {
          active.add(node.id);
          markParents(node.id);
        }
      }
    }
    markParents(owner.id);
    const toolbar = el('div', undefined, 'tech-toolbar');
    toolbar.append(el('p', 'Read down to see what each result needs. Lines point up toward the result they support.', 'tech-meta'));
    const groupLabel = el('label', 'Group ');
    const select = el('select');
    for (const group of [byId.get('root'), ...children.get('root')]) {
      const option = el('option', group.id === 'root' ? 'Full tree' : navNames[group.id]);
      option.value = group.id;
      select.append(option);
    }
    select.value = owner.id;
    select.addEventListener('change', () => { location.hash = select.value === 'root' ? 'tech-tree' : 'proof-' + select.value; });
    groupLabel.append(select);
    toolbar.append(groupLabel);
    if (owner.id !== 'root') {
      toolbar.append(link('root', 'Full tree'));
    }
    container.append(toolbar);
    const scroll = el('div', undefined, 'tech-canvas-scroll');
    scroll.tabIndex = 0;
    scroll.setAttribute('aria-label', 'Complete Stage 1 tree. Scroll horizontally on a small screen.');
    const canvas = el('div', undefined, 'tech-canvas');
    const svg = svgElement('svg', {'aria-hidden': 'true', class: 'tech-wires'});
    const definitions = svgElement('defs', {});
    const marker = svgElement('marker', {id: 'tree-arrow', viewBox: '0 0 10 10', refX: '9', refY: '5', markerWidth: '6', markerHeight: '6', orient: 'auto-start-reverse'});
    marker.append(svgElement('path', {d: 'M 0 0 L 10 5 L 0 10 z', fill: 'var(--accent)'}));
    definitions.append(marker);
    svg.append(definitions);
    canvas.append(svg);
    const cards = new Map();
    const rowCount = Math.max(...techAssembly.map(node => node.row)) + 1;
    for (let index = 0; index < rowCount; index++) {
      const row = el('div', undefined, 'tech-assembly-row');
      for (const entry of techAssembly.filter(node => node.row === index)) {
        const card = link(entry.id, '', 'tech-assembly-card');
        if (entry.id !== 'root') card.href = '#proof-' + entry.id;
        card.replaceChildren(el('strong', navNames[entry.id] || byId.get(entry.id).label));
        card.style.gridColumn = entry.column + ' / span 2';
        card.dataset.node = entry.id;
        if (active.has(entry.id)) card.classList.add('on-path');
        if (entry.id === owner.id) card.setAttribute('aria-current', 'location');
        if (entry.caption) card.append(el('span', entry.caption, 'tech-meta'));
        card.append(countBadges(entry.id));
        cards.set(entry.id, card);
        row.append(card);
      }
      canvas.append(row);
    }
    scroll.append(canvas);
    container.append(scroll);
    container.append(el('p', 'Dashed lines show shared primitives. This is the main assembly of Stage 1; counts retain their local scopes. Individual proof references are in the requirement details.', 'tech-note'));
    const detailSection = el('section', undefined, 'tech-detail-section');
    const graphScope = byId.get(proofScopeId || (selected.kind === 'group' ? id : owner.id));
    const heading = el('h2', owner.id === 'root' ? 'Requirements and evidence' : (navNames[graphScope.id] || graphScope.label) + ' — proof connections');
    heading.id = 'tech-detail-title';
    heading.tabIndex = -1;
    detailSection.append(heading);
    if (owner.id !== 'root') {
      const overview = link('root', 'Back to the full Stage 1 tree ↑', 'tech-record-link');
      detailSection.append(overview);
      const graph = el('div');
      detailSection.append(graph);
      proofGraph.render(graph, graphScope.id, selected.kind === 'leaf' ? selected.id : null);
    }
    const records = el('details', undefined, 'proof-records');
    records.append(el('summary', 'Requirements, status and evidence'));
    scopeControls.hidden = false;
    records.append(scopeControls);
    const list = el('ul', undefined, 'req-roots');
    const record = renderRecord(graphScope);
    record.querySelector('details').open = true;
    list.append(record);
    records.append(list);
    detailSection.append(records);
    container.append(detailSection);
    function draw() {
      svg.querySelectorAll(':scope > path').forEach(path => path.remove());
      const bounds = canvas.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return;
      svg.setAttribute('viewBox', '0 0 ' + bounds.width + ' ' + bounds.height);
      for (const parent of techAssembly) {
        const target = cards.get(parent.id).getBoundingClientRect();
        for (const childId of parent.needs) {
          const child = assemblyById.get(childId);
          const source = cards.get(childId).getBoundingClientRect();
          const x1 = source.left + source.width / 2 - bounds.left;
          const y1 = source.top - bounds.top;
          const x2 = target.left + target.width / 2 - bounds.left;
          const y2 = target.bottom - bounds.top;
          const shared = child.row - parent.row > 1;
          let d;
          if (shared) {
            const rail = childId === 'F' ? 6 : bounds.width - 6;
            const sourceX = (childId === 'F' ? source.left : source.right) - bounds.left;
            const sourceY = source.top + source.height / 2 - bounds.top;
            const gap = parseFloat(getComputedStyle(canvas).rowGap);
            const turnY = y2 + gap / 2;
            d = `M ${sourceX} ${sourceY} H ${rail} V ${turnY} H ${x2} V ${y2 + 2}`;
          } else {
            const mid = (y1 + y2) / 2;
            d = `M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2 + 2}`;
          }
          const highlighted = active.has(parent.id) && active.has(childId);
          svg.append(svgElement('path', {d, class: 'tech-wire' + (shared ? ' shared-wire' : '') + (highlighted ? ' on-path' : ''), 'marker-end': 'url(#tree-arrow)'}));
        }
      }
    }
    observer = new ResizeObserver(draw);
    observer.observe(canvas);
    for (const card of cards.values()) observer.observe(card);
    frame = requestAnimationFrame(draw);
  };
}

if (typeof module !== 'undefined') module.exports = {techAssembly, techTreeRoute};
