function proofMapEdges(diagram, selected, all) {
  return diagram.edges.filter(edge => selected
    ? edge.source === selected || edge.target === selected
    : all || edge.overview);
}

function createProofStructureMap({diagram, byId, el, sourceUrl, premiseLink}) {
  const kinds = {theorem: 'Proved', premise: 'Premise', assumption: 'Assumed', definition: 'Definition', open: 'Open'};
  const relations = {uses: 'Uses result', supplies: 'Supplies premise', requires: 'Requires premise', open: 'Open connection'};
  const nodes = new Map(diagram.nodes.map(node => [node.id, node]));
  let resizeObserver;
  const destroy = () => resizeObserver?.disconnect();
  function svgElement(tag, attrs = {}) {
    const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
    return element;
  }
  function render(container, initialSelection) {
    destroy(); container.replaceChildren();
    let selected = nodes.has(initialSelection) ? initialSelection : null, filter = 'all';
    const controls = el('div', undefined, 'structure-controls');
    const filters = el('div', undefined, 'structure-filters');
    const filterButtons = new Map();
    for (const [kind, label] of [['all', 'All'], ...Object.entries(kinds)]) {
      const button = el('button', label, 'structure-filter kind-' + kind);
      button.type = 'button'; button.setAttribute('aria-pressed', String(kind === 'all'));
      button.addEventListener('click', () => { filter = kind; highlight(); });
      filterButtons.set(kind, button); filters.append(button);
    }
    const allLabel = el('label', undefined, 'structure-all');
    const all = el('input'); all.type = 'checkbox';
    all.addEventListener('change', highlight);
    allLabel.append(all, 'All connections');
    const clear = el('button', 'Clear selection', 'copy-link'); clear.type = 'button';
    clear.addEventListener('click', () => select(null));
    controls.append(filters, allLabel, clear);

    const board = el('div', undefined, 'structure-board');
    const main = el('div', undefined, 'structure-main'), side = el('div', undefined, 'structure-side');
    const wires = svgElement('svg', {class: 'structure-wires', 'aria-hidden': 'true'});
    board.append(wires, main, side);
    const cards = new Map();
    for (const layer of diagram.layers) {
      const group = el('section', undefined, 'structure-group group-' + layer.id);
      group.append(el('h2', layer.title));
      for (const row of layer.rows) {
        const rowElement = el('div', undefined, 'structure-row');
        for (const id of row) {
          const node = nodes.get(id), label = node.label.join(' ');
          const card = el('button', undefined, 'structure-node kind-' + node.kind);
          card.type = 'button'; card.dataset.mapNode = id;
          card.setAttribute('aria-label', label + ', ' + kinds[node.kind]);
          card.setAttribute('aria-pressed', 'false');
          card.title = kinds[node.kind] + ': ' + node.summary;
          card.append(el('span', undefined, 'structure-dot'), el('span', label));
          card.addEventListener('click', () => select(id));
          cards.set(id, card); rowElement.append(card);
        }
        group.append(rowElement);
      }
      (layer.column === 'main' ? main : side).append(group);
    }

    const inspector = el('aside', undefined, 'structure-inspector');
    inspector.setAttribute('aria-label', 'Selected proof');
    inspector.setAttribute('aria-live', 'polite');
    inspector.tabIndex = -1;
    const note = el('p', 'Main connections shown. Dashed arrows mark premises or open work. All results retain their stated conditions.', 'structure-note');
    const download = el('a', 'Map and evidence in Markdown'); download.href = 'proof-map.md';
    container.append(controls, board, inspector, note, download);
    for (const surface of [board, inspector]) surface.addEventListener('keydown', event => {
      if (event.key === 'Escape' && selected) { const previous = selected; select(null); cards.get(previous).focus(); }
    });

    function draw() {
      const rect = board.getBoundingClientRect();
      if (!rect.width) return;
      wires.setAttribute('viewBox', `0 0 ${rect.width} ${rect.height}`);
      wires.replaceChildren();
      const defs = svgElement('defs');
      for (const kind of Object.keys(relations)) {
        const marker = svgElement('marker', {id: 'structure-arrow-' + kind, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto'});
        marker.append(svgElement('path', {d: 'M0 0 L10 5 L0 10 Z', class: 'arrow-' + kind})); defs.append(marker);
      }
      wires.append(defs);
      for (const edge of proofMapEdges(diagram, selected, all.checked)) {
        const a = cards.get(edge.source).getBoundingClientRect(), b = cards.get(edge.target).getBoundingClientRect();
        const sameRow = Math.abs(a.top - b.top) < 1;
        const acrossColumns = cards.get(edge.source).closest('.structure-side') !== cards.get(edge.target).closest('.structure-side');
        let x1 = a.left + a.width / 2 - rect.left, y1 = a.top + a.height / 2 - rect.top;
        let x2 = b.left + b.width / 2 - rect.left, y2 = b.top + b.height / 2 - rect.top;
        let path;
        if (sameRow || acrossColumns) {
          const sign = x2 > x1 ? 1 : -1;
          x1 += sign * a.width / 2; x2 -= sign * b.width / 2;
          const mid = (x1 + x2) / 2;
          path = `M${x1} ${y1} C${mid} ${y1},${mid} ${y2},${x2} ${y2}`;
        } else {
          const sign = y2 > y1 ? 1 : -1;
          y1 += sign * a.height / 2; y2 -= sign * b.height / 2;
          const mid = (y1 + y2) / 2;
          path = `M${x1} ${y1} C${x1} ${mid},${x2} ${mid},${x2} ${y2}`;
        }
        const filtered = filter !== 'all' && nodes.get(edge.source).kind !== filter && nodes.get(edge.target).kind !== filter;
        wires.append(svgElement('path', {d: path, class: 'structure-edge edge-' + edge.kind + (filtered ? ' filtered' : ''), 'marker-end': 'url(#structure-arrow-' + edge.kind + ')'}));
      }
    }
    function highlight() {
      const edges = proofMapEdges(diagram, selected, true);
      const related = new Set(edges.flatMap(edge => [edge.source, edge.target]));
      for (const [kind, button] of filterButtons) button.setAttribute('aria-pressed', String(filter === kind));
      for (const node of diagram.nodes) {
        const card = cards.get(node.id);
        card.classList.toggle('filtered', filter !== 'all' && filter !== node.kind);
        card.classList.toggle('muted', !!selected && !related.has(node.id));
        card.classList.toggle('related', !!selected && related.has(node.id));
        card.setAttribute('aria-pressed', String(node.id === selected));
      }
      clear.hidden = !selected; all.disabled = !!selected;
      draw();
    }
    function sourceLinks(parent, refs) {
      for (const ref of refs) {
        const link = el('a', ref.symbol, 'structure-source'); link.href = sourceUrl(ref.path, ref.line);
        parent.append(link, el('span', ref.path.split('/').at(-1) + ':' + ref.line, 'source-path'));
      }
    }
    function describe() {
      inspector.hidden = !selected; inspector.replaceChildren();
      if (!selected) return;
      const node = nodes.get(selected), record = byId.get(node.record);
      const close = el('button', 'Close ×', 'copy-link structure-close'); close.type = 'button';
      close.addEventListener('click', () => { const previous = selected; select(null); cards.get(previous).focus(); });
      inspector.append(close, el('span', kinds[node.kind], 'structure-detail-kind kind-' + node.kind), el('h2', node.label.join(' ')), el('p', node.summary));
      for (const incoming of [true, false]) {
        const edges = diagram.edges.filter(edge => (incoming ? edge.target : edge.source) === selected);
        if (!edges.length) continue;
        inspector.append(el('h3', incoming ? 'Uses' : 'Used by'));
        const list = el('ul', undefined, 'structure-links');
        for (const edge of edges) {
          const other = nodes.get(incoming ? edge.source : edge.target);
          const item = el('li');
          const button = el('button', undefined, 'structure-reference kind-' + other.kind); button.type = 'button';
          button.append(el('span', undefined, 'structure-dot'), el('span', other.label.join(' ')));
          button.addEventListener('click', () => { filter = 'all'; select(other.id); cards.get(other.id).scrollIntoView({block: 'center'}); });
          const detail = el('details', undefined, 'structure-edge-detail');
          detail.append(el('summary', relations[edge.kind]), el('p', edge.note));
          sourceLinks(detail, edge.code); item.append(button, detail); list.append(item);
        }
        inspector.append(list);
      }
      const evidence = el('details', undefined, 'structure-evidence');
      evidence.append(el('summary', 'Statement and Lean evidence'), el('p', node.role));
      if (node.kind === 'theorem' && record.proof !== 'proved') evidence.append(el('p', 'This declaration is proved under conditions. The linked requirement retains its broader assumption or scope.'));
      const recordLink = el('a', 'Open requirement'); recordLink.href = '#req-' + node.record;
      evidence.append(recordLink, el('p', `Proof: ${record.proof}. Link: ${record.connection}. Rust: ${record.rust}.`, 'tech-meta'));
      if (node.assumption) evidence.append(premiseLink(node.assumption));
      sourceLinks(evidence, node.code); inspector.append(evidence);
    }
    function select(id) {
      selected = id;
      history.replaceState(null, '', '#proof-map' + (id ? ':' + id : ''));
      describe(); highlight();
      if (selected) {
        if (matchMedia('(max-width: 950px)').matches) cards.get(selected).scrollIntoView({block: 'start'});
        inspector.scrollTop = 0; inspector.focus({preventScroll: true});
      }
    }
    describe(); highlight();
    resizeObserver = new ResizeObserver(draw); resizeObserver.observe(board);
  }
  return {render, destroy};
}
if (typeof module !== 'undefined') module.exports = {proofMapEdges};
