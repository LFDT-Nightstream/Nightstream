(() => {
  const data = JSON.parse(document.getElementById('requirements-data').textContent);
  const publication = JSON.parse(document.getElementById('publication-data').textContent);
  const references = JSON.parse(document.getElementById('reference-data').textContent);
  const flowData = JSON.parse(document.getElementById('protocol-flow-data').textContent);
  const replayView = document.getElementById('prover-replay');
  const assurance = RequirementAssurance.build(data, publication, references);
  const nodes = data.nodes;
  const byId = new Map(nodes.map(node => [node.id, node]));
  const children = new Map();
  for (const node of nodes) {
    if (node.id === 'root') continue;
    if (!children.has(node.parent)) children.set(node.parent, []);
    children.get(node.parent).push(node);
  }
  const names = {
    proved: 'local proof exists', partial: 'partial', definition: 'defined',
    assumption: 'explicit assumption', not_required: 'not required here',
    not_reviewed: 'not verified by this review', connected: 'connected locally',
    open: 'open', tested_scoped: 'scoped tests passed', recorded_only: 'recorded evidence only',
    implemented: 'code exists'
  };
  const origins = {
    paper: 'Paper requirement', profile: 'Nightstream profile',
    implementation: 'Implementation requirement', assumption: 'External assumption',
    out_of_scope: 'Outside selected Stage 1'
  };
  const navNames = {
    F: 'Foundations', T: 'Hash & transcript', C: 'PiCCS', R: 'PiRLC', D: 'PiDEC',
    N: 'NIFS composition', H: 'HyperNova recursion', L: 'Circuit & export',
    P: 'Production acceptance', O: 'Outside Stage 1'
  };
  const el = (tag, text, className) => {
    const result = document.createElement(tag);
    if (text !== undefined) result.textContent = text;
    if (className) result.className = className;
    return result;
  };
  const descendants = id => (children.get(id) || []).flatMap(node => node.kind === 'leaf' ? [node] : descendants(node.id));
  const unresolved = node => node.connection === 'open' || node.connection === 'partial' || node.proof === 'partial';
  const tone = status => ['open', 'partial'].includes(status) ? 'open' : status === 'assumption' ? 'assumption' : ['proved', 'connected', 'tested_scoped'].includes(status) ? 'good' : 'neutral';
  const axes = [{key: 'proof', label: 'Proof'}, {key: 'connection', label: 'Link'}, {key: 'rust', label: 'Rust'}];
  const progress = new Map(nodes.filter(node => node.kind === 'group').map(node => {
    return [node.id, axes.map(axis => ({...axis, counts: RequirementAssurance.counts(descendants(node.id), axis.key)}))];
  }));
  const breakdown = axis => RequirementAssurance.categories[axis.key].map(([key, label]) => label + ' ' + axis.counts[key]).join(' · ');
  function countBadges(id) {
    const counts = el('span', undefined, 'axis-counts');
    for (const axis of progress.get(id)) {
      const badge = el('span', undefined, 'axis-count axis-' + axis.key);
      const {finished, total} = RequirementAssurance.progress(axis.key, axis.counts);
      const fraction = total ? finished + '/' + total : 'n/a';
      badge.append(el('span', axis.label), el('strong', fraction));
      badge.title = axis.label + ': ' + fraction + ' · ' + breakdown(axis);
      badge.setAttribute('aria-label', badge.title);
      counts.append(badge);
    }
    return counts;
  }
  const stats = node => {
    const leaves = descendants(node.id);
    if (node.origin === 'out_of_scope') return leaves.length + ' scope exclusions';
    const open = leaves.filter(unresolved).length;
    const assumptions = leaves.filter(item => item.proof === 'assumption').length;
    return leaves.length + ' requirements' + (open ? ' · ' + open + ' unresolved proof/link ' + (open === 1 ? 'item' : 'items') : '') + (assumptions ? ' · ' + assumptions + ' explicit ' + (assumptions === 1 ? 'assumption' : 'assumptions') : '');
  };
  function pair(box, label, value, className = '') {
    if (!value) return;
    const dl = el('dl', undefined, 'req-pair ' + className);
    dl.append(el('dt', label), el('dd', value));
    box.append(dl);
  }
  function anchorActions(node) {
    const row = el('div', undefined, 'req-anchor-row');
    const anchor = el('a', '#req-' + node.id, 'req-anchor');
    anchor.href = '#req-' + node.id;
    anchor.setAttribute('aria-label', 'Permanent link to ' + node.label);
    const copy = el('button', 'Copy link', 'copy-link');
    copy.type = 'button';
    copy.setAttribute('aria-label', 'Copy link to ' + node.label);
    copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(anchor.href);
        copy.textContent = 'Copied';
        document.getElementById('view-announcement').textContent = 'Link copied for ' + node.label;
      } catch {
        let field = row.querySelector('input');
        if (!field) {
          field = el('input', undefined, 'copy-link-fallback');
          field.type = 'text';
          field.readOnly = true;
          field.setAttribute('aria-label', 'Select and copy this item link');
          row.append(field);
        }
        field.value = anchor.href;
        field.focus();
        field.select();
        document.getElementById('view-announcement').textContent = 'Select and copy the displayed link.';
      }
    });
    const techLink = el('a', 'Proof connections');
    let owner = node;
    while (owner.parent && owner.parent !== 'root') owner = byId.get(owner.parent);
    techLink.href = node.id === 'root' ? '#tech-tree' : node.kind === 'group' ?
      '#proof-' + node.id : '#proof-' + owner.id + ':' + node.id;
    row.append(el('span', 'Linked item', 'target-indicator'), anchor, copy, techLink);
    if (flowData.owners[node.id]) {
      const flowLink = el('a', 'View in protocol flow');
      flowLink.href = '#protocol-flow:' + flowData.owners[node.id]; row.append(flowLink);
    }
    return row;
  }
  function render(node) {
    const li = el('li');
    li.dataset.record = node.id;
    const details = el('details');
    details.id = 'req-' + node.id;
    const summary = el('summary');
    summary.append(el('span', node.label, 'req-title'));
    if (node.kind === 'group' && node.origin !== 'out_of_scope') summary.append(countBadges(node.id));
    const state = el('span', undefined, 'req-state');
    if (node.kind === 'group') state.textContent = stats(node);
    else if (node.origin === 'out_of_scope') state.textContent = 'Outside selected Stage 1';
    else for (const [axis, label] of [['proof', 'Proof'], ['connection', 'Link'], ['rust', 'Rust']]) {
      state.append(el('span', label + ': ' + names[node[axis]], 'status status-' + tone(node[axis])));
    }
    summary.append(state);
    details.append(summary);
    const content = el('div', undefined, 'req-detail');
    content.append(anchorActions(node), el('span', origins[node.origin], 'req-id'));
    pair(content, 'Scope', node.scope.join(' · ').replaceAll('_', ' '));
    pair(content, 'Requirement', node.requirement);
    pair(content, 'Remaining', node.remaining, 'remaining' + (unresolved(node) ? ' remaining-open' : ''));
    const sources = [...(node.paper || []), ...(node.code || [])];
    if (sources.length) {
      const evidence = el('details', undefined, 'req-sources');
      evidence.append(el('summary', 'Paper & code references (' + sources.length + ')'));
      const refs = el('div', undefined, 'req-evidence');
      for (const source of sources) {
        refs.append(el('div', source.section || source.symbol || source.path, 'source-name'));
        const url = node.origin === 'out_of_scope' ? null : assurance.sourceUrl(source.path, source.line);
        const path = el(url ? 'a' : 'div', source.path + ':' + source.line, 'source-path');
        if (url) path.href = url;
        else path.append(el('span', node.origin === 'out_of_scope' ? ' (outside this reference check)' : ' (local paper corpus; hash in reference record)'));
        refs.append(path);
      }
      evidence.append(refs);
      content.append(evidence);
    }
    for (const [label, ids] of [
      ['Uses', node.depends_on || []],
      ['Used by', nodes.filter(other => other.depends_on?.includes(node.id)).map(other => other.id)]
    ]) {
      if (!ids.length) continue;
      const uses = el('div', undefined, 'uses');
      uses.append(el('strong', label));
      for (const id of ids) {
        const link = el('a', byId.get(id).label);
        link.href = '#req-' + id;
        uses.append(link);
      }
      content.append(uses);
    }
    if (node.assumption_ids?.length) {
      const premises = el('div', undefined, 'uses');
      premises.append(el('strong', 'Assumptions'));
      for (const id of node.assumption_ids) premises.append(assurance.premiseLink(id));
      content.append(premises);
    }
    for (const [id, note] of Object.entries(node.dependency_notes || {})) pair(content, 'Dependency ' + id, note);
    details.append(content);
    if (children.has(node.id)) {
      const list = el('ul', undefined, 'req-children');
      children.get(node.id).forEach(child => list.append(render(child)));
      details.append(list);
    }
    li.append(details);
    return li;
  }
  document.getElementById('source-meta').textContent = 'SuperNeo v1.2 + HyperNova · Code ' + data.provenance.code_commit.slice(0, 8) + ' · Map ' + (publication.map_commit?.slice(0, 8) || 'uncommitted preview') + ' · Production gates remain open.';
  const scope = document.getElementById('claim-scope');
  const viewNames = {readiness: 'Readiness', assumptions: 'Assumption ledger', risk: 'Error budget', evidence: 'Source and evidence'};
  for (const [id, view] of Object.entries(assurance.views)) {
    view.id = 'view-' + id;
    document.getElementById('assurance-views').append(view);
  }
  const techTree = document.getElementById('tech-tree');
  if (data.prover_replay) replayView.append(createReplayProgress({data, byId, el}));
  const renderTechTree = createTechTree({nodes, byId, children, navNames, el, names,
    countBadges, premiseLink: assurance.premiseLink, sourceUrl: assurance.sourceUrl,
    renderRecord: render, scopeControls: document.getElementById('scope-controls')});
  const techViewLink = document.getElementById('tech-view-link');
  const mapViewLink = document.getElementById('proof-map-view-link');
  const protocolViewLink = document.getElementById('protocol-view-link');
  const protocolView = document.getElementById('protocol-flow');
  const protocolFlow = createProtocolFlow({flow: flowData,
    byId, el, names, replay: data.prover_replay, sourceUrl: assurance.sourceUrl, premiseLink: assurance.premiseLink});
  const structureView = document.getElementById('proof-structure');
  const structureMap = createProofStructureMap({
    diagram: JSON.parse(document.getElementById('proof-map-data').textContent),
    byId, el, sourceUrl: assurance.sourceUrl, premiseLink: assurance.premiseLink});
  function selectView(id) {
    const tech = id === 'tech';
    document.body.classList.toggle('tech-mode', tech);
    document.body.classList.toggle('map-mode', id === 'map');
    document.body.classList.toggle('flow-mode', id === 'flow');
    replayView.hidden = !data.prover_replay || !['tech', 'flow'].includes(id);
    protocolView.hidden = id !== 'flow';
    if (id !== 'flow') protocolFlow.close();
    structureView.hidden = id !== 'map';
    if (id !== 'map') structureMap.destroy();
    techTree.hidden = !tech;
    for (const [view, link] of [['tech', techViewLink], ['map', mapViewLink], ['flow', protocolViewLink]]) {
      if (id === view) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }
    for (const [name, view] of Object.entries(assurance.views)) view.hidden = name !== id;
    for (const link of document.querySelectorAll('.review-nav a')) {
      if (link.hash === '#view-' + id) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }
  }
  function applyScope() {
    for (const item of techTree.querySelectorAll('li[data-record]')) {
      const node = byId.get(item.dataset.record);
      const matches = scope.value === 'all' || (node.kind === 'leaf' ? node.scope.includes(scope.value) : descendants(node.id).some(n => n.scope.includes(scope.value)));
      item.hidden = !matches;
    }
  }
  function showAssurance(id) {
    selectView(id);
    document.getElementById('view-title').textContent = viewNames[id];
    document.getElementById('view-progress').replaceChildren();
    document.getElementById('view-summary').textContent = '';
    document.getElementById('breadcrumb').textContent = 'Selected Stage 1 / ' + viewNames[id];
    document.getElementById('view-announcement').textContent = viewNames[id];
    document.title = viewNames[id] + ' — Nightstream';
  }
  function navigate() {
    const fragment = decodeURIComponent(location.hash.slice(1)) || 'protocol-flow';
    if (fragment === 'protocol-flow' || fragment.startsWith('protocol-flow:')) {
      selectView('flow');
      protocolFlow.render(protocolView, fragment.split(':')[1]);
      document.getElementById('view-title').textContent = flowData.title;
      document.getElementById('view-progress').replaceChildren();
      document.getElementById('breadcrumb').textContent = 'Selected Stage 1 / Protocol flow';
      document.getElementById('view-summary').textContent = flowData.description;
      document.getElementById('view-announcement').textContent = 'Protocol flow';
      document.title = 'Protocol flow — Nightstream';
    } else if (fragment === 'proof-map' || fragment.startsWith('proof-map:')) {
      selectView('map');
      structureMap.render(structureView, fragment.split(':')[1]);
      document.getElementById('view-title').textContent = 'Nightstream proof map';
      document.getElementById('view-progress').replaceChildren();
      document.getElementById('breadcrumb').textContent = 'Selected Stage 1 / Proof map';
      document.getElementById('view-summary').textContent = 'Read upward from verifier acceptance to the security goal. Select a result for its connections and Lean proof.';
      document.getElementById('view-announcement').textContent = 'Nightstream proof map';
      document.title = 'Proof structure — Nightstream';
      window.scrollTo({top: 0});
    } else if (fragment.startsWith('view-') && assurance.views[fragment.slice(5)]) {
      showAssurance(fragment.slice(5));
      window.scrollTo({top: 0});
    } else if (fragment.startsWith('assumption-') && document.getElementById(fragment)) {
      showAssurance('assumptions');
      const target = document.getElementById(fragment);
      target.open = true;
      target.querySelector('summary').focus({preventScroll: true});
      target.scrollIntoView({block: 'start'});
    } else {
      const {id, proofScopeId, recordId} = techTreeRoute(fragment, byId);
      const node = byId.get(proofScopeId || id);
      selectView('tech');
      renderTechTree(techTree, id, proofScopeId);
      if (recordId) scope.value = 'all';
      applyScope();
      document.getElementById('view-title').textContent = 'Tech tree';
      document.getElementById('view-progress').replaceChildren();
      document.getElementById('breadcrumb').textContent = 'Selected Stage 1 / Tech tree';
      document.getElementById('view-summary').textContent = 'The goal is at the top, with its supporting groups below. Select a group for its proof connections, requirements and evidence.';
      document.getElementById('view-announcement').textContent = 'Tech tree: ' + node.label;
      document.title = 'Tech tree: ' + node.label + ' — Nightstream';
      if (recordId) {
        const target = document.getElementById('req-' + recordId);
        let current = target;
        while (current && current !== techTree) {
          if (current.tagName === 'DETAILS') current.open = true;
          current = current.parentElement;
        }
        target.querySelector('.req-anchor').focus({preventScroll: true});
        requestAnimationFrame(() => target.scrollIntoView({block: 'start'}));
      } else {
        const heading = document.getElementById(id === 'root' ? 'view-title' : 'tech-detail-title');
        heading.tabIndex = -1;
        heading.focus({preventScroll: true});
        if (id === 'root') window.scrollTo({top: 0});
        else requestAnimationFrame(() => heading.scrollIntoView({block: 'start'}));
      }
    }
  }
  scope.addEventListener('change', applyScope);
  window.addEventListener('hashchange', navigate);
  document.addEventListener('click', event => {
    const anchor = event.target.closest('a[href^="#req-"]');
    if (anchor && anchor.hash === location.hash) { event.preventDefault(); navigate(); }
  });
  navigate();
})();
