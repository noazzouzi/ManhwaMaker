// File d'attente et suivi détaillé d'un traitement : étape de chaque chapitre, temps restant, journal.
import { useEffect, useRef, useState } from 'preact/hooks';
import { Button, EtaText, Icon, IconButton, Modal, ProgressBar, Pulse, StatusChip, api, fmtClock, fmtElapsed, fmtRemaining, html, navigate, toast, useInterval } from '../ui.js';

const STAGES = [['prep', 'Préparation', 'téléchargement, cases, personnages'], ['script', 'Script', 'écriture par l’IA'],
  ['voice', 'Voix', 'synthèse Kokoro'], ['montage', 'Montage', 'plan, brouillon, aperçu']];

function JobRow({ job }) {
  return html`<li class="job-row card">
    <a class="job-row-link" href=${'#/queue/' + job.id}>
      <span class="job-row-main">
        <span class="job-row-title">${job.title}<span class="muted"> · ${job.subtitle}</span></span>
        <span class="job-row-line">${job.status === 'running' || job.status === 'queued' ? (job.current || [])[0] : job.error ? job.error.slice(0, 140) : doneLine(job)}</span>
      </span>
      ${job.status === 'running' && html`<span class="job-row-progress"><${ProgressBar} percent=${job.percent} /><span class="small"><${EtaText} job=${job} /></span></span>`}
      ${job.status === 'queued' && html`<span class="job-row-progress small muted"><${EtaText} job=${job} /></span>`}
      <${StatusChip} status=${job.status} />
    </a>
  </li>`;
}

function doneLine(job) {
  const when = job.finished ? `le ${new Date(job.finished * 1000).toLocaleDateString('fr-FR', { day: 'numeric', month: 'short' })} à ${fmtClock(job.finished)}` : '';
  const took = job.started && job.finished ? ` en ${fmtElapsed(job.finished - job.started)}` : '';
  const counts = job.counts || {};
  const extra = counts.failed ? ` · ${counts.failed} chapitre${counts.failed > 1 ? 's' : ''} en échec` : '';
  return `Fini ${when}${took}${extra}`;
}

export function Queue({ live }) {
  const [all, setAll] = useState(null);
  const load = () => api('/api/jobs').then(setAll).catch(() => {});
  useEffect(() => { load(); }, [live.jobs.map((j) => j.id + j.status).join()]);
  const jobs = all || [];
  const byId = Object.fromEntries(live.jobs.map((j) => [j.id, j]));
  const merged = jobs.map((j) => byId[j.id] || j);
  const groups = [['Traitements en cours', merged.filter((j) => j.status === 'running')],
    ['En attente', merged.filter((j) => j.status === 'queued')], ['Terminés', merged.filter((j) => !['running', 'queued'].includes(j.status))]];
  return html`<div class="page">
    <header class="page-head"><div class="grow"><h1>File d'attente</h1>
      <p class="muted">Un lot à la fois, et un rendu à la fois. Tout continue sur le serveur, même page fermée.</p></div>
      <${Button} kind="primary" icon="plus" href="#/new">Nouvelle vidéo<//></header>
    ${all && all.length === 0 && html`<div class="empty card"><p>Aucun traitement pour l'instant.</p></div>`}
    ${groups.filter(([, list]) => list.length).map(([label, list]) => html`<section class="job-group" aria-label=${label}>
      <h2 class="eyebrow">${label}</h2><ul class="job-list">${list.map((j) => html`<${JobRow} job=${j} key=${j.id} />`)}</ul></section>`)}
  </div>`;
}

// --- Détail d'un traitement --------------------------------------------------------------------
function StageCell({ chapter, stage }) {
  const index = STAGES.findIndex((s) => s[0] === stage);
  const current = STAGES.findIndex((s) => s[0] === chapter.stage);
  const span = chapter.stages?.[stage];
  const took = span?.start && span?.end ? fmtElapsed(span.end - span.start) : null;
  if (chapter.status === 'skipped') return html`<span class="cell cell-skip">déjà fait</span>`;
  if (chapter.status === 'failed' && current === index) {
    return html`<span class="cell cell-fail"><${Icon} name="alert" size=${18} /><strong>Échec</strong></span>`;
  }
  if (chapter.status === 'failed' && current < index) {
    return html`<span class="cell cell-pending"><span class="dot-pending"></span><span class="sr-only">non fait</span></span>`;
  }
  if (chapter.status === 'done' || (current > index) || (span && span.end)) {
    return html`<span class="cell cell-done"><span class="dot-done"><${Icon} name="check" size=${13} /></span>${took && html`<span class="mono muted">${took}</span>`}<span class="sr-only">fait</span></span>`;
  }
  if (chapter.status === 'processing' && current === index) {
    if (chapter.state === 'waiting') {
      return html`<span class="cell cell-wait"><${Icon} name="hourglass" size=${15} /><span>${chapter.reason || 'En attente'}</span></span>`;
    }
    const step = chapter.step;
    const frac = step && step.total ? Math.min(1, (step.done || 0) / step.total) : null;
    return html`<span class="cell cell-run"><span class="run-top"><${Pulse} /><span class="mono">${fmtElapsed(Date.now() / 1000 - (chapter.since || Date.now() / 1000))}</span></span>
      <span class="run-label">${step ? step.label : 'Démarrage'}${step && step.total ? ` · ${['melt', 'preview'].includes(step.id) ? Math.round(frac * 100) + ' %' : `${Math.min(step.done || 0, step.total)}/${step.total}`}` : ''}</span>
      ${frac != null && html`<${ProgressBar} percent=${frac * 100} />`}</span>`;
  }
  return html`<span class="cell cell-pending"><span class="dot-pending"></span><span class="sr-only">à venir</span></span>`;
}

function ChapterTable({ job }) {
  const chapters = job.chapters || [];
  if (!chapters.length) {
    return html`<p class="muted">${job.status === 'running' ? 'Recherche des chapitres…' : 'Aucun chapitre.'}</p>`;
  }
  return html`<div class="ctable" role="table" aria-label="Avancement des chapitres">
    <div class="ctable-row ctable-head" role="row">
      <span role="columnheader">Chapitre</span>
      ${STAGES.map(([, label, hint]) => html`<span role="columnheader" title=${hint}>${label}</span>`)}
      <span role="columnheader">Fin prévue</span>
    </div>
    ${chapters.map((c) => html`<div class=${'ctable-row' + (c.status === 'failed' ? ' is-failed' : '') + (c.status === 'done' || c.status === 'skipped' ? ' is-done' : '')} role="row" key=${c.key}>
      <span role="cell" class="ctable-name">
        ${c.out_dir && (c.status === 'done' || c.status === 'skipped') && !job.params?.compile
          ? html`<a href=${'#/video/' + encodeURIComponent(c.out_dir)}>Chapitre ${c.episode_no ?? '?'}</a>` : html`<strong>Chapitre ${c.episode_no ?? '?'}</strong>`}
        ${c.video_s ? html`<span class="muted small">${fmtElapsed(c.video_s)} de vidéo</span>` : null}
      </span>
      ${STAGES.map(([stage]) => html`<span role="cell"><${StageCell} chapter=${c} stage=${stage} /></span>`)}
      <span role="cell" class="mono small">${c.status === 'processing' || c.status === 'pending' ? (c.eta_at ? `${fmtClock(c.eta_at)}` : '—') : c.status === 'done' ? html`<span class="ok-text">prêt</span>` : ''}</span>
      ${c.status === 'failed' && c.error && html`<p class="ctable-error" role="cell">${c.error}</p>`}
    </div>`)}
    ${job.params?.compile && html`<div class=${'ctable-row ctable-compile' + (job.phase === 'compile' ? ' is-running' : '')} role="row">
      <span role="cell" class="ctable-name"><strong>Compilation</strong><span class="muted small">une seule vidéo</span></span>
      <span role="cell" class="compile-cell">${job.phase === 'compile' ? html`<span class="cell cell-run"><span class="run-top"><${Pulse} /></span><span class="run-label">${job.run_step?.label || 'Assemblage'}${job.run_step?.total ? ` · ${Math.round(100 * (job.run_step.done || 0) / job.run_step.total)} %` : ''}</span></span>`
        : job.status === 'done' ? html`<span class="cell cell-done"><span class="dot-done"><${Icon} name="check" size=${13} /></span>faite</span>`
        : html`<span class="muted small">après le dernier chapitre : assemblage, brouillon CapCut, extrait de 2 min</span>`}</span>
    </div>`}
  </div>`;
}

function RenderSteps({ job }) {
  const phase = job.phase;
  const step = job.run_step;
  const steps = [['project', 'Préparation du projet', 'images des cases, calques, projet .kdenlive'], ['render', 'Rendu de la vidéo', 'melt, encodage'], ['done', 'Vidéo finale', '']];
  const order = ['starting', 'project', 'render', 'done'];
  const at = order.indexOf(job.status === 'done' ? 'done' : phase || 'starting');
  return html`<ol class="rsteps">${steps.map(([key, label, hint], i) => {
    const state = job.status === 'failed' && order.indexOf(key) === at ? 'fail' : order.indexOf(key) < at || (key === 'done' && job.status === 'done') ? 'done' : order.indexOf(key) === at ? 'run' : 'todo';
    return html`<li class=${'rstep rstep-' + state}>
      <span class="rstep-dot">${state === 'done' ? html`<${Icon} name="check" size=${13} />` : state === 'run' ? html`<${Pulse} />` : state === 'fail' ? html`<${Icon} name="alert" size=${13} />` : i + 1}</span>
      <span class="rstep-text"><strong>${label}</strong><span class="muted small">${state === 'run' && step ? step.label + (step.total ? ` · ${step.id === 'melt' ? Math.round(100 * (step.done || 0) / step.total) + ' %' : `${Math.min(step.done || 0, step.total)}/${step.total}`}` : '') : hint}</span>
        ${state === 'run' && step?.total ? html`<${ProgressBar} percent=${100 * (step.done || 0) / step.total} />` : null}</span>
    </li>`;
  })}</ol>`;
}

function Journal({ job }) {
  const [errorsOnly, setErrorsOnly] = useState(false);
  const [follow, setFollow] = useState(true);
  const box = useRef();
  const lines = (job.log || []).filter((l) => !errorsOnly || /ERROR|WARNING|FAILED|Traceback|Error/.test(l));
  useEffect(() => { if (follow && box.current) box.current.scrollTop = box.current.scrollHeight; }, [job.log?.length, errorsOnly, follow]);
  return html`<section class="card journal" aria-labelledby="journal-title">
    <div class="journal-head"><h2 id="journal-title">Journal</h2>
      <fieldset class="chips small-chips"><legend class="sr-only">Filtrer le journal</legend>
        <button type="button" class=${'fchip' + (!errorsOnly ? ' is-on' : '')} aria-pressed=${!errorsOnly} onClick=${() => setErrorsOnly(false)}>Tout</button>
        <button type="button" class=${'fchip' + (errorsOnly ? ' is-on' : '')} aria-pressed=${errorsOnly} onClick=${() => setErrorsOnly(true)}>Alertes</button>
      </fieldset>
      <label class="check inline"><input type="checkbox" checked=${follow} onChange=${() => setFollow(!follow)} /><span>Suivre</span></label>
      <a class="small" href=${'/api/jobs/' + job.id + '/log'} target="_blank" rel="noopener">Journal complet</a>
    </div>
    <pre class="log" ref=${box} tabindex="0" aria-label="Dernières lignes du journal">${lines.length ? lines.map((l) => html`<span class=${/ERROR|FAILED|Traceback/.test(l) ? 'log-err' : /WARNING/.test(l) ? 'log-warn' : ''}>${l}</span>`) : 'Rien pour l’instant.'}</pre>
  </section>`;
}

export function JobDetail({ id, live }) {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [confirmStop, setConfirmStop] = useState(false);
  const load = () => api('/api/jobs/' + id).then((j) => { setJob(j); setError(null); }).catch((e) => setError(e.message));
  useEffect(() => { setJob(null); load(); }, [id]);
  const active = job && (job.status === 'running' || job.status === 'queued');
  useInterval(load, active ? 1000 : 0, [id, active]);
  const summary = live.jobs.find((j) => j.id === id);
  useEffect(() => { if (summary && job && summary.status !== job.status) load(); }, [summary?.status]);

  if (error) return html`<div class="page"><a class="back" href="#/queue"><${Icon} name="arrowL" size=${16} />File d'attente</a><p class="alert" role="alert">${error}</p></div>`;
  if (!job) return html`<div class="page"><p class="muted">Chargement…</p></div>`;

  const stop = async () => {
    setConfirmStop(false);
    try { await api(`/api/jobs/${id}/cancel`, { method: 'POST' }); toast('Traitement arrêté'); load(); } catch (e) { toast(e.message, { tone: 'error' }); }
  };
  const retry = async () => {
    try { const j = await api(`/api/jobs/${id}/retry`, { method: 'POST' }); navigate('#/queue/' + j.id); } catch (e) { toast(e.message, { tone: 'error' }); }
  };
  const remove = async () => {
    try { await api(`/api/jobs/${id}`, { method: 'DELETE' }); navigate('#/queue'); } catch (e) { toast(e.message, { tone: 'error' }); }
  };
  const counts = job.counts || {};
  const elapsed = job.started ? (job.finished || Date.now() / 1000) - job.started : 0;
  const target = job.target;
  return html`<div class="page">
    <a class="back" href="#/queue"><${Icon} name="arrowL" size=${16} />File d'attente</a>
    <header class="page-head">
      <div class="grow"><h1>${job.title}</h1><p class="muted">${job.subtitle}</p></div>
      ${job.status === 'running' && html`<${Button} kind="ghost" icon="stop" onClick=${() => setConfirmStop(true)}>Arrêter<//>`}
      ${job.status === 'queued' && html`<${Button} kind="ghost" icon="x" onClick=${stop}>Retirer de la file<//>`}
      ${['failed', 'cancelled', 'interrupted'].includes(job.status) && html`<${Button} kind="primary" icon="redo" onClick=${retry}>Relancer<//>`}
      ${!['running', 'queued'].includes(job.status) && html`<${Button} kind="ghost" icon="trash" onClick=${remove}>Retirer de la liste<//>`}
      ${target && job.status === 'done' && html`<${Button} kind="secondary" icon="play" href=${'#/video/' + encodeURIComponent(target)}>Voir la vidéo<//>`}
      ${!target && job.status === 'done' && html`<${Button} kind="secondary" icon="film" href="#/">Voir les rendus<//>`}
    </header>

    <section class="now card" aria-live="polite" aria-labelledby="now-title">
      <div class="now-main">
        <span class="eyebrow">${job.status === 'running' ? html`<${Pulse} />Étape en cours` : html`<${StatusChip} status=${job.status} />`}</span>
        <h2 id="now-title" class="now-step">${job.status === 'running' ? (job.current || []).join(' · ') || 'Démarrage…'
          : job.status === 'queued' ? (job.current || [])[0]
          : job.status === 'done' ? `Terminé en ${fmtElapsed(elapsed)}${counts.failed ? ` · ${counts.failed} chapitre${counts.failed > 1 ? 's' : ''} en échec` : ''}`
          : job.error || 'Arrêté'}</h2>
        ${(job.status === 'running' || job.status === 'queued') && html`<${ProgressBar} percent=${job.percent} label="Avancement du traitement" />`}
        <p class="muted small">${job.status === 'running' ? `Démarré à ${fmtClock(job.started)} · écoulé ${fmtElapsed(elapsed)} · estimation ${job.basis}.` : ''}</p>
      </div>
      ${(job.status === 'running' || job.status === 'queued') && html`<div class="now-eta">
        <span class="muted">Temps restant</span><strong class="big">${fmtRemaining(job.remaining_s)}</strong>
        <span class="muted">fin vers <strong class="tx">${fmtClock(job.eta_at)}</strong></span>
        ${job.overrun && html`<span class="warn-text small">Plus long que d'habitude</span>`}
      </div>`}
    </section>

    ${job.kind === 'batch' && counts.total ? html`<div class="stats">
      <div><span class="muted">Terminés</span><strong>${counts.done + (counts.skipped || 0)}<span class="muted"> / ${counts.total}</span></strong></div>
      <div><span class="muted">En cours</span><strong class="blue">${counts.running}</strong></div>
      <div><span class="muted">En attente</span><strong>${counts.waiting + counts.pending}</strong></div>
      <div><span class="muted">Échecs</span><strong class=${counts.failed ? 'warn-text' : ''}>${counts.failed}</strong></div>
    </div>` : null}

    <div class="detail-grid">
      <section class="card pad" aria-label="Étapes">
        ${job.kind === 'batch' ? html`<${ChapterTable} job=${job} />` : html`<${RenderSteps} job=${job} />`}
        ${job.error && job.status !== 'running' && html`<p class="alert" role="alert"><${Icon} name="alert" />${job.error}</p>`}
        <details class="cmd"><summary>Commande exacte</summary><code>${job.cmd}</code></details>
      </section>
      <${Journal} job=${job} />
    </div>

    ${confirmStop && html`<${Modal} title="Arrêter ce traitement ?" subtitle=${job.title + ' · ' + job.subtitle} onClose=${() => setConfirmStop(false)}
      footer=${html`<${Button} kind="ghost" onClick=${() => setConfirmStop(false)}>Continuer le traitement<//><${Button} kind="danger" icon="stop" onClick=${stop}>Arrêter<//>`}>
      <p>Les chapitres terminés sont gardés. Le chapitre en cours s'arrête au milieu de son étape ; « Relancer » le reprendra là où les fichiers déjà écrits le permettent.</p>
    <//>`}
  </div>`;
}
