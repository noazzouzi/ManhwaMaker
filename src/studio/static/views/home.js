// Accueil : toutes les vidéos, le traitement en cours en tête.
import { useEffect, useMemo, useState } from 'preact/hooks';
import { Button, EtaText, Icon, IconButton, ProgressBar, Pulse, StatusChip, api, fmtDate, fmtDuration, html, navigate, toast, useInterval } from '../ui.js';

const FILTERS = [
  ['all', 'Toutes', () => true],
  ['todo', 'À rendre', (v) => v.status === 'preview' || v.status === 'montage'],
  ['final', 'Prêtes', (v) => v.status === 'final'],
  ['failed', 'Échecs', (v) => v.status === 'failed' || v.status === 'incomplete'],
];

export function useVideos(live) {
  const [videos, setVideos] = useState(null);
  const [error, setError] = useState(null);
  const load = () => api('/api/videos').then((v) => { setVideos(v); setError(null); }).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);
  useInterval(load, 30000);
  // Un traitement vient de finir : la bibliothèque a changé.
  const finished = live.jobs.filter((j) => !['queued', 'running'].includes(j.status)).map((j) => j.id + j.status).join();
  useEffect(() => { if (videos) load(); }, [finished]);
  return { videos, error, reload: load };
}

function LiveBanner({ job }) {
  const counts = job.counts || {};
  const segments = [];
  if (job.kind === 'batch' && counts.total) {
    for (const [key, n] of [['done', counts.done + (counts.skipped || 0)], ['running', counts.running], ['waiting', counts.waiting], ['failed', counts.failed], ['pending', counts.pending]]) {
      for (let i = 0; i < (n || 0); i++) segments.push(key);
    }
  }
  return html`<section class="banner card" aria-labelledby=${'banner-' + job.id}>
    <div class="banner-main">
      <div class="banner-top"><span class="eyebrow eyebrow-blue"><${Pulse} />Traitement en cours</span>
        <span class="muted">démarré à ${new Date(job.started * 1000).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}</span></div>
      <h2 id=${'banner-' + job.id}>${job.title} <span class="muted-strong">· ${job.subtitle}</span></h2>
      ${segments.length ? html`<div class="segbar" aria-hidden="true">${segments.map((s) => html`<span class=${'seg-' + s}></span>`)}</div>`
        : html`<${ProgressBar} percent=${job.percent} />`}
      <p class="banner-line">
        ${counts.total ? html`<span>${counts.done} terminé${counts.done > 1 ? 's' : ''} sur ${counts.total}</span>` : null}
        ${(job.current || []).slice(0, 1).map((l) => html`<span>${l}</span>`)}
        <strong><${EtaText} job=${job} /></strong>
      </p>
    </div>
    <${Button} href=${'#/queue/' + job.id} icon="chevR">Suivre<//>
  </section>`;
}

function VideoCard({ video, job, twin }) {
  const busy = job && (job.status === 'running' || job.status === 'queued');
  const failed = video.status === 'failed' || video.status === 'incomplete';
  const retry = async () => {
    try {
      const created = await api(`/api/videos/${encodeURIComponent(video.name)}/redo`, { method: 'POST', body: {} });
      toast('Chapitre relancé', { href: '#/queue/' + created.id });
      navigate('#/queue/' + created.id);
    } catch (e) { toast(e.message, { tone: 'error' }); }
  };
  let action;
  if (busy) action = html`<${Button} kind="secondary" href=${'#/queue/' + job.id} icon="queue" className="grow">Suivre<//>`;
  else if (failed) action = html`<${Button} kind="warn" icon="redo" onClick=${retry} className="grow" disabled=${!video.url}>Relancer<//>`;
  else if (video.status === 'final') action = html`<${Button} kind="secondary" icon="play" href=${'#/video/' + encodeURIComponent(video.name)} className="grow">Regarder<//>`;
  else action = html`<${Button} kind="primary" icon="export" href=${'#/video/' + encodeURIComponent(video.name) + '?render=1'} className="grow">Rendre la vidéo<//>`;
  const title = video.series;
  const meta = [video.kind === 'compilation' ? (video.episode || 'Compilation') : (video.episode || ''), fmtDate(video.updated),
    twin ? video.name : null].filter(Boolean).join(' · ');
  return html`<article class="vcard card">
    <a class="vcard-link" href=${'#/video/' + encodeURIComponent(video.name)}>
      <span class="vcard-thumb">
        ${video.has_timeline || video.media.length ? html`<img src=${`/api/videos/${encodeURIComponent(video.name)}/thumb`} alt="" loading="lazy" class=${failed ? 'dim' : ''} />`
          : html`<span class="thumb-empty"><${Icon} name="film" size=${28} /></span>`}
        <${StatusChip} status=${busy ? job.status : video.status} overlay />
        ${video.duration_s ? html`<span class="duration">${fmtDuration(video.duration_s)}</span>` : null}
      </span>
      <span class="vcard-body">
        <span class="vcard-title" title=${title}>${title}</span>
        <span class="vcard-meta">${meta}</span>
      </span>
    </a>
    ${busy && job.status === 'running' && html`<div class="vcard-progress"><${ProgressBar} percent=${job.percent} /><span class="muted small"><${EtaText} job=${job} /></span></div>`}
    ${failed && !busy && html`<p class="vcard-error">${video.error ? shortError(video.error) : 'Traitement interrompu avant le montage.'}</p>`}
    ${!failed && !busy && (video.broken || []).length > 0 && video.status !== 'final' && html`<p class="vcard-error">Rendu final interrompu : fichier illisible, à relancer.</p>`}
    <div class="vcard-actions">${action}
      <${IconButton} icon="folder" label=${'Ouvrir le dossier de ' + title} onClick=${() => api(`/api/videos/${encodeURIComponent(video.name)}/open`, { method: 'POST', body: { what: 'folder' } }).catch((e) => toast(e.message, { tone: 'error' }))} />
    </div>
  </article>`;
}

export function shortError(message) {
  if (/Quota/i.test(message)) return 'Quota Gemini épuisé pendant le script.';
  if (/Claude/i.test(message) && /limit/i.test(message)) return 'Limite d’usage de Claude atteinte.';
  return message.length > 110 ? message.slice(0, 110) + '…' : message;
}

export function Home({ live }) {
  const { videos, error } = useVideos(live);
  const [filter, setFilter] = useState('all');
  const [query, setQuery] = useState('');
  const running = live.jobs.filter((j) => j.status === 'running');
  const byTarget = useMemo(() => {
    const map = {};
    for (const job of live.jobs) if (job.target && (job.status === 'running' || job.status === 'queued')) map[job.target] = job;
    return map;
  }, [live.jobs]);
  const shown = (videos || []).filter((v) => FILTERS.find((f) => f[0] === filter)[2](v))
    .filter((v) => !query || (v.series + ' ' + v.episode + ' ' + v.name).toLowerCase().includes(query.toLowerCase()));
  const count = (key) => (videos || []).filter(FILTERS.find((f) => f[0] === key)[2]).length;
  // Même série, même épisode dans plusieurs dossiers (essais) : on affiche le nom du dossier.
  const twins = new Set();
  const seen = {};
  for (const v of videos || []) {
    const key = v.series + '|' + v.episode;
    if (seen[key]) { twins.add(v.name); twins.add(seen[key]); } else seen[key] = v.name;
  }
  return html`<div class="page">
    <header class="page-head">
      <div class="grow"><h1>Rendus</h1><p class="muted">${videos ? `${videos.length} vidéos` : 'Chargement…'}${running.length ? ` · ${running.length} traitement${running.length > 1 ? 's' : ''} en cours` : ''}</p></div>
      <label class="search"><${Icon} name="search" /><span class="sr-only">Rechercher une série</span>
        <input type="search" placeholder="Rechercher une série" value=${query} onInput=${(e) => setQuery(e.target.value)} /></label>
      <${Button} kind="primary" icon="plus" href="#/new">Nouvelle vidéo<//>
    </header>
    ${running.map((job) => html`<${LiveBanner} job=${job} key=${job.id} />`)}
    <div class="toolbar">
      <fieldset class="chips"><legend class="sr-only">Filtrer les vidéos</legend>
        ${FILTERS.map(([key, label]) => html`<button type="button" class=${'fchip' + (filter === key ? ' is-on' : '')} aria-pressed=${filter === key}
          onClick=${() => setFilter(key)}>${label}<span class="fchip-n">${count(key)}</span></button>`)}
      </fieldset>
    </div>
    ${error && html`<p class="alert" role="alert"><${Icon} name="alert" />${error}</p>`}
    ${videos && shown.length === 0 && html`<div class="empty card"><p>Aucune vidéo ${filter === 'all' ? '' : 'dans ce filtre'}.</p>
      <${Button} kind="primary" icon="plus" href="#/new">Créer une vidéo<//></div>`}
    <div class="vgrid">${shown.map((v) => html`<${VideoCard} key=${v.name} video=${v} job=${byTarget[v.name]} twin=${twins.has(v.name)} />`)}</div>
  </div>`;
}
