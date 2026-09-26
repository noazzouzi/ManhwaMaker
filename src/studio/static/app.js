// ManhwaMaker Studio : coquille de l'application (routes, état en direct, barre latérale).
import { render } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { EtaText, Icon, ProgressBar, Pulse, Toasts, api, fmtRemaining, html, toast } from './ui.js';
import { Home } from './views/home.js';
import { NewVideo } from './views/new.js';
import { JobDetail, Queue } from './views/queue.js';
import { VideoPage } from './views/video.js';

// --- Routes (#/..., rien ne dépend du serveur de fichiers) ---------------------------------
function parseRoute() {
  const raw = location.hash.replace(/^#/, '') || '/';
  const [path, query = ''] = raw.split('?');
  const parts = path.split('/').filter(Boolean).map(decodeURIComponent);
  return { parts, query: new URLSearchParams(query) };
}

function useRoute() {
  const [route, setRoute] = useState(parseRoute());
  useEffect(() => {
    const onChange = () => { setRoute(parseRoute()); document.querySelector('main')?.scrollTo(0, 0); };
    addEventListener('hashchange', onChange);
    return () => removeEventListener('hashchange', onChange);
  }, []);
  return route;
}

// --- État en direct : un instantané des traitements chaque seconde (Server-Sent Events) ------
function useLive() {
  const [state, setState] = useState({ connected: false, jobs: [], now: Date.now() / 1000, lastSeen: null });
  const previous = useRef(new Map());
  useEffect(() => {
    let source;
    let timer;
    // ?static : un seul relevé, sans flux ouvert (captures d'écran, navigateur sans tête).
    if (new URLSearchParams(location.search).has('static')) {
      api('/api/jobs').then((jobs) => setState({ connected: true, jobs, now: Date.now() / 1000, lastSeen: Date.now() })).catch(() => {});
      return undefined;
    }
    const connect = () => {
      source = new EventSource('/api/stream');
      source.onmessage = (event) => {
        const data = JSON.parse(event.data);
        for (const job of data.jobs) {
          const before = previous.current.get(job.id);
          if (before && before !== job.status && ['done', 'failed', 'interrupted'].includes(job.status)) {
            const ok = job.status === 'done';
            toast(`${job.title} · ${job.subtitle} : ${ok ? 'terminé' : 'en échec'}`, { tone: ok ? 'success' : 'error', href: `#/queue/${job.id}` });
          }
          previous.current.set(job.id, job.status);
        }
        setState({ connected: true, jobs: data.jobs, now: data.now, lastSeen: Date.now() });
      };
      source.onerror = () => {
        setState((s) => ({ ...s, connected: false }));
        source.close();
        timer = setTimeout(connect, 2000);
      };
    };
    connect();
    return () => { source?.close(); clearTimeout(timer); };
  }, []);
  return state;
}

// --- Barre latérale ------------------------------------------------------------------------
function ActivityCard({ job }) {
  const running = job.status === 'running';
  return html`<a class="activity" href=${'#/queue/' + job.id}>
    <span class=${'activity-state ' + (running ? 'is-running' : 'is-queued')}>
      ${running ? html`<${Pulse} />` : html`<${Icon} name="hourglass" size=${14} />`}
      ${running ? 'En cours' : 'En attente'}
      ${running && html`<span class="activity-pct">${Math.round(job.percent || 0)} %</span>`}
    </span>
    <span class="activity-title">${job.title}</span>
    <span class="activity-sub">${job.subtitle}</span>
    ${running && (job.current || []).slice(0, 2).map((line) => html`<span class="activity-line">${line}</span>`)}
    ${running && html`<${ProgressBar} percent=${job.percent} label=${'Avancement de ' + job.title} />`}
    <span class="activity-eta"><${EtaText} job=${job} /></span>
  </a>`;
}

function Sidebar({ route, live, system }) {
  const active = live.jobs.filter((j) => j.status === 'running' || j.status === 'queued');
  const current = route.parts[0] || '';
  const link = (href, key, icon, label, badge) => html`<a class=${'nav-item' + (current === key ? ' is-active' : '')} href=${href}
    aria-current=${current === key ? 'page' : undefined}>
    <${Icon} name=${icon} /><span>${label}</span>${badge ? html`<span class="nav-badge">${badge}</span>` : null}</a>`;
  const svc = (ok, label, value) => html`<li class="svc"><span class=${'svc-dot ' + (ok ? 'ok' : 'off')}></span><span>${label}</span><span class="muted">${value}</span></li>`;
  return html`<nav class="sidebar" aria-label="Navigation principale">
    <a class="brand" href="#/"><span class="brand-mark" aria-hidden="true">M</span>
      <span class="brand-text"><strong>ManhwaMaker</strong><span>Studio</span></span></a>
    <div class="nav">
      ${link('#/', '', 'film', 'Rendus')}
      ${link('#/queue', 'queue', 'queue', "File d'attente", active.length || null)}
    </div>
    <a class="btn btn-primary btn-block" href="#/new"><${Icon} name="plus" /><span>Nouvelle vidéo</span></a>
    <section class="activities" aria-label="Traitements en cours">
      ${active.length === 0 && html`<p class="activity-empty">Aucun traitement en cours.</p>`}
      ${active.map((job) => html`<${ActivityCard} job=${job} key=${job.id} />`)}
    </section>
    <div class="sidebar-foot">
      ${!live.connected && html`<p class="offline" role="status"><${Icon} name="wifiOff" size=${16} />Serveur injoignable. Les traitements continuent tant qu'il tourne ; reconnexion…</p>`}
      <h2 class="eyebrow">Services</h2>
      <ul class="svcs">
        ${system ? html`
          ${svc(system.claude, 'Claude (script)', system.claude ? 'prêt' : 'introuvable')}
          ${svc(system.gemini_keys > 0, 'Gemini (repli)', system.gemini_keys ? `${system.gemini_keys} clé${system.gemini_keys > 1 ? 's' : ''}` : 'aucune clé')}
          ${svc(system.kdenlive, 'Kdenlive (rendu)', system.kdenlive ? 'installé' : 'absent')}
          ${svc(!!system.gpu, 'Carte graphique', system.gpu ? system.gpu : 'processeur seul')}
        ` : html`<li class="muted">…</li>`}
      </ul>
    </div>
  </nav>`;
}

// --- Application --------------------------------------------------------------------------
function App() {
  const route = useRoute();
  const live = useLive();
  const [system, setSystem] = useState(null);
  useEffect(() => { api('/api/system').then(setSystem).catch(() => {}); }, []);

  const running = live.jobs.filter((j) => j.status === 'running');
  useEffect(() => {
    const job = running[0];
    document.title = job ? `(${Math.round(job.percent || 0)} %) ${fmtRemaining(job.remaining_s)} · ManhwaMaker Studio` : 'ManhwaMaker Studio';
  }, [running.map((j) => Math.round(j.percent || 0)).join()]);

  const [section, id] = route.parts;
  let view;
  if (section === 'new') view = html`<${NewVideo} live=${live} query=${route.query} />`;
  else if (section === 'queue' && id) view = html`<${JobDetail} id=${id} live=${live} />`;
  else if (section === 'queue') view = html`<${Queue} live=${live} />`;
  else if (section === 'video' && id) view = html`<${VideoPage} name=${id} live=${live} query=${route.query} />`;
  else view = html`<${Home} live=${live} />`;

  return html`<div class="app">
    <${Sidebar} route=${route} live=${live} system=${system} />
    <main id="main" tabindex="-1">${view}</main>
    <${Toasts} />
  </div>`;
}

render(html`<${App} />`, document.getElementById('root'));
