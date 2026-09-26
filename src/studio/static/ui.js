// Briques communes de l'interface : rendu (Preact + htm), icônes, formats, petits composants.
import { h } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import htm from 'htm';

export const html = htm.bind(h);

// --- Icônes (traits, 24x24) --------------------------------------------------------------
const PATHS = {
  plus: '<path d="M12 5v14M5 12h14"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  film: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/>',
  queue: '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
  play: '<path d="M7 4.5v15l12.5-7.5z" fill="currentColor"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  alert: '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  chevR: '<path d="m9 6 6 6-6 6"/>',
  chevD: '<path d="m6 9 6 6 6-6"/>',
  redo: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
  export: '<path d="M12 15V3M7 8l5-5 5 5"/><path d="M5 21h14"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  link: '<path d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1"/><path d="M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1"/>',
  open: '<path d="M14 4h6v6M20 4l-9 9"/><path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>',
  playc: '<circle cx="12" cy="12" r="9"/><path d="m10 8.5 5 3.5-5 3.5z"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
  phone: '<rect x="7" y="2.5" width="10" height="19" rx="2"/><path d="M11 18.5h2"/>',
  monitor: '<rect x="2" y="4" width="20" height="13" rx="2"/><path d="M8 21h8M12 17v4"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
  copies: '<rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/>',
  eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  arrowL: '<path d="M19 12H5M11 18l-6-6 6-6"/>',
  arrowR: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  hourglass: '<path d="M6 3h12M6 21h12M7 3c0 5 10 5 10 9s-10 4-10 9M17 3c0 5-10 5-10 9s10 4 10 9"/>',
  trash: '<path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/>',
  file: '<path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8z"/><path d="M14 3v5h5"/>',
  wifiOff: '<path d="M2 8.8a15 15 0 0 1 4.2-2.6M22 8.8a15 15 0 0 0-10-3.8M5 12.5a10 10 0 0 1 3.4-2M19 12.5a10 10 0 0 0-2.1-1.4M8.5 16a5 5 0 0 1 7 0M12 20h.01M3 3l18 18"/>',
};

export function Icon({ name, size = 18, className = '' }) {
  return html`<svg class=${'icon ' + className} width=${size} height=${size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"
    dangerouslySetInnerHTML=${{ __html: PATHS[name] || '' }}></svg>`;
}

// --- Formats -------------------------------------------------------------------------------
export function fmtDuration(seconds) {
  seconds = Math.max(0, Math.round(seconds || 0));
  const hh = Math.floor(seconds / 3600), mm = Math.floor((seconds % 3600) / 60), ss = seconds % 60;
  return hh ? `${hh}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}` : `${mm}:${String(ss).padStart(2, '0')}`;
}

export function fmtRemaining(seconds) {
  if (seconds == null) return '';
  seconds = Math.max(0, Math.round(seconds));
  if (seconds < 50) return `≈ ${Math.max(5, Math.round(seconds / 5) * 5)} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `≈ ${minutes} min`;
  return `≈ ${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, '0')}`;
}

export function fmtClock(epoch) {
  if (!epoch) return '';
  const d = new Date(epoch * 1000);
  const today = new Date();
  const time = d.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
  return d.toDateString() === today.toDateString() ? time : `${d.toLocaleDateString('fr-FR', { weekday: 'short' })} ${time}`;
}

export function fmtDate(epoch) {
  if (!epoch) return '';
  return new Date(epoch * 1000).toLocaleDateString('fr-FR', { day: 'numeric', month: 'short' });
}

export function fmtElapsed(seconds) {
  seconds = Math.max(0, Math.round(seconds || 0));
  if (seconds < 60) return `${seconds} s`;
  const m = Math.floor(seconds / 60), s = seconds % 60;
  if (m < 60) return `${m} min ${String(s).padStart(2, '0')}`;
  return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, '0')}`;
}

export const EMOTIONS = {
  tension: ['tension', '#C98B4A'], action: ['action', '#FF6B3D'], calm: ['calme', '#5FB3A8'], epic: ['épique', '#E6C65C'],
  mystery: ['mystère', '#9A8CF2'], sad: ['triste', '#5F8FD6'], happy: ['joie', '#A8D46A'], fear: ['peur', '#C0648A'],
  romance: ['romance', '#E48FB0'], comedy: ['comédie', '#9CD3E8'],
};
export const emotionColor = (e) => (EMOTIONS[e] || [e, '#5A5660'])[1];
export const emotionLabel = (e) => (EMOTIONS[e] || [e || 'neutre'])[0];

// --- API -----------------------------------------------------------------------------------
export async function api(path, { method = 'GET', body } = {}) {
  const res = await fetch(path, {
    method, headers: body ? { 'Content-Type': 'application/json' } : undefined, body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) throw new Error((data && data.detail) || `Erreur ${res.status}`);
  return data;
}

export function navigate(hash) { location.hash = hash; }

// --- Composants -----------------------------------------------------------------------------
export function Button({ kind = 'secondary', icon, children, href, onClick, disabled, type = 'button', className = '', title, ...rest }) {
  const cls = `btn btn-${kind} ${className}`;
  const inner = html`${icon && html`<${Icon} name=${icon} />`}${children && html`<span>${children}</span>`}`;
  if (href) return html`<a class=${cls} href=${href} title=${title} ...${rest}>${inner}</a>`;
  return html`<button class=${cls} type=${type} onClick=${onClick} disabled=${disabled} title=${title} ...${rest}>${inner}</button>`;
}

export function IconButton({ icon, label, onClick, href, className = '' }) {
  if (href) return html`<a class=${'btn-icon ' + className} href=${href} aria-label=${label} title=${label}><${Icon} name=${icon} /></a>`;
  return html`<button type="button" class=${'btn-icon ' + className} aria-label=${label} title=${label} onClick=${onClick}><${Icon} name=${icon} /></button>`;
}

export function ProgressBar({ percent, tone = 'blue', label }) {
  const value = Math.max(0, Math.min(100, percent || 0));
  return html`<div class=${'bar bar-' + tone} role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${Math.round(value)}
    aria-label=${label || 'Avancement'}><span style=${{ width: value + '%' }}></span></div>`;
}

export function Pulse() { return html`<span class="pulse" aria-hidden="true"></span>`; }

export function EtaText({ job }) {
  if (!job) return null;
  if (job.status === 'queued') return html`<span>${job.current?.[0] || 'En attente'}</span>`;
  if (job.status !== 'running') return null;
  return html`<span>${fmtRemaining(job.remaining_s)} restantes · fin vers ${fmtClock(job.eta_at)}${job.overrun ? ' · plus long que d’habitude' : ''}</span>`;
}

export function StatusChip({ status, overlay = false }) {
  const map = {
    final: ['check', 'Vidéo finale prête', 'chip-final'], preview: ['eye', 'Aperçu prêt', 'chip-preview'],
    montage: ['film', 'Montage prêt', 'chip-preview'], failed: ['alert', 'Échec', 'chip-fail'],
    incomplete: ['alert', 'Inachevé', 'chip-fail'], running: [null, 'En cours', 'chip-run'], queued: ['hourglass', 'En attente', 'chip-wait'],
    done: ['check', 'Terminé', 'chip-done'], cancelled: ['stop', 'Arrêté', 'chip-wait'], interrupted: ['alert', 'Interrompu', 'chip-fail'],
  };
  const [icon, label, cls] = map[status] || [null, status, 'chip-wait'];
  return html`<span class=${`chip ${cls} ${overlay ? 'chip-overlay' : ''}`}>${status === 'running' ? html`<${Pulse} />` : icon && html`<${Icon} name=${icon} size=${14} />`}${label}</span>`;
}

export function Modal({ title, subtitle, onClose, children, footer }) {
  const ref = useRef();
  useEffect(() => {
    const previous = document.activeElement;
    const first = ref.current?.querySelector('button, [href], input, select, textarea');
    first?.focus();
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('keydown', onKey); previous?.focus?.(); };
  }, []);
  return html`<div class="scrim" onClick=${(e) => e.target === e.currentTarget && onClose()}>
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title" ref=${ref}>
      <div class="modal-head">
        <div><h2 id="modal-title">${title}</h2>${subtitle && html`<p class="muted">${subtitle}</p>`}</div>
        <${IconButton} icon="x" label="Fermer" onClick=${onClose} className="ghost" />
      </div>
      <div class="modal-body">${children}</div>
      ${footer && html`<div class="modal-foot">${footer}</div>`}
    </div>
  </div>`;
}

let toastId = 0;
const toastListeners = new Set();
export function toast(message, { tone = 'info', href } = {}) {
  const item = { id: ++toastId, message, tone, href };
  toastListeners.forEach((fn) => fn(item));
}

export function Toasts() {
  const [items, setItems] = useState([]);
  useEffect(() => {
    const add = (item) => {
      setItems((list) => [...list, item]);
      setTimeout(() => setItems((list) => list.filter((t) => t.id !== item.id)), 7000);
    };
    toastListeners.add(add);
    return () => toastListeners.delete(add);
  }, []);
  return html`<div class="toasts" aria-live="polite">
    ${items.map((t) => html`<div class=${'toast toast-' + t.tone} key=${t.id}>
      <${Icon} name=${t.tone === 'error' ? 'alert' : 'check'} />
      <span>${t.message}</span>
      ${t.href && html`<a href=${t.href}>Voir</a>`}
    </div>`)}
  </div>`;
}

export function useInterval(fn, ms, deps = []) {
  useEffect(() => {
    if (!ms) return undefined;
    const id = setInterval(fn, ms);
    return () => clearInterval(id);
  }, [ms, ...deps]);
}
