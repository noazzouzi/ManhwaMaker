// Page d'une vidéo : lecteur, scènes (texte, émotion, images), rendu final et reprise d'une étape.
import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import { Button, EtaText, Icon, IconButton, Modal, ProgressBar, Pulse, StatusChip, api, emotionColor, emotionLabel, fmtClock, fmtDuration, fmtRemaining, html, navigate, toast } from '../ui.js';
import { shortError } from './home.js';

const enc = encodeURIComponent;

function RenderDialog({ video, onClose }) {
  const [fps, setFps] = useState(30);
  const [codec, setCodec] = useState('h264_amf');
  const [estimate, setEstimate] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  useEffect(() => {
    api(`/api/videos/${enc(video.name)}/render-estimate?fps=${fps}&codec=${codec}`).then(setEstimate).catch(() => setEstimate(null));
  }, [fps, codec]);
  const launch = async () => {
    setBusy(true);
    try {
      await api(`/api/videos/${enc(video.name)}/render`, { method: 'POST', body: { fps, codec } });
      toast('Rendu lancé : suivez-le ici ou dans la file d’attente');
      onClose(true);
    } catch (e) { setError(e.message); setBusy(false); }
  };
  const seg = (value, current, set, label, hint) => html`<label class=${'seg-opt' + (value === current ? ' is-on' : '')}>
    <input type="radio" class="sr-only" checked=${value === current} onChange=${() => set(value)} />${label}${hint && html`<span class="seg-hint">${hint}</span>`}</label>`;
  const existing = video.media.find((m) => m.kind === 'final');
  return html`<${Modal} title="Rendre la vidéo finale" subtitle=${`${video.series} · ${video.episode} · ${fmtDuration(video.duration_s)}`} onClose=${() => onClose(false)}
    footer=${html`<${Button} kind="ghost" onClick=${() => onClose(false)}>Annuler<//><${Button} kind="primary" icon="export" onClick=${launch} disabled=${busy}>${busy ? 'Lancement…' : 'Lancer le rendu'}<//>`}>
    <p class="muted">Rendu automatique par Kdenlive, sans ouvrir d'éditeur : style dynamique (transition à chaque coupe selon l'émotion), sous-titres incrustés. Le projet .kdenlive reste modifiable ensuite.</p>
    <div class="two-cols">
      <fieldset><legend class="field-label">Images par seconde</legend>
        <div class="segmented">${seg(30, fps, setFps, '30 i/s', 'recommandé')}${seg(60, fps, setFps, '60 i/s', '2× plus long')}</div></fieldset>
      <fieldset><legend class="field-label">Encodage</legend>
        <div class="segmented">${seg('h264_amf', codec, setCodec, 'Carte graphique', 'AMD')}${seg('libx264', codec, setCodec, 'Processeur', 'plus lent')}</div></fieldset>
    </div>
    <div class="estimate-box">
      <div><span class="muted small">Temps estimé</span><strong class="big">${estimate ? fmtRemaining(estimate.seconds) : '…'}</strong>
        <span class="muted small">${estimate ? `${estimate.wait_s > 30 ? `après le rendu en cours (${fmtRemaining(estimate.wait_s)}), ` : ''}fin vers ${fmtClock(Date.now() / 1000 + estimate.seconds + (estimate.wait_s || 0))} · ${estimate.basis}` : ''}</span></div>
      <div class="right"><span class="muted small">Fichier</span><span class="mono small">kdenlive/${estimate?.file || ''}</span>
        ${existing && html`<span class="muted small">une version existe déjà : ${existing.label}</span>`}</div>
    </div>
    ${error && html`<p class="alert" role="alert"><${Icon} name="alert" />${error}</p>`}
  <//>`;
}

function RedoBox({ video }) {
  const [stage, setStage] = useState('tts');
  const [confirm, setConfirm] = useState(false);
  const labels = { analyze: ['Script', 'Claude réécrit le script (≈ 1 $), puis voix et montage.'], tts: ['Voix', 'Nouvelle voix off avec les réglages actuels, puis montage. Aucun appel à l’IA.'], montage: ['Montage', 'Plan de montage, brouillon et aperçu refaits. Aucun appel à l’IA.'] };
  const go = async () => {
    setConfirm(false);
    try {
      const job = await api(`/api/videos/${enc(video.name)}/redo`, { method: 'POST', body: { stage } });
      toast('Reprise lancée', { href: '#/queue/' + job.id });
    } catch (e) { toast(e.message, { tone: 'error' }); }
  };
  return html`<section class="redo card pad" aria-labelledby="redo-title">
    <h2 id="redo-title" class="h3">Refaire à partir de</h2>
    <div class="segmented">${Object.entries(labels).map(([key, [label]]) => html`<label class=${'seg-opt' + (stage === key ? ' is-on' : '')}>
      <input type="radio" class="sr-only" checked=${stage === key} onChange=${() => setStage(key)} />${label}</label>`)}</div>
    <p class="muted small">${labels[stage][1]} Les étapes d'avant sont gardées.</p>
    <${Button} kind="secondary" icon="redo" onClick=${() => (stage === 'analyze' ? setConfirm(true) : go())}>Relancer<//>
    ${confirm && html`<${Modal} title="Réécrire le script ?" subtitle=${video.series + ' · ' + video.episode} onClose=${() => setConfirm(false)}
      footer=${html`<${Button} kind="ghost" onClick=${() => setConfirm(false)}>Annuler<//><${Button} kind="primary" icon="redo" onClick=${go}>Réécrire<//>`}>
      <p>Claude relit toutes les cases et écrit un nouveau script (≈ 1 $ au tarif API, décompté de l'abonnement). La voix et le montage suivent.</p>
    <//>`}
  </section>`;
}

function SceneBar({ scenes, total, time, mediaDuration, onSeek }) {
  if (!scenes.length || !total) return null;
  const limit = mediaDuration || total;
  return html`<div class="scenebar" aria-label="Scènes par émotion">
    <div class="scenebar-track">
      ${scenes.map((s) => {
        const playable = s.start_s < limit - 0.5;
        const current = time >= s.start_s && time < s.start_s + s.duration_s;
        return html`<button type="button" class=${'scenebar-seg' + (current ? ' is-current' : '') + (playable ? '' : ' is-off')}
          style=${{ flexGrow: Math.max(0.2, s.duration_s), background: emotionColor(s.emotion) }} disabled=${!playable}
          aria-label=${`Scène ${s.number}, ${emotionLabel(s.emotion)}, ${fmtDuration(s.start_s)}`} title=${`Scène ${s.number} · ${emotionLabel(s.emotion)} · ${fmtDuration(s.start_s)}`}
          onClick=${() => onSeek(s)}></button>`;
      })}
      <span class="scenebar-head" style=${{ left: `${Math.min(100, (100 * time) / total)}%` }} aria-hidden="true"></span>
    </div>
    ${mediaDuration && mediaDuration < total - 1 ? html`<p class="muted small">Cet aperçu ne couvre que les ${fmtDuration(mediaDuration)} premières : les scènes suivantes sont grisées.</p>` : null}
  </div>`;
}

function Scene({ video, scene, current, onSeek, sceneRef }) {
  return html`<li class=${'scene' + (current ? ' is-current' : '')} ref=${sceneRef}>
    <button type="button" class="scene-head" onClick=${() => onSeek(scene)} aria-label=${`Aller à la scène ${scene.number}`}>
      <span class="scene-n">${scene.number}</span>
      <span class="emo" style=${{ '--emo': emotionColor(scene.emotion) }}>${emotionLabel(scene.emotion)}</span>
      <span class="mono muted small">${fmtDuration(scene.start_s)} · ${Math.round(scene.duration_s)} s</span>
    </button>
    ${scene.images.length > 0 && html`<div class="scene-imgs">${scene.images.slice(0, 5).map((p) => html`<img loading="lazy" alt=""
      src=${`/api/videos/${enc(video.name)}/image?path=${enc(p)}&w=200`} />`)}</div>`}
    ${scene.narration && html`<p class="scene-text">${scene.narration}</p>`}
  </li>`;
}

export function VideoPage({ name, live, query }) {
  const [video, setVideo] = useState(null);
  const [error, setError] = useState(null);
  const [mediaPath, setMediaPath] = useState(null);
  const [time, setTime] = useState(0);
  const [mediaDuration, setMediaDuration] = useState(null);
  const [showRender, setShowRender] = useState(query.get('render') === '1');
  const player = useRef();
  const sceneRefs = useRef({});
  const load = () => api('/api/videos/' + enc(name)).then((v) => { setVideo(v); setError(null); }).catch((e) => setError(e.message));
  useEffect(() => { setVideo(null); load(); }, [name]);
  const jobs = live.jobs.filter((j) => j.target === name);
  const activeJob = jobs.find((j) => j.status === 'running' || j.status === 'queued');
  const lastFinished = jobs.filter((j) => j.status === 'done').map((j) => j.id).join();
  useEffect(() => { if (lastFinished) load(); }, [lastFinished]);
  useEffect(() => { if (video && !video.media.find((m) => m.path === mediaPath)) setMediaPath(video.media[0]?.path || null); }, [video]);

  const scenes = video?.scenes || [];
  const currentScene = useMemo(() => scenes.find((s) => time >= s.start_s && time < s.start_s + s.duration_s), [time, scenes]);
  useEffect(() => {
    const el = currentScene && sceneRefs.current[currentScene.number];
    if (el && !player.current?.paused) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [currentScene?.number]);

  if (error) return html`<div class="page"><a class="back" href="#/"><${Icon} name="arrowL" size=${16} />Rendus</a><p class="alert" role="alert">${error}</p></div>`;
  if (!video) return html`<div class="page"><p class="muted">Chargement…</p></div>`;

  const seek = (scene) => {
    const el = player.current;
    if (!el) return;
    el.currentTime = scene.start_s + 0.05;
    el.play?.().catch(() => {});
  };
  const open = (what) => api(`/api/videos/${enc(name)}/open`, { method: 'POST', body: { what } }).catch((e) => toast(e.message, { tone: 'error' }));
  const failed = video.status === 'failed' || video.status === 'incomplete';
  const retry = async () => {
    try { const job = await api(`/api/videos/${enc(name)}/redo`, { method: 'POST', body: {} }); navigate('#/queue/' + job.id); } catch (e) { toast(e.message, { tone: 'error' }); }
  };
  const meta = [fmtDuration(video.duration_s), video.n_scenes && `${video.n_scenes} scènes`, video.n_panels && `${video.n_panels} cases`,
    video.model && `script ${video.model}`, video.format === 'SHORT' ? 'Short 9:16' : 'Long 16:9'].filter(Boolean).join(' · ');

  return html`<div class="page">
    <nav aria-label="Fil d'Ariane" class="crumbs"><a href="#/">Rendus</a><span aria-hidden="true">/</span><span>${video.series}</span>
      <span aria-hidden="true">/</span><span aria-current="page">${video.episode || video.name}</span></nav>
    <header class="page-head">
      <div class="grow"><h1>${video.series}${video.episode ? html` <span class="muted-strong">· ${video.episode}</span>` : ''}</h1><p class="muted">${meta}</p></div>
      <${IconButton} icon="folder" label="Ouvrir le dossier" onClick=${() => open('folder')} />
      ${video.kdenlive_project && html`<${Button} kind="secondary" icon="open" onClick=${() => open('kdenlive')}>Ouvrir dans Kdenlive<//>`}
      ${video.has_timeline && html`<${Button} kind="primary" icon="export" onClick=${() => setShowRender(true)} disabled=${activeJob?.kind === 'render'}>
        ${video.status === 'final' ? 'Refaire la vidéo finale' : 'Rendre la vidéo finale'}<//>`}
    </header>

    ${activeJob && html`<section class="banner card" aria-live="polite">
      <div class="banner-main">
        <span class="eyebrow eyebrow-blue">${activeJob.status === 'running' ? html`<${Pulse} />` : html`<${Icon} name="hourglass" size=${14} />`}${activeJob.kind === 'render' ? 'Rendu final' : 'Traitement'} ${activeJob.status === 'running' ? 'en cours' : 'en attente'}</span>
        <p class="banner-line"><span>${(activeJob.current || [])[0]}</span><strong><${EtaText} job=${activeJob} /></strong></p>
        ${activeJob.status === 'running' && html`<${ProgressBar} percent=${activeJob.percent} />`}
      </div>
      <${Button} href=${'#/queue/' + activeJob.id} icon="chevR">Détail<//>
    </section>`}

    ${(video.broken || []).length > 0 && !activeJob && html`<section class="card pad failed-box" aria-label="Rendu illisible">
      <p class="alert" role="alert"><${Icon} name="alert" />
        Un rendu s'est arrêté avant la fin : ${video.broken.map((b) => `« ${b.label} » (${(b.size / 1e9).toFixed(1).replace('.', ',')} Go, ${new Date(b.mtime * 1000).toLocaleString('fr-FR', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })})`).join(', ')} ne se lit pas.
        Relancez le rendu : le fichier sera remplacé, et les longues vidéos se rendent maintenant par parties.</p>
      ${video.has_timeline && html`<${Button} kind="primary" icon="export" onClick=${() => setShowRender(true)}>Relancer le rendu final<//>`}
    </section>`}
    ${failed ? html`<section class="card pad failed-box">
      <p class="alert" role="alert"><${Icon} name="alert" />${video.error ? shortError(video.error) : 'Le traitement s’est arrêté avant le montage.'}</p>
      ${video.error && html`<details class="cmd"><summary>Message complet</summary><code>${video.error}</code></details>`}
      <${Button} kind="primary" icon="redo" onClick=${retry} disabled=${!video.url || !!activeJob}>Relancer ce chapitre<//>
    </section>` : html`<div class="video-layout">
      <div class="video-main">
        ${video.media.length > 0 ? html`
          ${video.media.length > 1 && html`<fieldset class="chips"><legend class="sr-only">Version</legend>
            ${video.media.map((m) => html`<button type="button" class=${'fchip' + (m.path === mediaPath ? ' is-on' : '')} aria-pressed=${m.path === mediaPath}
              onClick=${() => { setMediaPath(m.path); setMediaDuration(null); }}>${m.label}</button>`)}</fieldset>`}
          <video class="player" ref=${player} controls preload="metadata" key=${mediaPath} poster=${`/api/videos/${enc(name)}/thumb`}
            src=${mediaPath ? `/api/videos/${enc(name)}/file?path=${enc(mediaPath)}` : undefined}
            onTimeUpdate=${(e) => setTime(e.currentTarget.currentTime)} onLoadedMetadata=${(e) => setMediaDuration(e.currentTarget.duration)}></video>
        ` : html`<div class="player player-empty"><p>Pas encore de vidéo : l'aperçu n'a pas été rendu.</p></div>`}
        <${SceneBar} scenes=${scenes} total=${video.duration_s} time=${time} mediaDuration=${mediaDuration} onSeek=${seek} />
        <div class="legend">${[...new Set(scenes.map((s) => s.emotion).filter(Boolean))].map((e) => html`<span><i class="pill-key" style=${{ background: emotionColor(e) }}></i>${emotionLabel(e)}</span>`)}</div>
        ${video.kind === 'chapter' && html`<${RedoBox} video=${video} />`}
      </div>
      <section class="scenes card" aria-labelledby="scenes-title">
        <div class="scenes-head"><h2 id="scenes-title" class="h3">Scènes</h2><span class="muted small">${scenes.length}</span>
          ${currentScene && html`<span class="muted small">· lecture : scène ${currentScene.number}</span>`}</div>
        <ol class="scene-list">${scenes.map((s) => html`<${Scene} key=${s.number} video=${video} scene=${s} current=${currentScene?.number === s.number}
          onSeek=${seek} sceneRef=${(el) => { sceneRefs.current[s.number] = el; }} />`)}</ol>
      </section>
    </div>`}
    ${showRender && html`<${RenderDialog} video=${video} onClose=${(launched) => { setShowRender(false); if (query.get('render')) history.replaceState(null, '', '#/video/' + enc(name)); }} />`}
  </div>`;
}
