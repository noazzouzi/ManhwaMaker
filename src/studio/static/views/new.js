// Nouvelle vidéo : un lien, une plage de chapitres, quelques choix ; l'estimation suit chaque réglage.
import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import { Button, Icon, api, fmtClock, fmtRemaining, html, navigate, toast } from '../ui.js';

const VOICES = [['am_fenrir,am_michael', 'Fenrir + Michael (mélange)'], ['am_fenrir', 'Fenrir'], ['am_michael', 'Michael'], ['am_puck', 'Puck']];

function Section({ n, title, hint, children }) {
  return html`<section class="form-section" aria-labelledby=${'sec-' + n}>
    <div class="section-head"><span class="step-n">${n}</span><h2 id=${'sec-' + n}>${title}</h2>${hint && html`<span class="muted">${hint}</span>`}</div>
    ${children}
  </section>`;
}

function Choice({ name, checked, onChange, icon, title, desc }) {
  return html`<label class=${'choice' + (checked ? ' is-on' : '')}>
    <input type="radio" name=${name} checked=${checked} onChange=${onChange} />
    <${Icon} name=${icon} size=${20} />
    <span class="choice-text"><strong>${title}</strong><span>${desc}</span></span>
  </label>`;
}

function Check({ checked, onChange, title, desc }) {
  return html`<label class="check"><input type="checkbox" checked=${checked} onChange=${onChange} />
    <span><strong>${title}</strong>${desc && html`<span>${desc}</span>`}</span></label>`;
}

export function NewVideo({ query }) {
  const [url, setUrl] = useState(query.get('url') || '');
  const [info, setInfo] = useState(null);
  const [inspecting, setInspecting] = useState(false);
  const [inspectError, setInspectError] = useState(null);
  const [start, setStart] = useState(null);
  const [end, setEnd] = useState(null);
  const [format, setFormat] = useState('LONG');
  const [compile, setCompile] = useState(true);
  const [keep, setKeep] = useState(false);
  const [voice, setVoice] = useState(VOICES[0][0]);
  const [speed, setSpeed] = useState(1.2);
  const [adv, setAdv] = useState(false);
  const [scriptAi, setScriptAi] = useState('claude');
  const [language, setLanguage] = useState('en');
  const [cta, setCta] = useState(true);
  const [bgm, setBgm] = useState(true);
  const [sfx, setSfx] = useState(true);
  const [force, setForce] = useState(false);
  const [estimate, setEstimate] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const timer = useRef();

  // Lecture de la série dès que le lien est complet.
  useEffect(() => {
    clearTimeout(timer.current);
    setInspectError(null);
    if (!/^https:\/\/\S+\.\S+\/\S+/.test(url.trim())) { setInfo(null); return; }
    timer.current = setTimeout(async () => {
      setInspecting(true);
      try {
        const data = await api('/api/inspect?url=' + encodeURIComponent(url.trim()));
        setInfo(data);
        const known = new Set([...data.done, ...data.compiled]);
        const eps = data.episodes;
        const firstNew = data.is_series ? (eps.find((n) => !known.has(n)) ?? eps[0]) : (data.current ?? eps[0]);
        const from = Math.floor(firstNew ?? 1);
        setStart(from);
        const last = eps.length ? Math.floor(eps[eps.length - 1]) : from;
        setEnd(data.is_series ? Math.min(last, from + 19) : from);
      } catch (e) {
        setInfo(null);
        setInspectError(e.message);
      } finally { setInspecting(false); }
    }, 500);
    return () => clearTimeout(timer.current);
  }, [url]);

  const episodes = info?.episodes || [];
  const inRange = useMemo(() => episodes.filter((n) => start != null && end != null && n >= start && n <= end), [episodes, start, end]);
  const count = inRange.length || (start != null && end != null && end >= start ? end - start + 1 : 0);
  const done = new Set(info?.done || []);
  const compiled = new Set(info?.compiled || []);
  const failed = new Set(info?.failed || []);
  const doneInRange = inRange.filter((n) => done.has(n) || compiled.has(n));
  const compiledInRange = inRange.filter((n) => compiled.has(n));
  const toProcess = force ? count : count - doneInRange.length;

  useEffect(() => {
    if (!count) { setEstimate(null); return; }
    const id = setTimeout(() => api('/api/estimate', { method: 'POST', body: { count: Math.max(1, toProcess), compile } })
      .then(setEstimate).catch(() => setEstimate(null)), 250);
    return () => clearTimeout(id);
  }, [toProcess, compile]);

  const setPreset = (n) => { if (start != null) setEnd(Math.min(start + n - 1, Math.floor(episodes[episodes.length - 1] ?? start + n - 1))); };
  const last = episodes.length ? Math.floor(episodes[episodes.length - 1]) : null;

  const submit = async () => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const job = await api('/api/jobs', { method: 'POST', body: {
        url: url.trim(), start, end, format, compile, keep_chapters: keep, voice, speed, script_ai: scriptAi, language, cta, bgm, sfx, force,
        series: info?.series,
      } });
      toast('Traitement lancé', { href: '#/queue/' + job.id });
      navigate('#/queue/' + job.id);
    } catch (e) {
      setSubmitError(e.message);
    } finally { setSubmitting(false); }
  };

  const pillState = (n) => (compiled.has(n) ? 'compiled' : done.has(n) ? 'done' : failed.has(n) ? 'failed' : (n >= start && n <= end) ? 'on' : 'off');
  const canSubmit = info && count > 0 && !submitting && (toProcess > 0 || compile);
  const total = estimate ? estimate.seconds + (estimate.wait_s || 0) : null;

  return html`<div class="page">
    <a class="back" href="#/"><${Icon} name="arrowL" size=${16} />Rendus</a>
    <header class="page-head"><div class="grow"><h1>Nouvelle vidéo</h1>
      <p class="muted">Collez un lien, choisissez les chapitres, lancez. Le traitement tourne sur le serveur : vous pouvez fermer cette page.</p></div></header>
    <div class="form-layout">
      <div class="form">
        <${Section} n="1" title="Source">
          <label class="field-label" for="url">Lien de la série ou d'un chapitre</label>
          <div class=${'input-url' + (inspectError ? ' has-error' : '')}>
            <${Icon} name="link" />
            <input id="url" type="url" placeholder="https://asurascans.com/comics/… ou https://www.webtoons.com/en/…" value=${url}
              onInput=${(e) => setUrl(e.target.value)} autocomplete="off" spellcheck="false" aria-describedby="url-help" />
          </div>
          <p id="url-help" class="help">Webtoons ou Asura Scans. Un lien de série donne accès à tous ses chapitres.</p>
          ${inspecting && html`<p class="help" role="status"><span class="spinner" aria-hidden="true"></span>Lecture de la liste des chapitres…</p>`}
          ${inspectError && html`<p class="alert" role="alert"><${Icon} name="alert" />${inspectError}</p>`}
          ${info && html`<div class="series card">
            <div class="grow"><strong>${info.series}</strong>
              <span class="muted">${info.site} · ${episodes.length} chapitre${episodes.length > 1 ? 's' : ''} disponible${episodes.length > 1 ? 's' : ''}
              ${info.done.length + info.compiled.length ? ` · ${info.done.length + info.compiled.length} déjà fait${info.done.length + info.compiled.length > 1 ? 's' : ''}` : ''}</span></div>
            <span class="ok-text"><${Icon} name="check" size=${16} />Série trouvée</span>
          </div>`}
        <//>

        <${Section} n="2" title="Chapitres" hint=${count ? `${count} chapitre${count > 1 ? 's' : ''}` : ''}>
          <div class="row-end">
            <label class="field-label col">Du chapitre
              <input class="input num" type="number" min=${episodes[0] ?? 0} value=${start ?? ''} disabled=${!info}
                onInput=${(e) => { const v = parseInt(e.target.value, 10); if (!Number.isNaN(v)) { setStart(v); if (end != null && end < v) setEnd(v); } }} /></label>
            <label class="field-label col">au chapitre
              <input class="input num" type="number" min=${start ?? 0} value=${end ?? ''} disabled=${!info}
                onInput=${(e) => { const v = parseInt(e.target.value, 10); if (!Number.isNaN(v)) setEnd(v); }} /></label>
            <fieldset class="chips"><legend class="sr-only">Raccourcis</legend>
              ${[5, 10, 20].map((n) => html`<button type="button" class=${'fchip' + (count === n ? ' is-on' : '')} aria-pressed=${count === n}
                disabled=${!info} onClick=${() => setPreset(n)}>${n} chapitres</button>`)}
              ${last != null && html`<button type="button" class=${'fchip' + (end === last ? ' is-on' : '')} aria-pressed=${end === last}
                disabled=${!info} onClick=${() => setEnd(last)}>Jusqu'au dernier</button>`}
            </fieldset>
          </div>
          ${episodes.length > 0 && episodes.length <= 160 && html`<div class="pills" aria-hidden="true">
            ${episodes.map((n) => html`<span class=${'pill pill-' + pillState(n)} title=${'Chapitre ' + n}>${n}</span>`)}
          </div>
          <div class="legend"><span><i class="pill-key pill-on"></i>à traiter</span><span><i class="pill-key pill-done"></i>déjà fait</span>
            <span><i class="pill-key pill-compiled"></i>fait puis compilé</span><span><i class="pill-key pill-failed"></i>en échec</span></div>`}
          ${doneInRange.length > 0 && html`<div class="note">
            <p>${doneInRange.length} chapitre${doneInRange.length > 1 ? 's' : ''} de la plage ${doneInRange.length > 1 ? 'sont' : 'est'} déjà fait${doneInRange.length > 1 ? 's' : ''} :
              ${force ? ' ils seront refaits entièrement (nouveau script, nouveau coût).' : ' ils sont repris tels quels, sans nouveau coût.'}
              ${compiledInRange.length > 0 && compile && !force ? ` Attention : ${compiledInRange.length} ont été compilés et leurs dossiers supprimés ; la compilation échouera sans les refaire.` : ''}</p>
            <${Check} checked=${force} onChange=${() => setForce(!force)} title="Refaire aussi les chapitres déjà faits" />
          </div>`}
        <//>

        <${Section} n="3" title="Format et sortie">
          <div class="two-cols">
            <fieldset class="choices"><legend class="field-label">Format</legend>
              <${Choice} name="fmt" checked=${format === 'LONG'} onChange=${() => setFormat('LONG')} icon="monitor" title="Long · 16:9" desc="YouTube · 1920 × 1080" />
              <${Choice} name="fmt" checked=${format === 'SHORT'} onChange=${() => setFormat('SHORT')} icon="phone" title="Short · 9:16" desc="TikTok, Shorts · recadré, coupes rapides" />
            </fieldset>
            <fieldset class="choices"><legend class="field-label">Sortie</legend>
              <${Choice} name="out" checked=${compile} onChange=${() => setCompile(true)} icon="layers" title="Une seule vidéo" desc="Chapitres enchaînés en une compilation" />
              <${Choice} name="out" checked=${!compile} onChange=${() => setCompile(false)} icon="copies" title="Une vidéo par chapitre" desc="Un aperçu et un projet par chapitre" />
            </fieldset>
          </div>
          ${compile && html`<${Check} checked=${keep} onChange=${() => setKeep(!keep)} title="Garder les dossiers de chapitres"
            desc="Sinon ils sont supprimés après la compilation, qui reste complète." />`}
        <//>

        <${Section} n="4" title="Voix">
          <div class="voice-row">
            <label class="field-label col">Voix
              <select class="input" value=${voice} onChange=${(e) => setVoice(e.target.value)}>
                ${VOICES.map(([v, label]) => html`<option value=${v}>${label}</option>`)}
              </select></label>
            <label class="field-label col"><span class="row-between">Vitesse<strong class="mono">${speed.toFixed(1).replace('.', ',')}×</strong></span>
              <input type="range" min="1" max="1.5" step="0.1" value=${speed} onInput=${(e) => setSpeed(parseFloat(e.target.value))} /></label>
          </div>
        <//>

        <div class="advanced">
          <button type="button" class="disclosure" aria-expanded=${adv} onClick=${() => setAdv(!adv)}>
            <${Icon} name=${adv ? 'chevD' : 'chevR'} />Options avancées<span class="muted">IA du script, langue, appel à l'abonnement, musique, bruitages</span></button>
          ${adv && html`<div class="adv-grid">
            <label class="field-label col">IA du script
              <select class="input" value=${scriptAi} onChange=${(e) => setScriptAi(e.target.value)}>
                <option value="claude">Claude Opus 5.5 (Gemini en repli)</option><option value="gemini">Gemini seul</option></select></label>
            <label class="field-label col">Langue de la narration
              <select class="input" value=${language} onChange=${(e) => setLanguage(e.target.value)}>
                <option value="en">Anglais</option><option value="fr">Français</option></select></label>
            <${Check} checked=${cta} onChange=${() => setCta(!cta)} title="Appel à l'abonnement" desc="Inséré vers 40 % du script" />
            <${Check} checked=${bgm} onChange=${() => setBgm(!bgm)} title="Musique de fond" desc="Bouclée, à −22 dB" />
            <${Check} checked=${sfx} onChange=${() => setSfx(!sfx)} title="Bruitages" desc="Sur les scènes d'action" />
          </div>`}
        </div>
      </div>

      <aside class="summary card" aria-labelledby="recap">
        <h2 id="recap" class="eyebrow">Récapitulatif</h2>
        <dl class="recap">
          <dt>Série</dt><dd>${info?.series || '—'}</dd>
          <dt>Chapitres</dt><dd>${count ? (start === end ? `n° ${start}` : `${start} à ${end} (${count})`) : '—'}</dd>
          <dt>À traiter</dt><dd>${count ? `${toProcess} chapitre${toProcess > 1 ? 's' : ''}` : '—'}</dd>
          <dt>Format</dt><dd>${format === 'LONG' ? 'Long · 16:9' : 'Short · 9:16'}</dd>
          <dt>Sortie</dt><dd>${compile ? 'Une seule vidéo' : `${count || '—'} vidéo${count > 1 ? 's' : ''}`}</dd>
          <dt>Voix</dt><dd>${VOICES.find((v) => v[0] === voice)[1].split(' (')[0]} · ${speed.toFixed(1).replace('.', ',')}×</dd>
        </dl>
        <div class="divider"></div>
        <div class="estimate">
          <span class="muted">Temps de traitement estimé</span>
          <strong class="big">${estimate && count ? fmtRemaining(estimate.seconds).replace('≈ ', '≈ ') : '—'}</strong>
          ${estimate && count ? html`<span class="muted small">${estimate.wait_s > 30 ? `Démarre après le traitement en cours (${fmtRemaining(estimate.wait_s)}), ` : ''}fin vers ${fmtClock(Date.now() / 1000 + total)}</span>
            <span class="muted small">Estimation ${estimate.basis}.</span>` : null}
        </div>
        ${scriptAi === 'claude' && toProcess > 0 && html`<p class="small muted">Usage Claude ≈ ${Math.round(1.04 * toProcess)} $ au tarif API, décompté de l'abonnement.</p>`}
        ${submitError && html`<p class="alert" role="alert"><${Icon} name="alert" />${submitError}</p>`}
        <${Button} kind="primary" icon="arrowR" className="btn-lg btn-block" disabled=${!canSubmit} onClick=${submit}>
          ${submitting ? 'Lancement…' : count ? `Lancer ${count} chapitre${count > 1 ? 's' : ''}` : 'Lancer'}<//>
        <p class="small muted">Le suivi s'affiche aussitôt : étape en cours, temps restant, journal.</p>
      </aside>
    </div>
  </div>`;
}
