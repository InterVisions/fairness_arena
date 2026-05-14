/**
 * Minimal i18n module for Fairness Arena.
 *
 * Usage:
 *   await I18n.init(['en','es','ca'], 'en');
 *   I18n.t('ui.vote_left');          // translated UI string
 *   I18n.queryLabel('nurse');        // translated query label (display only)
 *   I18n.queryKey('nurse');          // canonical EN key sent to server (always EN)
 *   I18n.onLanguageChange(callback); // called after each language switch
 */

const I18n = (() => {
  let _current = 'en';
  let _translations = {};   // lang -> parsed JSON
  let _callbacks = [];
  let _langs = ['en'];

  async function _load(lang) {
    if (_translations[lang]) return;
    const res = await fetch(`/static/i18n/${lang}.json`);
    if (!res.ok) throw new Error(`i18n: failed to load ${lang}.json`);
    _translations[lang] = await res.json();
  }

  async function init(langs, defaultLang) {
    _langs = langs;
    const saved = localStorage.getItem('arena_lang');
    _current = langs.includes(saved) ? saved : defaultLang;
    await Promise.all(langs.map(_load));
    _apply();
  }

  function t(key) {
    const parts = key.split('.');
    let val = _translations[_current];
    for (const p of parts) val = val?.[p];
    if (val === undefined) {
      // fallback to English
      val = _translations['en'];
      for (const p of parts) val = val?.[p];
    }
    return val ?? key;
  }

  /** Display label for a query (translated, gender-neutral). */
  function queryLabel(enKey) {
    return _translations[_current]?.queries?.[enKey] ?? enKey;
  }

  /** Canonical key always sent to the server — always English. */
  function queryKey(enKey) {
    return enKey;
  }

  /** All query canonical keys (English). */
  function queryKeys() {
    return Object.keys(_translations['en']?.queries ?? {});
  }

  function currentLang() { return _current; }
  function availableLangs() { return _langs; }

  function onLanguageChange(cb) { _callbacks.push(cb); }

  async function switchTo(lang) {
    if (!_langs.includes(lang)) return;
    await _load(lang);
    _current = lang;
    localStorage.setItem('arena_lang', lang);
    _apply();
    _callbacks.forEach(cb => cb(lang));
  }

  function _apply() {
    // Update all elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      const attr = el.getAttribute('data-i18n-attr');
      const val = t(key);
      if (attr) {
        el.setAttribute(attr, val);
      } else {
        el.textContent = val;
      }
    });
    // Update lang switcher button states
    document.querySelectorAll('.lang-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.lang === _current);
    });
    // Update <html lang>
    document.documentElement.lang = _current;
  }

  return { init, t, queryLabel, queryKey, queryKeys, currentLang, availableLangs, switchTo, onLanguageChange };
})();
