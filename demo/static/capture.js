/* Phone capture logic.
 *
 * Two paths, deliberately:
 *  1. <input capture>  - hands off to the native camera app. Works over plain
 *     HTTP on Android AND iOS, which is why it is the default for a LAN demo.
 *  2. getUserMedia     - in-page live viewfinder. Requires a secure context
 *     (HTTPS or localhost), so it is only offered when available.
 */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const viewSel = $("view");
  const statusEl = $("status");

  const saved = localStorage.getItem("ahadi.view");
  if (saved && [...viewSel.options].some((o) => o.value === saved)) viewSel.value = saved;
  viewSel.addEventListener("change", () => localStorage.setItem("ahadi.view", viewSel.value));

  let tick = null;
  function showStatus(text, busy) {
    statusEl.hidden = false;
    statusEl.className = "status" + (busy ? " busy" : "");
    if (tick) { clearInterval(tick); tick = null; }
    if (!busy) { statusEl.textContent = text; return; }
    // A vision call legitimately takes 10-60s. Without a running clock the page
    // looks frozen and people reload it, which is what "I think it timed out"
    // usually is.
    const t0 = Date.now();
    statusEl.textContent = text + " ... 0s";
    tick = setInterval(() => {
      const s = Math.round((Date.now() - t0) / 1000);
      statusEl.textContent = text + " ... " + s + "s" +
        (s >= 25 ? " (still waiting on the AI - this can take a minute)" : "");
    }, 1000);
  }

  // ---- staging: every angle is saved first, then counted in one pass --------
  const VIEWS = window.VIEWS || [];
  const LABELS = window.VIEW_LABELS || {};
  // Restored from the server so a reload or a dropped phone connection does not
  // silently lose photos that are already staged.
  const STAGED0 = window.STAGED || {};
  const stagedViews = new Set(Object.keys(STAGED0));
  let stagedCount = Object.values(STAGED0).reduce((a, b) => a + b, 0);

  function paintStaged() {
    const box = $("staged");
    if (!box) return;
    box.hidden = stagedCount === 0;
    if (!stagedCount) return;
    $("stagedcount").textContent = stagedCount;
    $("analyse").textContent =
      "Count and compare all " + stagedCount + " photo" + (stagedCount > 1 ? "s" : "");
    const bar = $("anglebtns");
    bar.innerHTML = "";
    for (const v of VIEWS.filter((v) => !stagedViews.has(v))) {
      const b = document.createElement("button");
      b.className = "btn ghost small";
      b.type = "button";
      b.textContent = "+ " + (LABELS[v] || v);
      b.addEventListener("click", () => {
        viewSel.value = v;
        localStorage.setItem("ahadi.view", v);
        $("file").click();
      });
      bar.appendChild(b);
    }
    $("askangles").hidden = bar.children.length === 0;
  }
  paintStaged();

  async function send(blob, filename) {
    const fd = new FormData();
    fd.append("file", blob, filename);
    fd.append("shelf", window.SHELF_ID);
    fd.append("view", viewSel.value);
    fd.append("source", "camera");
    fd.append("analyse", "0");
    showStatus("Saving photo…", true);
    let res, data;
    try {
      res = await fetch("/api/capture", { method: "POST", body: fd });
      data = await res.json();
    } catch (e) {
      showStatus("Upload failed — is the server still running?", false);
      return;
    }
    if (!data.ok) {
      showStatus("Could not save: " + (data.error || res.status), false);
      return;
    }
    stagedViews.add(data.view);
    stagedCount = data.staged ?? stagedCount + 1;
    paintStaged();
    const left = VIEWS.filter((v) => !stagedViews.has(v));
    showStatus(
      left.length
        ? `Saved ${data.view_label}. ${left.length} angle(s) still to go.`
        : `Saved ${data.view_label}. That is every angle — now count them.`,
      false
    );
  }

  if ($("analyse")) {
    window.AhadiAnalyse.wire($("analyse"), statusEl, window.SHELF_ID);
    $("discard").addEventListener("click", () => {
      if (!confirm("Discard the staged photos? Nothing has been counted yet.")) return;
      fetch(`/shelves/${window.SHELF_ID}/staged/discard`, { method: "POST" })
        .then(() => location.reload());
    });
  }

  // ---- clear this angle so a new baseline can be set -----------------------
  const clearBtn = $("clearview");
  if (clearBtn) {
    clearBtn.addEventListener("click", async () => {
      const label = viewSel.options[viewSel.selectedIndex].text;
      if (!confirm(`Delete all photos for "${label}"? The next photo becomes the new baseline.`)) return;
      showStatus("Clearing…", true);
      try {
        const r = await fetch(`/api/views/${window.SHELF_ID}/${viewSel.value}/clear`, { method: "POST" });
        const j = await r.json();
        if (!r.ok) throw new Error(j.detail || r.status);
        showStatus(`Cleared ${j.removed} photo(s). Take the next photo to set a new baseline.`, false);
      } catch (e) {
        showStatus("Could not clear: " + e.message, false);
      }
    });
  }

  // ---- path 1: native camera app -------------------------------------------
  $("file").addEventListener("change", (ev) => {
    const f = ev.target.files && ev.target.files[0];
    if (!f) return;
    send(f, f.name || "capture.jpg");
    ev.target.value = "";
  });

  // ---- path 2: in-page live viewfinder -------------------------------------
  const liveBtn = $("live");
  if (liveBtn) {
    let stream = null;
    const box = $("livebox"), video = $("video"), canvas = $("canvas");

    liveBtn.addEventListener("click", async () => {
      box.hidden = false;
      showStatus("Starting camera…", true);
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 } },
          audio: false,
        });
        video.srcObject = stream;
        await video.play();
        statusEl.hidden = true;
      } catch (e) {
        showStatus("Camera blocked: " + e.name + " — use 'Take photo' instead.", false);
        box.hidden = true;
      }
    });

    $("shutter").addEventListener("click", () => {
      if (!stream) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);
      canvas.toBlob((b) => { if (b) send(b, "live.jpg"); }, "image/jpeg", 0.9);
    });

    $("closelive").addEventListener("click", () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
      stream = null;
      box.hidden = true;
      statusEl.hidden = true;
    });
  }
})();
