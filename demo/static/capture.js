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
  const resultEl = $("result");

  const saved = localStorage.getItem("ahadi.view");
  if (saved && [...viewSel.options].some((o) => o.value === saved)) viewSel.value = saved;
  viewSel.addEventListener("change", () => localStorage.setItem("ahadi.view", viewSel.value));

  function showStatus(text, busy) {
    statusEl.hidden = false;
    resultEl.hidden = true;
    statusEl.className = "status" + (busy ? " busy" : "");
    statusEl.textContent = text;
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
    );
  }

  async function send(blob, filename) {
    const fd = new FormData();
    fd.append("file", blob, filename);
    fd.append("shelf", window.SHELF_ID);
    fd.append("view", viewSel.value);
    fd.append("source", "camera");
    showStatus("Analysing photo…", true);
    let res, data;
    try {
      res = await fetch("/api/capture", { method: "POST", body: fd });
      data = await res.json();
    } catch (e) {
      showStatus("Upload failed — is the server still running?", false);
      return;
    }
    if (!data.ok) {
      showStatus("Could not analyse: " + (data.error || res.status), false);
      return;
    }
    statusEl.hidden = true;
    render(data);
  }

  function render(data) {
    const s = data.summary || {};
    const c = s.counts || {};
    let cls = "ok", head = data.kind === "baseline"
      ? `Baseline set — ${data.total} watches`
      : (c.missing ? `${c.missing} MISSING` : (c.new || c.swapped ? "Changed" : (c.moved ? "Moved only" : "No change")));

    if (data.kind !== "baseline") {
      cls = c.missing ? "bad" : (c.new || c.swapped || c.moved ? "warn" : "ok");
    }

    let html = `<div class="headline ${cls}">
      <div class="big-count">${data.kind === "baseline" ? "" :
        `<span class="from">${s.base_total}</span><span class="arrow">→</span><span class="to">${s.cap_total}</span><span class="unit">watches</span>`}</div>
      <div class="headline-text">${esc(head)}</div>
      <div class="muted small">${esc(viewSel.options[viewSel.selectedIndex].text)} · ${data.latency_ms || 0} ms</div>
    </div><div class="card">`;

    if (data.kind === "baseline") {
      html += `<p>Now take a second photo after removing a watch to see the diff.</p>`;
    } else if (!s.changes || !s.changes.length) {
      html += `<p class="pill ok">Nothing changed on this angle.</p>`;
    } else {
      html += `<ul>`;
      for (const ch of s.changes) {
        html += `<li><span class="pill ${ch.kind === "missing" ? "bad" : ch.kind === "new" ? "new" : "warn"}">${esc(ch.kind)}</span>
                  position <b>#${ch.n}</b>${ch.to ? ` → <b>#${ch.to}</b>` : ""}
                  <span class="muted small">${esc(ch.desc)}</span></li>`;
      }
      html += `</ul>`;
    }
    if (data.quality && data.quality !== "good") {
      html += `<p class="pill warn">photo quality: ${esc(data.quality)} — consider a retake</p>`;
    }
    if (data.diff_id) {
      html += `<p><a class="btn small" href="/diffs/${data.diff_id}">Open full report with annotated photos</a></p>`;
    }
    html += `</div>`;
    resultEl.innerHTML = html;
    resultEl.hidden = false;
    resultEl.scrollIntoView({ behavior: "smooth", block: "start" });
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
