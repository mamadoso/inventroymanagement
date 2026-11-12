/* "Count and compare all", shared by the phone capture screen and the desktop
 * shelf page so the two mean exactly the same thing.
 *
 * The press starts a job on the server and this polls it. Three angles of real
 * vision is 30-75 seconds, which no proxy will sit and wait for, so the browser
 * asks instead of hanging. The shot statuses in the database are the truth; if
 * the process dies mid-job the poll stops reporting it as running.
 */
(function () {
  "use strict";

  function wire(button, statusEl, shelfId) {
    if (!button) return;
    button.addEventListener("click", async () => {
      button.disabled = true;
      const t0 = Date.now();
      const say = (msg) => {
        if (statusEl) {
          statusEl.hidden = false;
          statusEl.className = "status busy";
          statusEl.textContent = msg;
        }
      };
      say("Starting…");
      let start;
      try {
        const res = await fetch(`/shelves/${shelfId}/analyse`, { method: "POST" });
        start = await res.json();
      } catch (e) {
        say("Could not reach the server — is it still running?");
        button.disabled = false;
        return;
      }
      if (start.complete || !start.started) {
        location.href = start.redirect || `/shelves/${shelfId}`;
        return;
      }
      const total = start.started;
      const timer = setInterval(async () => {
        let p;
        try {
          p = await (await fetch(`/api/analyse/${shelfId}/progress`)).json();
        } catch (e) {
          return;  // transient; the next tick tries again
        }
        const secs = Math.round((Date.now() - t0) / 1000);
        const done = p.done ?? 0;
        say(`Counting ${total} photo(s)… ${done}/${total} · ${secs}s` +
            (p.failed ? ` · ${p.failed} failed` : ""));
        if (p.complete) {
          clearInterval(timer);
          location.href = `/shelves/${shelfId}/report`;
        }
      }, 1500);
    });
  }

  window.AhadiAnalyse = { wire: wire };
})();
