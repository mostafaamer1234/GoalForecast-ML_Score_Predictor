// GoalForecast matchday: live state from /api/live, model parameters from data.json
// (backtest/export_site_data.py). Falls back to the bundled 2024/25 state if live data is down.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const pct = (p) => `${Math.round(p * 100)}%`;
  let M = null, L = null, TEAMS = {}, FORM = {};

  function predict(home, away) {
    const h = TEAMS[home], a = TEAMS[away];
    const v = { elo_diff: h.elo + M.home_advantage - a.elo, home_elo: h.elo, away_elo: a.elo, home_gf5: h.gf5, home_ga5: h.ga5,
      away_gf5: a.gf5, away_ga5: a.ga5, home_pts5: h.pts5, away_pts5: a.pts5 };
    const x = M.features.map((f, i) => (v[f] - M.scaler.mean[i]) / M.scaler.scale[i]);
    const z = M.coef.map((w, k) => w.reduce((s, wi, i) => s + wi * x[i], M.intercept[k]));
    const top = Math.max(...z), e = z.map((t) => Math.exp(t - top)), sum = e.reduce((s, t) => s + t, 0);
    const out = {};
    M.class_order.forEach((c, k) => (out[c] = e[k] / sum));
    return out;
  }

  const dots = (team) => `<span class="dots">${(FORM[team] || []).map((r) => `<i class="${r}">${r}</i>`).join("") || '<span style="color:var(--ink-3)">–</span>'}</span>`;
  const when = (d, t) => {
    const dt = new Date(`${d}T${t || "15:00"}:00Z`);
    return dt.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short", timeZone: "Europe/London" }) + (t ? ` · ${t} UK` : "");
  };

  function fixtures() {
    const list = (L && L.fixtures) || [];
    if (!list.length) {
      $("fixtures").innerHTML = `<p style="color:var(--ink-2)">${L ? "No Premier League fixtures are published for the next few days. Try any matchup below." : "Live fixtures are unavailable right now. You can still price any matchup below."}</p>`;
      return;
    }
    $("fixtures").innerHTML = list.map((f) => {
      const p = f.model, m = f.market;
      let edge = "";
      if (m) {
        const diffs = ["H", "D", "A"].map((c) => [c, p[c] - m[c]]).sort((x, y) => y[1] - x[1]);
        if (diffs[0][1] > 0.06) edge = `<div class="edge">Model rates ${diffs[0][0] === "H" ? esc(f.home) : diffs[0][0] === "A" ? esc(f.away) : "the draw"} ${Math.round(diffs[0][1] * 100)} points higher than the market</div>`;
      }
      return `<button class="fx" type="button" data-h="${esc(f.home)}" data-a="${esc(f.away)}">
        <div class="when">${esc(when(f.date, f.time))}</div>
        <div class="teams"><span>${esc(f.home)}</span><span class="v">v</span><span>${esc(f.away)}</span></div>
        <div class="probs" aria-label="Home ${pct(p.H)}, draw ${pct(p.D)}, away ${pct(p.A)}"><i class="h" style="flex:${p.H}">${pct(p.H)}</i><i class="d" style="flex:${p.D}">${pct(p.D)}</i><i class="a" style="flex:${p.A}">${pct(p.A)}</i></div>
        ${m ? `<div class="mkt"><span>Bookmakers</span><b>${pct(m.H)} · ${pct(m.D)} · ${pct(m.A)}</b></div>` : ""}${edge}
      </button>`;
    }).join("");
  }

  function scoreboard() {
    const home = $("home").value, away = $("away").value;
    if (!home || home === away) return;
    const p = predict(home, away), h = TEAMS[home], a = TEAMS[away];
    $("p-h").textContent = pct(p.H); $("p-d").textContent = pct(p.D); $("p-a").textContent = pct(p.A);
    $("l-h").textContent = `${home} win`; $("l-a").textContent = `${away} win`;
    $("b-h").style.flex = p.H; $("b-d").style.flex = p.D; $("b-a").style.flex = p.A;
    const row = (x, label, y) => `<tr><td>${x}</td><td>${label}</td><td>${y}</td></tr>`;
    $("compare").innerHTML = row(Math.round(h.elo), "Elo rating", Math.round(a.elo)) + row(h.pts5.toFixed(1), "Points per game, last 5", a.pts5.toFixed(1)) +
      row(h.gf5.toFixed(1), "Scored per game", a.gf5.toFixed(1)) + row(h.ga5.toFixed(1), "Conceded per game", a.ga5.toFixed(1)) +
      (Object.keys(FORM).length ? `<tr><td>${dots(home)}</td><td>Last five</td><td>${dots(away)}</td></tr>` : "");
    $("fair").textContent = `Fair decimal odds ${(1 / p.H).toFixed(2)} · ${(1 / p.D).toFixed(2)} · ${(1 / p.A).toFixed(2)}`;
    history.replaceState(null, "", `#${encodeURIComponent(home)}-v-${encodeURIComponent(away)}`);
  }

  function table() {
    const rows = (L && L.table) || [];
    if (!rows.length) { $("league").innerHTML = `<tr><td colspan="10" class="l" style="color:var(--ink-2)">The live table is unavailable right now.</td></tr>`; return; }
    $("league").innerHTML = rows.map((t, i) => `<tr data-team="${esc(t.team)}"><td>${i + 1}</td><td class="l club">${esc(t.team)}</td><td>${t.p}</td><td>${t.w}</td><td>${t.d}</td><td>${t.l}</td>
      <td>${t.gf - t.ga > 0 ? "+" : ""}${t.gf - t.ga}</td><td class="pts">${t.pts}</td><td>${Math.round(TEAMS[t.team] ? TEAMS[t.team].elo : 1500)}</td><td class="l">${dots(t.team)}</td></tr>`).join("");
    $("table-note").textContent = `${L.season} after ${L.played} matches. Tap a club to put it on the scoreboard.`;
  }

  function teamsUI() {
    const names = Object.keys(TEAMS).sort();
    const opts = names.map((t) => `<option value="${esc(t)}">${esc(t)}</option>`).join("");
    $("home").innerHTML = opts; $("away").innerHTML = opts;
    const [h, a] = decodeURIComponent(location.hash.slice(1)).split("-v-");
    const first = L && L.fixtures && L.fixtures[0];
    $("home").value = names.includes(h) ? h : first && names.includes(first.home) ? first.home : names[0];
    $("away").value = names.includes(a) && a !== $("home").value ? a : first && names.includes(first.away) ? first.away : names.find((t) => t !== $("home").value);
  }

  function setMatch(h, a) {
    if (!TEAMS[h] || !TEAMS[a]) return;
    $("home").value = h; $("away").value = a; scoreboard();
    $("matchup").scrollIntoView({ behavior: "smooth" });
  }

  document.addEventListener("click", (e) => {
    const f = e.target.closest(".fx[data-h]");
    if (f) return setMatch(f.dataset.h, f.dataset.a);
    const r = e.target.closest("tr[data-team]");
    if (r) {
      const t = r.dataset.team;
      if ($("home").value === t) return;
      $("away").value === t ? setMatch(t, $("home").value) : setMatch(t, $("away").value);
    }
  });
  const fix = (changed) => {
    if ($("home").value === $("away").value) $(changed === "home" ? "away" : "home").value = Object.keys(TEAMS).sort().find((t) => t !== $(changed).value);
    scoreboard();
  };
  $("home").addEventListener("change", () => fix("home"));
  $("away").addEventListener("change", () => fix("away"));
  $("swap").addEventListener("click", () => { [$("home").value, $("away").value] = [$("away").value, $("home").value]; scoreboard(); });

  Promise.all([fetch("data.json").then((r) => r.json()), fetch("/api/live").then((r) => (r.ok ? r.json() : null)).catch(() => null)]).then(([m, live]) => {
    M = m; L = live;
    if (L && L.teams && Object.keys(L.teams).length) {
      TEAMS = L.teams;
      L.table.forEach((t) => (FORM[t.team] = t.last));
      $("season").textContent = `Premier League ${L.season}`;
      const last = L.last_result ? new Date(L.last_result + "T12:00:00Z").toLocaleDateString(undefined, { day: "numeric", month: "long", timeZone: "UTC" }) : "";
      $("status").textContent = `Ratings include every result up to ${last}. ${L.fixtures.length ? `${L.fixtures.length} upcoming fixtures priced.` : ""}`;
    } else {
      TEAMS = m.teams;
      $("season").textContent = "Premier League";
      $("status").innerHTML = '<span class="notice">Live results are unavailable right now, so ratings are as of the end of 2024/25.</span>';
    }
    teamsUI(); fixtures(); scoreboard(); table();
  });
})();
