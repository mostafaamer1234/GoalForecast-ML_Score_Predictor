// Recomputes the exported logistic-regression model (backtest/export_site_data.py) in the browser.
const $ = (id) => document.getElementById(id);
const pct = (p) => `${Math.round(p * 100)}%`;

function features(m, home, away) {
  const h = m.teams[home], a = m.teams[away];
  const v = {
    elo_diff: h.elo + m.home_advantage - a.elo,
    home_elo: h.elo, away_elo: a.elo,
    home_gf5: h.gf5, home_ga5: h.ga5, away_gf5: a.gf5, away_ga5: a.ga5,
    home_pts5: h.pts5, away_pts5: a.pts5,
  };
  return m.features.map((f, i) => (v[f] - m.scaler.mean[i]) / m.scaler.scale[i]);
}

function predict(m, home, away) {
  const x = features(m, home, away);
  const z = m.coef.map((w, k) => w.reduce((s, wi, i) => s + wi * x[i], m.intercept[k]));
  const top = Math.max(...z);
  const e = z.map((v) => Math.exp(v - top));
  const sum = e.reduce((s, v) => s + v, 0);
  const out = {};
  m.class_order.forEach((c, k) => (out[c] = e[k] / sum));
  return out;
}

function render(m) {
  const home = $("home").value, away = $("away").value;
  if (home === away) return;
  const p = predict(m, home, away);
  $("p-h").textContent = pct(p.H); $("p-d").textContent = pct(p.D); $("p-a").textContent = pct(p.A);
  $("l-h").textContent = `${home} win`; $("l-a").textContent = `${away} win`;
  $("b-h").style.flexBasis = `${p.H * 100}%`; $("b-d").style.flexBasis = `${p.D * 100}%`; $("b-a").style.flexBasis = `${p.A * 100}%`;
  const best = ["H", "D", "A"].reduce((a, b) => (p[a] >= p[b] ? a : b));
  document.querySelectorAll(".outcome .q-figure").forEach((el, i) => el.classList.toggle("lead", ["H", "D", "A"][i] === best));
  $("fair").textContent = `Fair decimal odds · ${(1 / p.H).toFixed(2)} · ${(1 / p.D).toFixed(2)} · ${(1 / p.A).toFixed(2)}`;
  const h = m.teams[home], a = m.teams[away];
  $("n-h").textContent = home; $("n-a").textContent = away;
  $("elo-h").textContent = Math.round(h.elo); $("elo-a").textContent = Math.round(a.elo);
  $("pts-h").textContent = h.pts5.toFixed(1); $("pts-a").textContent = a.pts5.toFixed(1);
  $("gf-h").textContent = h.gf5.toFixed(1); $("gf-a").textContent = a.gf5.toFixed(1);
  $("ga-h").textContent = h.ga5.toFixed(1); $("ga-a").textContent = a.ga5.toFixed(1);
  history.replaceState(null, "", `#${encodeURIComponent(home)}-v-${encodeURIComponent(away)}`);
}

function board(m) {
  const rows = Object.entries(m.metrics);
  const base = rows.find(([n]) => n.startsWith("Base"))[1].log_loss;
  const book = rows.find(([n]) => n.startsWith("Bookmaker"))[1].log_loss;
  const ours = rows.find(([n]) => n.startsWith("Logistic"))[1].log_loss;
  $("board").innerHTML = rows.map(([name, s]) => {
    const closed = Math.max(0, Math.min(1, (base - s.log_loss) / (base - book)));
    const color = name.startsWith("Bookmaker") ? "var(--ink)" : "var(--accent)";
    return `<tr class="${name.startsWith("Logistic") ? "best" : ""}">
      <td><div class="q-bar"><i style="width:${Math.max(3, closed * 100)}%;background:${color}"></i><b>${name}</b></div></td>
      <td>${s.log_loss.toFixed(3)}</td><td>${s.brier.toFixed(3)}</td><td>${(s.accuracy * 100).toFixed(1)}%</td></tr>`;
  }).join("");
  $("board-note").textContent = `${m.test_matches.toLocaleString()} matches across 2022/23, 2023/24 and 2024/25. Lower log loss and Brier are better. The logistic model closes ${Math.round(((base - ours) / (base - book)) * 100)}% of the gap to the bookmaker without seeing any odds.`;
}

function calibration(m) {
  const S = 320, pad = 34, w = S - pad - 8, y0 = S - pad;
  const X = (v) => pad + v * w, Y = (v) => y0 - v * w;
  let s = "";
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    s += `<line class="grid" x1="${X(0)}" x2="${X(1)}" y1="${Y(t)}" y2="${Y(t)}"/>`;
    s += `<text x="${X(0) - 8}" y="${Y(t) + 4}" text-anchor="end">${t * 100}%</text>`;
    s += `<text x="${X(t)}" y="${y0 + 18}" text-anchor="middle">${t * 100}%</text>`;
  });
  s += `<line class="ideal" x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}"/>`;
  s += `<polyline class="line" points="${m.calibration.map((b) => `${X(b.predicted)},${Y(b.actual)}`).join(" ")}"/>`;
  s += m.calibration.map((b) => `<circle class="pt" cx="${X(b.predicted)}" cy="${Y(b.actual)}" r="4"><title>Predicted ${pct(b.predicted)}, actual ${pct(b.actual)} (${b.n} matches)</title></circle>`).join("");
  s += `<text x="${X(0.5)}" y="${S + 4}" text-anchor="middle">Predicted home-win chance</text>`;
  $("calib-chart").innerHTML = s;
  const top = m.calibration[m.calibration.length - 1];
  $("calib-note").textContent = `Strongest favourites: predicted ${pct(top.predicted)}, won ${pct(top.actual)} of ${top.n} matches. The model is slightly overconfident at the top end.`;
}

fetch("data.json").then((r) => r.json()).then((m) => {
  const teams = Object.keys(m.teams).sort();
  const opts = teams.map((t) => `<option value="${t}">${t}</option>`).join("");
  $("home").innerHTML = opts; $("away").innerHTML = opts;
  const [h, a] = decodeURIComponent(location.hash.slice(1)).split("-v-");
  $("home").value = teams.includes(h) ? h : "Arsenal";
  $("away").value = teams.includes(a) && a !== $("home").value ? a : "Liverpool";
  const fix = (changed) => {
    if ($("home").value === $("away").value) {
      const other = changed === "home" ? "away" : "home";
      $(other).value = teams.find((t) => t !== $(changed).value);
    }
    render(m);
  };
  $("home").addEventListener("change", () => fix("home"));
  $("away").addEventListener("change", () => fix("away"));
  $("swap").addEventListener("click", () => { [$("home").value, $("away").value] = [$("away").value, $("home").value]; render(m); });
  render(m); board(m); calibration(m);
});
