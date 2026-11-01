// Charts are rendered to HTML and SVG at build time, so they need no chart library and show their
// numbers without JavaScript. Colors come from CSS variables, so both themes pick their own steps.

const esc = (text) =>
  String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

export const fmt = (value) => {
  if (value >= 1000) return Math.round(value).toLocaleString("en-US");
  if (value >= 100) return String(Math.round(value));
  return String(Number(value.toPrecision(3)));
};

const niceMax = (max) => {
  const step = 10 ** Math.floor(Math.log10(max));
  for (const factor of [1, 2, 3, 4, 6, 8, 10]) {
    if (factor * step >= max) return factor * step;
  }
  return 10 * step;
};

const legend = (series) =>
  series.length < 2
    ? ""
    : `<ul class="legend">${series
        .map((s) => `<li><button type="button" data-legend="${esc(s.name)}" aria-pressed="true"><i class="swatch" style="--c:var(--${s.color})"></i>${esc(s.name)}</button></li>`)
        .join("")}</ul>`;

const dataTable = (headers, rows) => `<details class="chart__data"><summary>Show the numbers</summary>
<div class="table-wrap"><table><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead>
<tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${esc(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div></details>`;

const frame = ({ title, note, source, id }, body, extras = "") => `<figure class="chart"${id ? ` id="${id}"` : ""}>
<figcaption><h3 class="chart__title">${esc(title)}</h3>${note ? `<p class="chart__note">${esc(note)}</p>` : ""}</figcaption>
${body}${extras}
${source ? `<p class="chart__source">Source: <a href="${esc(source.url)}">${esc(source.label)}</a></p>` : ""}
</figure>`;

// Sequence counts sit on a log scale, so a doubling is the same distance wherever it starts.
export function lineChart(spec) {
  const { x, series, unit, xLabel } = spec;
  const [width, height] = [560, 330];
  const margin = { top: 16, right: 62, bottom: 50, left: 54 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const top = niceMax(Math.max(...series.flatMap((s) => s.values)));
  const px = (i) => margin.left + (plotW * Math.log2(x[i] / x[0])) / Math.log2(x.at(-1) / x[0]);
  const py = (v) => margin.top + plotH * (1 - v / top);

  const grid = [0, 0.25, 0.5, 0.75, 1]
    .map((f) => {
      const y = py(top * f);
      return `<line class="grid" x1="${margin.left}" x2="${width - margin.right}" y1="${y}" y2="${y}"/><text class="tick" x="${margin.left - 8}" y="${y + 5}" text-anchor="end">${fmt(top * f)}</text>`;
    })
    .join("");
  const ticks = x
    .map((label, i) => `<text class="tick" x="${px(i)}" y="${height - margin.bottom + 22}" text-anchor="middle">${esc(label)}</text>`)
    .join("");

  // End labels move apart when two series finish close together.
  const ends = series
    .map((s) => ({ s, y: py(s.values.at(-1)) + 5 }))
    .sort((a, b) => a.y - b.y);
  ends.forEach((end, i) => {
    if (i > 0 && end.y - ends[i - 1].y < 18) end.y = ends[i - 1].y + 18;
  });

  const lines = series
    .map((s) => {
      const points = s.values.map((v, i) => `${px(i)},${py(v)}`).join(" ");
      const dots = s.values
        .map(
          (v, i) =>
            `<circle class="dot" cx="${px(i)}" cy="${py(v)}" r="5" style="--c:var(--${s.color})"/><circle class="hit" cx="${px(i)}" cy="${py(v)}" r="16" tabindex="0" data-tip="${esc(`${s.name}, ${x[i]} ${xLabel}: ${fmt(v)} ${unit}`)}"/>`,
        )
        .join("");
      const label = ends.find((end) => end.s === s);
      return `<g data-series="${esc(s.name)}"><polyline class="line" pathLength="1" points="${points}" style="--c:var(--${s.color})"/>${dots}<text class="end" x="${width - margin.right + 10}" y="${label.y}">${fmt(s.values.at(-1))}</text></g>`;
    })
    .join("");

  const summary = series.map((s) => `${s.name} reaches ${fmt(s.values.at(-1))} ${unit}`).join(", ");
  const svg = `<svg class="chart__plot" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(`${spec.title}. ${summary}.`)}">
${grid}${ticks}<text class="axis" x="${margin.left + plotW / 2}" y="${height - 8}" text-anchor="middle">${esc(xLabel)}</text>${lines}</svg>`;
  const table = dataTable(
    [xLabel, ...series.map((s) => `${s.name} (${unit})`)],
    x.map((label, i) => [label, ...series.map((s) => s.values[i])]),
  );
  return frame(spec, `${legend(series)}<p class="chart__unit">${esc(unit)}</p>${svg}`, table);
}

// Horizontal bars carry long labels at phone width, where SVG text would shrink out of reach.
export function barChart(spec) {
  const { groups, series, unit } = spec;
  const max = Math.max(...groups.flatMap((g) => g.values));
  const rows = groups
    .map((group) => {
      const bars = group.values
        .map((value, i) => {
          const s = series[i];
          const display = group.display?.[i] ?? `${fmt(value)} ${unit}`;
          const tip = `${group.label}${s.name ? `, ${s.name}` : ""}: ${display}`;
          return `<div class="bar" data-series="${esc(s.name)}" tabindex="0" data-tip="${esc(tip)}"><span class="bar__fill" style="--w:${((value / max) * 100).toFixed(2)}%;--c:var(--${s.color})"></span><span class="bar__value">${esc(display)}</span></div>`;
        })
        .join("");
      return `<div class="bars__row"><div class="bars__label">${esc(group.label)}</div><div class="bars__set">${bars}</div></div>`;
    })
    .join("");
  const table = dataTable(
    ["", ...series.map((s) => (s.name ? `${s.name} (${unit})` : unit))],
    groups.map((g) => [g.label, ...g.values]),
  );
  return frame(spec, `${legend(series)}<div class="bars">${rows}</div>`, table);
}

// Percent difference around zero: wins and losses point opposite ways and differ in color, so the
// losses stay as visible as the wins.
export function divergingChart(spec) {
  const { rows } = spec;
  const reach = Math.max(...rows.map((row) => Math.abs(row.percent)));
  const body = rows
    .map((row) => {
      const width = ((Math.abs(row.percent) / reach) * 100).toFixed(2);
      const sign = row.percent >= 0 ? "+" : "−";
      const text = `${sign}${Math.abs(Math.round(row.percent))}%`;
      const tip = `${row.label}: Tur ${fmt(row.tur)}, ${spec.other} ${fmt(row.base)} ${spec.unit}`;
      return `<div class="dv__row" tabindex="0" data-tip="${esc(tip)}"><div class="dv__label">${esc(row.label)}</div>
<div class="dv__plot ${row.percent >= 0 ? "is-win" : "is-loss"}" style="--w:${width}%"><span class="dv__fill"></span><span class="dv__value">${text}</span></div></div>`;
    })
    .join("");
  const key = `<ul class="legend"><li><i class="swatch" style="--c:var(--s1)"></i>Tur faster</li><li><i class="swatch" style="--c:var(--s3)"></i>${esc(spec.other)} faster</li></ul>`;
  const table = dataTable(
    ["Test", `Tur (${spec.unit})`, `${spec.other} (${spec.unit})`, "Difference"],
    rows.map((row) => [row.label, row.tur, row.base, `${Math.round(row.percent)}%`]),
  );
  return frame(spec, `${key}<div class="dv">${body}</div>`, table);
}
