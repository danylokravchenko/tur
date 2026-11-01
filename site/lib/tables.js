// Reads measurement tables out of the book, so a chart can't disagree with the chapter it cites.
// A renamed heading or column fails the build with the lookup that no longer matches.
import { readBookFile } from "./book.js";

const splitRow = (line) =>
  line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());

export function parseTables(markdown) {
  const tables = [];
  let h2 = "";
  let h3 = "";
  const lines = markdown.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith("## ")) [h2, h3] = [line.slice(3), ""];
    if (line.startsWith("### ")) h3 = line.slice(4);
    if (!line.startsWith("|") || !/^\|[\s:|-]+\|$/.test(lines[i + 1] ?? "")) continue;
    const headers = splitRow(line);
    const rows = [];
    let j = i + 2;
    for (; lines[j]?.startsWith("|"); j++) rows.push(splitRow(lines[j]));
    tables.push({ h2, h3, headers, rows });
    i = j - 1;
  }
  return tables;
}

export function openChapter(file) {
  const tables = parseTables(readBookFile(file));
  return {
    // The n-th table in a section whose first column is `firstHeader`.
    table(section, firstHeader, nth = 0) {
      const matches = tables.filter(
        (table) => (table.h3 === section || (table.h2 === section && table.h3 === "")) && table.headers[0] === firstHeader,
      );
      const found = matches[nth];
      if (!found) throw new Error(`${file}: no table "${firstHeader}" #${nth} in section "${section}"`);
      return wrap(found, `${file} > ${section}`);
    },
  };
}

function wrap(table, where) {
  const column = (name) => {
    const index = table.headers.indexOf(name);
    if (index < 0) throw new Error(`${where}: no column "${name}" in ${table.headers.join(", ")}`);
    return index;
  };
  const row = (label) => {
    const found = table.rows.find((cells) => cells[0] === label);
    if (!found) throw new Error(`${where}: no row "${label}"`);
    return found;
  };
  return {
    headers: table.headers,
    rows: table.rows,
    cell: (label, name) => row(label)[column(name)],
    num: (label, name) => toNumber(row(label)[column(name)], `${where}: ${label} / ${name}`),
    rowNums: (label) => row(label).slice(1).map((cell) => toNumber(cell, `${where}: ${label}`)),
    firstColumn: () => table.rows.map((cells) => cells[0]),
  };
}

// Takes the first number in a cell, so "647 tok/s" and "3,799" both read as values.
export function toNumber(cell, where = "cell") {
  const match = cell.replace(/,/g, "").match(/-?\d+(\.\d+)?/);
  if (!match) throw new Error(`${where}: "${cell}" has no number`);
  return Number(match[0]);
}

// Reads the number after a label inside a cell such as "TTFT p50 0.56 s".
export function numberAfter(cell, label) {
  const at = cell.indexOf(label);
  if (at < 0) throw new Error(`"${cell}" has no "${label}"`);
  return toNumber(cell.slice(at + label.length), label);
}
