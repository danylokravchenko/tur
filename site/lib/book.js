// Turns docs/src into website chapters. SUMMARY.md decides the navigation, so a chapter added to
// the book appears on the site with no change here. A broken internal link fails the build.
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import hljs from "highlight.js";
import MarkdownIt from "markdown-it";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const bookDir = path.join(root, "docs/src");
const repoUrl = "https://github.com/danylokravchenko/tur";

export const readRepoFile = (relative) => readFileSync(path.join(root, relative), "utf8");
export const readBookFile = (relative) => readFileSync(path.join(bookDir, relative), "utf8");

export const chapterUrl = (file) => {
  if (file === "introduction.md") return "/docs/";
  return `/docs/${file.replace(/\.md$/, "")}/`;
};

export function parseSummary(text) {
  const parts = [];
  let part = { title: "", chapters: [] };
  parts.push(part);
  for (const line of text.split("\n")) {
    const heading = line.match(/^# (.+)$/);
    if (heading && heading[1] !== "Summary") {
      part = { title: heading[1], chapters: [] };
      parts.push(part);
      continue;
    }
    const link = line.match(/^(\s*)(?:- )?\[(.+?)\]\((.+?\.md)\)/);
    if (link) {
      part.chapters.push({
        title: link[2],
        file: link[3],
        nested: link[1].length > 0,
        url: chapterUrl(link[3]),
      });
    }
  }
  return parts;
}

// Same ids as mdBook, so an anchor means one thing in both renderings.
function slugger() {
  const seen = new Map();
  return (text) => {
    const base = text
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s_-]/gu, "")
      .trim()
      .replace(/\s/g, "-");
    const count = seen.get(base) ?? 0;
    seen.set(base, count + 1);
    return count === 0 ? base : `${base}-${count}`;
  };
}

const inlineText = (token) =>
  (token.children ?? [])
    .filter((child) => child.type === "text" || child.type === "code_inline")
    .map((child) => child.content)
    .join("");

const escapeHtml = (text) =>
  text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function createRenderer() {
  const md = new MarkdownIt({
    html: true,
    highlight(code, lang) {
      const body = hljs.getLanguage(lang)
        ? hljs.highlight(code, { language: lang, ignoreIllegals: true }).value
        : escapeHtml(code);
      return `<pre data-lang="${escapeHtml(lang)}"><code>${body}</code></pre>`;
    },
  });

  md.renderer.rules.heading_open = (tokens, idx, _options, env) => {
    const level = Number(tokens[idx].tag.slice(1));
    const text = inlineText(tokens[idx + 1]);
    const id = env.slug(text);
    env.headings.push({ level, text, id });
    env.openId = id;
    return `<h${level} id="${id}">`;
  };
  md.renderer.rules.heading_close = (tokens, idx, _options, env) => {
    const level = tokens[idx].tag.slice(1);
    if (level === "1") return "</h1>";
    return `<a class="anchor" href="#${env.openId}" aria-hidden="true" tabindex="-1">#</a></h${level}>`;
  };

  const defaultLink = md.renderer.renderToken.bind(md.renderer);
  md.renderer.rules.link_open = (tokens, idx, options, env) => {
    const token = tokens[idx];
    token.attrSet("href", env.resolve(token.attrGet("href")));
    return defaultLink(tokens, idx, options);
  };

  md.renderer.rules.table_open = () => '<div class="table-wrap"><table>';
  md.renderer.rules.table_close = () => "</table></div>";
  return md;
}

function renderChapter(md, chapter, known) {
  const links = [];
  const env = {
    headings: [],
    slug: slugger(),
    resolve(href) {
      if (/^[a-z]+:/i.test(href)) return href;
      if (href.startsWith("#")) {
        links.push({ file: chapter.file, anchor: href.slice(1) });
        return href;
      }
      const [target, anchor] = href.split("#");
      const file = path.posix.normalize(path.posix.join(path.posix.dirname(chapter.file), target));
      if (known.has(file)) {
        if (anchor) links.push({ file, anchor });
        return chapterUrl(file) + (anchor ? `#${anchor}` : "");
      }
      // A book chapter missing from SUMMARY.md would have no page to land on.
      if (file.endsWith(".md") && !file.startsWith("..")) {
        throw new Error(`${chapter.file}: link to ${href} is not in SUMMARY.md`);
      }
      // Anything outside the book, such as a source file, is read on GitHub.
      const repoPath = path.posix.normalize(path.posix.join("docs/src", file));
      return `${repoUrl}/blob/main/${repoPath}`;
    },
  };
  const source = readBookFile(chapter.file);
  const html = md.render(source, env);
  return { html, headings: env.headings, links, source };
}

// One search entry per heading, so a hit lands on the section instead of a long chapter's top.
function searchSections(md, chapter, source) {
  const slug = slugger();
  const sections = [];
  let current = { title: chapter.title, url: chapter.url, text: [] };
  const tokens = md.parse(source, {});
  tokens.forEach((token, index) => {
    if (token.type !== "inline") return;
    const previous = tokens[index - 1];
    if (previous?.type === "heading_open") {
      sections.push(current);
      const title = inlineText(token);
      const id = slug(title);
      current = {
        title: previous.tag === "h1" ? chapter.title : title,
        url: previous.tag === "h1" ? chapter.url : `${chapter.url}#${id}`,
        text: [],
      };
    } else {
      current.text.push(inlineText(token));
    }
  });
  sections.push(current);
  return sections
    .filter((section) => section.text.length > 0)
    .map((section) => ({
      chapter: chapter.title,
      title: section.title,
      url: section.url,
      text: section.text.join(" "),
    }));
}

const firstParagraph = (md, source) => {
  const tokens = md.parse(source, {});
  const index = tokens.findIndex((token) => token.type === "paragraph_open");
  return index < 0 ? "" : inlineText(tokens[index + 1]);
};

export function loadBook() {
  const parts = parseSummary(readBookFile("SUMMARY.md"));
  const listed = parts.flatMap((part) => part.chapters.map((chapter) => ({ ...chapter, part: part.title })));
  const known = new Set(listed.map((chapter) => chapter.file));
  const md = createRenderer();

  const links = [];
  const search = [];
  const anchors = new Map();
  const chapters = listed.map((chapter, index) => {
    const rendered = renderChapter(md, chapter, known);
    links.push(...rendered.links.map((link) => ({ ...link, from: chapter.file })));
    search.push(...searchSections(md, chapter, rendered.source));
    anchors.set(chapter.file, new Set(rendered.headings.map((heading) => heading.id)));
    return {
      ...chapter,
      html: rendered.html,
      toc: rendered.headings.filter((heading) => heading.level === 2 || heading.level === 3),
      description: firstParagraph(md, rendered.source),
      editUrl: `${repoUrl}/edit/main/docs/src/${chapter.file}`,
      prev: index > 0 ? { title: listed[index - 1].title, url: listed[index - 1].url } : null,
      next: index + 1 < listed.length ? { title: listed[index + 1].title, url: listed[index + 1].url } : null,
    };
  });

  const broken = links.filter((link) => !anchors.get(link.file)?.has(link.anchor));
  if (broken.length > 0) {
    const list = broken.map((link) => `${link.from} -> ${link.file}#${link.anchor}`).join("\n");
    throw new Error(`Broken book anchors:\n${list}`);
  }

  return { parts, chapters, search };
}
