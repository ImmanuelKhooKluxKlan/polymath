#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function inline(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

function isTableDivider(line) {
  return /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/.test(line);
}

function cells(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim());
}

function render(markdown) {
  const lines = markdown.replaceAll('\r\n', '\n').split('\n');
  const output = [];
  let paragraph = [];
  let list = null;
  let inCode = false;
  let code = [];

  const flushParagraph = () => {
    if (paragraph.length) output.push(`<p>${inline(paragraph.join(' '))}</p>`);
    paragraph = [];
  };
  const flushList = () => {
    if (list) output.push(`</${list}>`);
    list = null;
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (line.trim() === '[[PAGEBREAK]]') {
      flushParagraph();
      flushList();
      output.push('<div class="page-break" aria-hidden="true"></div>');
      continue;
    }
    if (line.startsWith('```')) {
      flushParagraph();
      flushList();
      if (inCode) {
        output.push(`<pre><code>${escapeHtml(code.join('\n'))}</code></pre>`);
        code = [];
      }
      inCode = !inCode;
      continue;
    }
    if (inCode) {
      code.push(line);
      continue;
    }
    if (line.includes('|') && index + 1 < lines.length && isTableDivider(lines[index + 1])) {
      flushParagraph();
      flushList();
      const headings = cells(line);
      output.push(`<table><thead><tr>${headings.map((cell) => `<th>${inline(cell)}</th>`).join('')}</tr></thead><tbody>`);
      index += 2;
      while (index < lines.length && lines[index].includes('|') && lines[index].trim()) {
        output.push(`<tr>${cells(lines[index]).map((cell) => `<td>${inline(cell)}</td>`).join('')}</tr>`);
        index += 1;
      }
      output.push('</tbody></table>');
      index -= 1;
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length;
      output.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }
    const quote = line.match(/^>\s*(.*)$/);
    if (quote) {
      flushParagraph();
      flushList();
      output.push(`<aside>${inline(quote[1])}</aside>`);
      continue;
    }
    const unordered = line.match(/^\s*-\s+(.+)$/);
    const ordered = line.match(/^\s*\d+\.\s+(.+)$/);
    if (unordered || ordered) {
      flushParagraph();
      const wanted = ordered ? 'ol' : 'ul';
      if (list !== wanted) {
        flushList();
        list = wanted;
        output.push(`<${list}>`);
      }
      output.push(`<li>${inline((unordered || ordered)[1])}</li>`);
      continue;
    }
    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }
    paragraph.push(line.trim().replace(/\s{2}$/, ''));
  }
  flushParagraph();
  flushList();
  return output.join('\n');
}

async function main() {
  const [input, output, title = 'Polymath Training Report'] = process.argv.slice(2);
  if (!input || !output) throw new Error('Usage: renderMarkdownReport.mjs input.md output.html [title]');
  const markdown = await fs.readFile(path.resolve(input), 'utf8');
  const body = render(markdown);
  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>${escapeHtml(title)}</title>
<style>
@page { size: A4; margin: 17mm 15mm 18mm; }
:root { color: #17213b; font-family: Arial, Helvetica, sans-serif; font-size: 16px; }
* { box-sizing: border-box; }
body { max-width: 900px; margin: 0 auto; line-height: 1.65; letter-spacing: .012em; background: #fff; }
h1 { color: #4b3dbb; font-size: 2.25rem; line-height: 1.15; margin: 0 0 1.2rem; padding: 1rem 0 1.2rem; border-bottom: 5px solid #d8d4ff; }
h2 { color: #5142c3; font-size: 1.55rem; line-height: 1.25; margin: 2.2rem 0 .75rem; break-after: avoid; }
h3 { color: #674bc7; font-size: 1.18rem; margin: 1.5rem 0 .55rem; break-after: avoid; }
h4 { color: #7a356b; font-size: 1.02rem; margin: 1.2rem 0 .45rem; break-after: avoid; }
p, li { max-width: 75ch; }
p { margin: .6rem 0 1rem; }
li { margin: .35rem 0; padding-left: .25rem; }
ul, ol { padding-left: 1.5rem; margin: .5rem 0 1.15rem; }
strong { color: #352990; }
code { font-family: Consolas, monospace; color: #9b245f; background: #f4f1ff; border-radius: 4px; padding: .1rem .3rem; overflow-wrap: anywhere; }
pre { background: #171a31; color: #f3f1ff; border-left: 6px solid #7b68ee; padding: 1rem; border-radius: 8px; overflow-wrap: anywhere; white-space: pre-wrap; break-inside: avoid; }
pre code { color: inherit; background: transparent; padding: 0; }
aside { max-width: 75ch; margin: .8rem 0 1.1rem; padding: .75rem 1rem; border-left: 6px solid #ef6aa8; border-radius: 6px; background: #fff0f7; color: #422745; font-weight: 600; }
.page-break { break-before: page; page-break-before: always; height: 0; }
table { width: 100%; border-collapse: collapse; margin: .8rem 0 1.4rem; font-size: .9rem; break-inside: avoid; }
th { color: #fff; background: #5545be; text-align: left; }
th, td { border: 1px solid #d8d4ed; padding: .55rem .6rem; vertical-align: top; }
tr:nth-child(even) td { background: #f6f4ff; }
h2, h3, table, pre { page-break-inside: avoid; }
@media print { a { color: inherit; text-decoration: none; } body { max-width: none; } }
</style></head><body>${body}</body></html>`;
  await fs.writeFile(path.resolve(output), html, 'utf8');
}

main().catch((error) => {
  process.stderr.write(`${error.message}\n`);
  process.exitCode = 1;
});
