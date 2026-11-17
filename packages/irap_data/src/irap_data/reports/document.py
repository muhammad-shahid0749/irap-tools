"""A report as a sequence of blocks, rendered as an HTML page.

The text of the blocks may have the inline Markdown subset `code`, **strong** and *emphasis*,
which `to_html` converts to HTML elements.
"""

import dataclasses as dc
import html
import re
import typing as T


@dc.dataclass(frozen=True)
class Heading:
    text: str
    level: int  # 1 for the title


@dc.dataclass(frozen=True)
class Paragraph:
    text: str


@dc.dataclass(frozen=True)
class BulletList:
    items: tuple[str, ...]


@dc.dataclass(frozen=True)
class Table:
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    alignment: str  # 'l' or 'r' for each column
    footer: tuple[tuple[str, ...], ...] = ()  # Rows after `rows` that are not sorted, e.g. totals


@dc.dataclass(frozen=True)
class Figure:
    """A figure file, shown by its format: an HTML page in a frame, a PDF as a link, as the HTML
    `img` element does not display it, and other formats as an image."""
    path: str  # Relative to the report
    caption: str

    @property
    def format(self) -> str:
        return self.path.rsplit(".", 1)[-1].lower()

    @property
    def link_text(self) -> str:
        """The caption with the format, e.g. 'Plot (PDF)', for a link to the figure."""
        return f"{self.caption} ({self.format.upper()})"


Block = Heading | Paragraph | BulletList | Table | Figure


def make_table(header: T.Sequence[str], rows: T.Iterable[T.Sequence[str]],
               alignment: str, footer: T.Iterable[T.Sequence[str]] = ()) -> Table:
    return Table(tuple(header), tuple(map(tuple, rows)), alignment, tuple(map(tuple, footer)))


def get_title(blocks: T.Sequence[Block]) -> str:
    """The plain text of the first level-1 heading."""
    heading = next(b for b in blocks if isinstance(b, Heading) and b.level == 1)
    return re.sub(r"[`*]", "", heading.text)


# HTML #############################################################################################

_INLINE_PATTERNS = ((re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
                    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
                    (re.compile(r"\*(.+?)\*"), r"<em>\1</em>"))

#: The colors of the light and the dark theme, as CSS variables. The segment map uses them too.
THEME_CSS = """\
:root { color-scheme: light dark; --fg: #1f2328; --bg: #ffffff; --muted: #59636e;
        --border: #d1d9e0; --stripe: #f6f8fa; --code: #eff1f3; --link: #0969da; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e6edf3; --bg: #0d1117; --muted: #9198a1; --border: #3d444d;
          --stripe: #151b23; --code: #262c36; --link: #4493f8; }
}
"""

_HTML_CSS = THEME_CSS + """\
body { margin: 0 auto; max-width: 1200px; padding: 16px 24px 48px; color: var(--fg);
       background: var(--bg); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 1.6em; } h2 { margin-top: 2em; border-bottom: 1px solid var(--border); }
h3 { margin-top: 1.5em; }
a { color: var(--link); }
code { padding: 0.1em 0.3em; border-radius: 4px; background: var(--code); font-size: 0.9em; }
.table-wrapper { overflow-x: auto; margin: 1em 0; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 10px; border: 1px solid var(--border); white-space: nowrap; }
th { background: var(--stripe); cursor: pointer; user-select: none; }
th::after { content: " \\2195"; opacity: 0.35; }
th[aria-sort="ascending"]::after { content: " \\25B2"; opacity: 1; }
th[aria-sort="descending"]::after { content: " \\25BC"; opacity: 1; }
tbody tr:nth-child(even) { background: var(--stripe); }
tfoot td { font-weight: 600; }
tfoot tr:first-child td { border-top: 2px solid var(--muted); }
.l { text-align: left; } .r { text-align: right; }
figure { margin: 1em 0; } figure img { max-width: 100%; background: #ffffff; }
figure iframe { width: 100%; height: 70vh; border: 1px solid var(--border); }
figcaption { color: var(--muted); font-size: 0.9em; }
"""

# Sorts a table by a column on a click on its header, which cycles through the first direction,
# the other direction and the original order. A right-aligned column is numeric: it is sorted by
# the leading number of its cells, e.g. 1234 in '1,234 (5.6%)' and 0.1 in '<0.1%', descending
# first, and the cells without a number are last in both directions.
_HTML_SCRIPT = r"""
for (const table of document.querySelectorAll("table")) {
  const body = table.tBodies[0];
  const originalRows = Array.from(body.rows);
  const headers = Array.from(table.tHead.rows[0].cells);
  headers.forEach((th, column) => {
    const isNumeric = th.classList.contains("r");
    const [firstDirection, secondDirection] = isNumeric ? ["descending", "ascending"]
                                                        : ["ascending", "descending"];
    const getKey = row => {
      const text = row.cells[column].textContent.trim();
      if (!isNumeric) return text;
      // A bound such as '<0.1%' is sorted by its number.
      const match = text.replace(/,/g, "").replace(/^[<>]/, "").match(/^-?\d*\.?\d+/);
      return match ? parseFloat(match[0]) : null;
    };
    const compare = (a, b) => isNumeric ? a - b : a.localeCompare(b, undefined, {numeric: true});
    const sort = () => {
      const current = th.getAttribute("aria-sort");
      const direction = current === null ? firstDirection
                        : current === firstDirection ? secondDirection : null;
      headers.forEach(other => other.removeAttribute("aria-sort"));
      let rows = originalRows;
      if (direction) {
        th.setAttribute("aria-sort", direction);
        const sign = direction === "ascending" ? 1 : -1;
        rows = originalRows.map(row => [getKey(row), row]).sort(([a], [b]) =>
          a === null ? (b === null ? 0 : 1) : b === null ? -1 : sign * compare(a, b)
        ).map(([, row]) => row);
      }
      body.append(...rows);
    };
    th.tabIndex = 0;
    th.addEventListener("click", sort);
    th.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); sort(); }
    });
  });
}
"""


def to_html_inline(text: str) -> str:
    """Escapes `text` for HTML and converts its inline Markdown (see the module docstring)."""
    text = html.escape(text, quote=False)
    for pattern, replacement in _INLINE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _to_html_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"[`*]", "", text).lower()).strip("-")


def _to_html_block(block: Block) -> str:
    if isinstance(block, Heading):
        return (f'<h{block.level} id="{_to_html_id(block.text)}">{to_html_inline(block.text)}'
                f"</h{block.level}>")
    if isinstance(block, Paragraph):
        return f"<p>{to_html_inline(block.text)}</p>"
    if isinstance(block, BulletList):
        items = "".join(f"\n  <li>{to_html_inline(item)}</li>" for item in block.items)
        return f"<ul>{items}\n</ul>"
    if isinstance(block, Table):
        def to_row(cells: T.Sequence[str], tag: str) -> str:
            return "<tr>" + "".join(f'<{tag} class="{a}">{to_html_inline(cell)}</{tag}>'
                                    for cell, a in zip(cells, block.alignment)) + "</tr>"

        def to_rows(rows: T.Sequence[T.Sequence[str]]) -> str:
            return "".join(f"\n{to_row(row, 'td')}" for row in rows)

        # The sorting script sorts only the body, so the footer stays last.
        footer = f"<tfoot>{to_rows(block.footer)}\n</tfoot>\n" if block.footer else ""
        return (f'<div class="table-wrapper"><table>\n<thead>{to_row(block.header, "th")}'
                f"</thead>\n<tbody>{to_rows(block.rows)}\n</tbody>\n{footer}</table></div>")
    if isinstance(block, Figure):
        path = html.escape(block.path)
        link = f'<a href="{path}">{to_html_inline(block.link_text)}</a>'
        if block.format == "html":
            return (f'<figure><iframe src="{path}" title="{html.escape(block.caption)}"'
                    f' loading="lazy"></iframe><figcaption>{link}</figcaption></figure>')
        if block.format == "pdf":
            return f"<p>{link}</p>"
        return (f'<figure><a href="{path}"><img src="{path}" alt="{html.escape(block.caption)}">'
                f"</a><figcaption>{to_html_inline(block.caption)}</figcaption></figure>")
    raise TypeError(f"Unknown block type {type(block).__name__}.")


def to_html(blocks: T.Sequence[Block]) -> str:
    """An HTML page with inline styles, titled with `get_title`, with a light and a dark theme and
    tables that can be sorted by a column (see `_HTML_SCRIPT`)."""
    body = "\n\n".join(map(_to_html_block, blocks))
    return (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{html.escape(get_title(blocks))}</title>\n<style>\n{_HTML_CSS}</style>\n"
            f"</head>\n<body>\n{body}\n<script>{_HTML_SCRIPT}</script>\n</body>\n</html>\n")
