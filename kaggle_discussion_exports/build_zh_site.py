#!/usr/bin/env python3
import html
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser


ROOT = pathlib.Path(__file__).resolve().parent
CACHE_PATH = ROOT / ".translation_cache_en_zh.json"
OUT_DIR = ROOT / "zh_site"

SKIP_TAGS = {"script", "style", "svg", "math", "code", "pre", "kbd", "samp"}


def load_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Translator:
    def __init__(self):
        self.cache = load_json(CACHE_PATH, {})
        self.dirty = False

    def save(self):
        if self.dirty:
            save_json(CACHE_PATH, self.cache)
            self.dirty = False

    def translate(self, text):
        if not should_translate(text):
            return text
        lead = text[: len(text) - len(text.lstrip())]
        trail = text[len(text.rstrip()) :]
        core = text.strip()
        if core in self.cache:
            return lead + self.cache[core] + trail
        translated = self._translate_core(core)
        self.cache[core] = translated
        self.dirty = True
        if len(self.cache) % 25 == 0:
            self.save()
        return lead + translated + trail

    def _translate_core(self, text):
        chunks = split_text(text, 1800)
        out = []
        for chunk in chunks:
            out.append(self._request(chunk))
            time.sleep(0.08)
        return "".join(out)

    def _request(self, text):
        params = {
            "client": "gtx",
            "sl": "en",
            "tl": "zh-CN",
            "dt": "t",
            "q": text,
        }
        url = "https://translate.googleapis.com/translate_a/single?" + urllib.parse.urlencode(params)
        last_err = None
        for _ in range(4):
            try:
                with urllib.request.urlopen(url, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return "".join(part[0] for part in data[0] if part and part[0])
            except Exception as exc:
                last_err = exc
                time.sleep(1.0)
        raise RuntimeError(f"translation failed: {text[:80]!r}: {last_err}")


def should_translate(text):
    core = text.strip()
    if not core:
        return False
    if not re.search(r"[A-Za-z]", core):
        return False
    if re.fullmatch(r"[\W\d_]+", core):
        return False
    if re.fullmatch(r"https?://\S+", core):
        return False
    return True


def split_text(text, max_len):
    if len(text) <= max_len:
        return [text]
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for part in parts:
        if len(current) + len(part) + 1 <= max_len:
            current = f"{current} {part}".strip()
        else:
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks


class TranslatingHTMLParser(HTMLParser):
    def __init__(self, translator):
        super().__init__(convert_charrefs=False)
        self.translator = translator
        self.parts = []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        self.stack.append(tag.lower())
        self.parts.append("<" + tag + format_attrs(attrs) + ">")

    def handle_startendtag(self, tag, attrs):
        self.parts.append("<" + tag + format_attrs(attrs) + ">")

    def handle_endtag(self, tag):
        if self.stack:
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i] == tag.lower():
                    del self.stack[i:]
                    break
        self.parts.append(f"</{tag}>")

    def handle_data(self, data):
        if any(tag in SKIP_TAGS for tag in self.stack):
            self.parts.append(data)
        else:
            self.parts.append(html.escape(self.translator.translate(data), quote=False))

    def handle_entityref(self, name):
        self.parts.append(f"&{name};")

    def handle_charref(self, name):
        self.parts.append(f"&#{name};")

    def handle_comment(self, data):
        self.parts.append(f"<!--{data}-->")

    def get_html(self):
        return "".join(self.parts)


def format_attrs(attrs):
    if not attrs:
        return ""
    rendered = []
    for key, value in attrs:
        if value is None:
            rendered.append(key)
        else:
            rendered.append(f'{key}="{html.escape(value, quote=True)}"')
    return " " + " ".join(rendered)


def article_dirs():
    return sorted(
        [p for p in ROOT.iterdir() if p.is_dir() and re.match(r"^\d+_", p.name)],
        key=lambda p: p.name,
    )


def rewrite_relative_paths(fragment, article_dir):
    prefix = "../" + article_dir.name + "/"

    def repl_src(match):
        value = match.group(1)
        if re.match(r"^[a-z]+:", value) or value.startswith("#") or value.startswith("../"):
            return match.group(0)
        return f'src="{prefix}{value}"'

    return re.sub(r'src="([^"]+)"', repl_src, fragment)


def build_article_page(article_dir, translator):
    meta = load_json(article_dir / "metadata.json", {})
    source_html = (article_dir / "content.html").read_text(encoding="utf-8")
    parser = TranslatingHTMLParser(translator)
    parser.feed(source_html)
    translated_fragment = rewrite_relative_paths(parser.get_html(), article_dir)
    slug = article_dir.name + ".html"
    page = page_shell(
        title=f"{meta.get('rank', '')} · {meta.get('title', article_dir.name)}",
        body=f"""
        <nav class="topbar"><a href="index.html">← 返回目录</a><a href="{html.escape(meta.get('url', '#'), quote=True)}" target="_blank" rel="noreferrer">Kaggle 原帖</a></nav>
        <main class="article-page">
          <section class="meta-panel">
            <div class="rank">{html.escape(meta.get('rank', ''), quote=False)}</div>
            <h1>{html.escape(meta.get('title', article_dir.name), quote=False)}</h1>
            <p>{html.escape(meta.get('author', ''), quote=False)} · {html.escape(meta.get('type', ''), quote=False)}</p>
          </section>
          <section class="translated-article">{translated_fragment}</section>
        </main>
        """,
    )
    out_path = OUT_DIR / slug
    out_path.write_text(page, encoding="utf-8")
    return slug


def page_shell(title, body):
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title, quote=False)}</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
{body}
</body>
</html>
"""


def build_index(items):
    cards = []
    for item in items:
        meta = item["meta"]
        cards.append(
            f"""<a class="card" href="{html.escape(item['href'], quote=True)}">
  <span class="rank">{html.escape(meta.get('rank', ''), quote=False)}</span>
  <strong>{html.escape(meta.get('title', ''), quote=False)}</strong>
  <span>{html.escape(meta.get('author', ''), quote=False)}</span>
</a>"""
        )
    body = f"""
    <main class="index">
      <header class="hero">
        <p>Orbit Wars Kaggle Discussion</p>
        <h1>获奖方案中文翻译</h1>
        <p>按排名整理，保留原帖图片、公式和表格结构。所有图片均使用相对路径，可随文件夹迁移。</p>
      </header>
      <section class="grid">
        {''.join(cards)}
      </section>
    </main>
    """
    (OUT_DIR / "index.html").write_text(page_shell("Orbit Wars 获奖方案中文翻译", body), encoding="utf-8")


def write_css():
    css = """
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --text: #18212f;
  --muted: #5d6878;
  --line: #dfe4ea;
  --accent: #0b74de;
  --accent-soft: #e6f2ff;
  --code: #f0f3f7;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 16px/1.72 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", sans-serif;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.index, .article-page { width: min(1120px, calc(100% - 32px)); margin: 0 auto; }
.hero { padding: 56px 0 28px; }
.hero p { margin: 0 0 10px; color: var(--muted); }
.hero h1 { margin: 0 0 12px; font-size: clamp(32px, 5vw, 54px); line-height: 1.08; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; padding-bottom: 56px; }
.card {
  display: grid;
  gap: 8px;
  min-height: 150px;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  color: var(--text);
}
.card:hover { text-decoration: none; border-color: #aacbeb; box-shadow: 0 8px 24px rgba(20, 40, 70, .08); }
.rank {
  display: inline-flex;
  width: max-content;
  align-items: center;
  padding: 3px 9px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: #07549f;
  font-size: 13px;
  font-weight: 700;
}
.card strong { font-size: 19px; line-height: 1.28; }
.card span:last-child { color: var(--muted); }
.topbar {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 12px max(16px, calc((100vw - 1120px) / 2));
  border-bottom: 1px solid var(--line);
  background: rgba(255,255,255,.92);
  backdrop-filter: blur(8px);
}
.article-page { padding: 26px 0 64px; }
.meta-panel {
  margin: 18px 0 16px;
  padding: 22px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}
.meta-panel h1 { margin: 12px 0 8px; font-size: clamp(28px, 4vw, 42px); line-height: 1.16; }
.meta-panel p { margin: 0; color: var(--muted); }
.translated-article {
  padding: 28px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}
.translated-article article > header,
.translated-article article > div:first-child { display: none; }
.translated-article h1,
.translated-article h2,
.translated-article h3 { line-height: 1.25; margin: 1.6em 0 .55em; }
.translated-article h1 { font-size: 30px; }
.translated-article h2 { font-size: 24px; border-top: 1px solid var(--line); padding-top: 20px; }
.translated-article h3 { font-size: 19px; }
.translated-article p, .translated-article ul, .translated-article ol { margin: 0 0 1em; }
.translated-article li { margin: .25em 0; }
.translated-article img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 18px auto;
  border: 1px solid var(--line);
  border-radius: 6px;
}
.translated-article table {
  width: 100%;
  border-collapse: collapse;
  margin: 18px 0;
  overflow: auto;
  display: block;
}
.translated-article th, .translated-article td {
  border: 1px solid var(--line);
  padding: 8px 10px;
  vertical-align: top;
}
.translated-article code {
  padding: 2px 5px;
  border-radius: 4px;
  background: var(--code);
}
.translated-article pre {
  overflow: auto;
  padding: 14px;
  border-radius: 6px;
  background: var(--code);
}
.MathJax_SVG_Display { overflow-x: auto; }
@media (max-width: 680px) {
  body { font-size: 15px; }
  .translated-article { padding: 18px; }
  .topbar { align-items: flex-start; flex-direction: column; }
}
"""
    (OUT_DIR / "style.css").write_text(css.lstrip(), encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    translator = Translator()
    items = []
    for article_dir in article_dirs():
        meta = load_json(article_dir / "metadata.json", {})
        href = build_article_page(article_dir, translator)
        items.append({"href": href, "meta": meta})
        print(f"built {href}")
    translator.save()
    write_css()
    build_index(items)
    print(f"site: {OUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
