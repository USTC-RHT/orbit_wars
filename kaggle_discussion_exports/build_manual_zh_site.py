#!/usr/bin/env python3
import html
import json
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "zh_manual_site"


def load_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def article_dirs():
    return sorted([p for p in ROOT.iterdir() if p.is_dir() and re.match(r"^\d+_", p.name)])


def render_inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def markdown_to_html(markdown, article_dir):
    lines = markdown.splitlines()
    out = []
    para = []
    in_math = False
    math = []

    def flush_para():
        if para:
            out.append("<p>" + render_inline(" ".join(para).strip()) + "</p>")
            para.clear()

    for line in lines:
        stripped = line.strip()
        if stripped == "$$":
            if in_math:
                out.append('<div class="formula"><pre>$$\n' + html.escape("\n".join(math), quote=False) + "\n$$</pre></div>")
                math.clear()
                in_math = False
            else:
                flush_para()
                in_math = True
            continue
        if in_math:
            math.append(line)
            continue
        if not stripped:
            flush_para()
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            flush_para()
            level = len(m.group(1))
            out.append(f"<h{level}>{render_inline(m.group(2))}</h{level}>")
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if m:
            flush_para()
            alt, src = m.groups()
            out.append(f'<figure><img src="../{article_dir.name}/{html.escape(src, quote=True)}" alt="{html.escape(alt, quote=True)}"><figcaption>{html.escape(alt, quote=False)}</figcaption></figure>')
            continue
        para.append(stripped)
    flush_para()
    return "\n".join(out)


def shell(title, body):
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


def write_css():
    css = """
:root {
  --bg: #f6f7f9;
  --paper: #fff;
  --text: #172033;
  --muted: #667085;
  --line: #dde3ea;
  --accent: #0b73d9;
  --soft: #e7f2ff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 16px/1.78 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", sans-serif;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { width: min(1060px, calc(100% - 32px)); margin: 0 auto; }
.topbar {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 12px max(16px, calc((100vw - 1060px) / 2));
  border-bottom: 1px solid var(--line);
  background: rgba(255,255,255,.94);
}
.hero { padding: 54px 0 26px; }
.hero h1 { margin: 0 0 12px; font-size: clamp(32px, 5vw, 52px); line-height: 1.08; }
.hero p { margin: 0; color: var(--muted); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; padding-bottom: 56px; }
.card {
  display: grid;
  gap: 8px;
  padding: 18px;
  min-height: 142px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--paper);
  color: var(--text);
}
.card.disabled { opacity: .58; pointer-events: none; }
.rank {
  display: inline-flex;
  width: max-content;
  padding: 2px 9px;
  border-radius: 999px;
  background: var(--soft);
  color: #07549f;
  font-size: 13px;
  font-weight: 700;
}
.article {
  margin: 24px auto 64px;
  padding: 30px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--paper);
}
.article h1 { font-size: clamp(30px, 4vw, 44px); line-height: 1.12; margin: 0 0 14px; }
.article h2 { margin: 1.8em 0 .65em; padding-top: 18px; border-top: 1px solid var(--line); font-size: 25px; }
.article h3 { margin: 1.5em 0 .5em; font-size: 20px; }
.article p { margin: 0 0 1em; }
.article img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 18px auto 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
}
figure { margin: 22px 0; }
figcaption { text-align: center; color: var(--muted); font-size: 14px; }
code, pre {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: #f1f4f8;
  border-radius: 5px;
}
code { padding: 2px 5px; }
pre { overflow-x: auto; padding: 14px; }
.formula { text-align: center; }
.formula pre { display: inline-block; text-align: left; }
@media (max-width: 700px) {
  .article { padding: 18px; }
  .topbar { flex-direction: column; }
}
"""
    (OUT / "style.css").write_text(css.lstrip(), encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    cards = []
    for d in article_dirs():
        meta = load_json(d / "metadata.json", {})
        translation = d / "translation_zh.md"
        if translation.exists():
            article_html = markdown_to_html(translation.read_text(encoding="utf-8"), d)
            page = shell(
                meta.get("title", d.name),
                f'<nav class="topbar"><a href="index.html">← 返回目录</a><a href="{html.escape(meta.get("url", "#"), quote=True)}" target="_blank" rel="noreferrer">Kaggle 原帖</a></nav><main class="wrap article">{article_html}</main>',
            )
            href = d.name + ".html"
            (OUT / href).write_text(page, encoding="utf-8")
            disabled = ""
            status = "已完成"
        else:
            href = "#"
            disabled = " disabled"
            status = "待翻译"
        cards.append(
            f'<a class="card{disabled}" href="{href}"><span class="rank">{html.escape(meta.get("rank", ""), quote=False)}</span><strong>{html.escape(meta.get("title", d.name), quote=False)}</strong><span>{html.escape(meta.get("author", ""), quote=False)} · {status}</span></a>'
        )
    write_css()
    index = shell(
        "Orbit Wars 获奖方案中文人工翻译",
        f'<main class="wrap"><header class="hero"><h1>Orbit Wars 获奖方案中文人工翻译</h1><p>人工按技术语境翻译，保留图片和公式位置；资源使用相对路径，便于迁移。</p></header><section class="grid">{"".join(cards)}</section></main>',
    )
    (OUT / "index.html").write_text(index, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
