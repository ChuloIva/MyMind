import os
from pathlib import Path

REPO_ROOT   = Path("/Users/ivanculo/Desktop/Projects/MyMind")  # absolute or relative
OUTPUT_FILE = "repo_code_dump_mymind.md"

# ─── Collect all .py and .md files ─────────────────────────────────────────────
def collect_files(root_dir: Path, exts=(".py", ".md")):
    results = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]   # skip hidden dirs
        for name in files:
            if Path(name).suffix.lower() in exts:
                results.append(Path(root) / name)
    return results

# ─── Figure out the fence label we should use ─────────────────────────────────
def fence_lang(path: Path) -> str:
    return {
        ".py": "python",
        ".md": "markdown",
    }.get(path.suffix.lower(), "")   # empty → plain fences

# ─── Write the combined markdown dump ─────────────────────────────────────────
def write_markdown(files, output_path: Path):
    with output_path.open("w", encoding="utf-8") as md:
        for fp in sorted(files, key=str):
            lang = fence_lang(fp)
            md.write(f"\n## `{fp}`\n\n```{lang}\n")
            try:
                md.write(fp.read_text(encoding="utf-8"))
            except Exception as e:
                md.write(f"# Error reading file: {e}")
            md.write("\n```\n")

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    files = collect_files(REPO_ROOT)           # defaults to .py + .md
    write_markdown(files, Path(OUTPUT_FILE))
    print(f"✅ Code dumped to {OUTPUT_FILE}")