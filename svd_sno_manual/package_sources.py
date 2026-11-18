"""Package editable manual sources and ready illustrations for XeLaTeX."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json

ROOT=Path(__file__).resolve().parent


def main():
    files=[ROOT/n for n in ('main.tex','build.sh','README.md','QA.md')]
    files+=sorted(ROOT.glob('*.py'))
    files+=sorted((ROOT/'chapters').glob('*.tex'))
    files+=sorted((ROOT/'assets').glob('*.pdf'))
    files+=sorted((ROOT/'assets').glob('*.json'))
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest_path=ROOT/'source_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    files.append(manifest_path)
    archive=ROOT/'output/svd_sno_latex_sources.zip'
    archive.parent.mkdir(exist_ok=True)
    with ZipFile(archive,'w',compression=ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:z.write(p,str(Path(ROOT.name)/p.relative_to(ROOT)))
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for p in ROOT.glob('chapters/*.tex'):
            assert str(Path(ROOT.name)/p.relative_to(ROOT)) in z.namelist()
    print(f'{archive}: {len(files)} files, {archive.stat().st_size} bytes')


if __name__=='__main__':main()
