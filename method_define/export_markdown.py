"""Export the authoritative structured manuscript as readable Markdown."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
lines=[]
for page in json.loads((root/'chapters.json').read_text())['pages']:
 lines += ['# '+page['title'],'']
 for b in page['blocks']:
  kind=b['type']
  if kind in ('p','h1','h2','h3'):lines += [('#'*int(kind[-1])+' ' if kind.startswith('h') else '')+b['text'],'']
  elif kind=='equation':lines += ['$$',b['latex'],'$$',b.get('caption',''),'']
  elif kind=='figure':lines += [f"![{b.get('caption','Схема')} ]({b['path']})",b.get('caption',''),'']
  elif kind=='table':
   lines += ['| '+' | '.join(b['headers'])+' |','| '+' | '.join(['---']*len(b['headers']))+' |']
   lines += ['| '+' | '.join(map(str,row))+' |' for row in b['rows']];lines += ['']
  elif kind=='code':lines += ['```',b['text'],'```','']
(root/'METHOD.md').write_text('\n'.join(lines))
