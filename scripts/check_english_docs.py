#!/usr/bin/env python3
from pathlib import Path
import argparse

def main():
 ap=argparse.ArgumentParser(description='Fail if Markdown or text documentation contains CJK characters.'); ap.add_argument('--root',type=Path,default=Path('.')); a=ap.parse_args(); bad=[]
 skip={'data','results','outputs','logs','.venv','.git'}
 for p in a.root.rglob('*'):
  if not p.is_file() or p.suffix.lower() not in {'.md','.txt','.rst'}: continue
  if p.relative_to(a.root).parts[0] in skip: continue
  try:s=p.read_text(encoding='utf-8')
  except UnicodeDecodeError: continue
  if any('\u4e00'<=c<='\u9fff' for c in s): bad.append(str(p))
 if bad:
  print('Non-English documentation files:'); print('\n'.join(bad)); raise SystemExit(1)
 print('English documentation check passed')
if __name__=='__main__':main()
