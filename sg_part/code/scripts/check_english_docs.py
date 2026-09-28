#!/usr/bin/env python3
"""Fail if Markdown or text documentation contains CJK characters."""
from pathlib import Path
import argparse

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('.')); a=ap.parse_args(); bad=[]
 for p in a.root.rglob('*'):
  if not p.is_file() or p.suffix.lower() not in {'.md','.txt','.rst'}: continue
  try:s=p.read_text(encoding='utf-8')
  except UnicodeDecodeError: continue
  if any('\u4e00'<=c<='\u9fff' for c in s): bad.append(str(p))
 if bad:
  print('Non-English documentation files:'); print('\n'.join(bad)); raise SystemExit(1)
 print('English documentation check passed')
if __name__=='__main__':main()
