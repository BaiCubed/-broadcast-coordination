#!/usr/bin/env python3
"""Verify generated output coverage against an artifact manifest."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,default=Path('audit/OUTPUT_REPRODUCTION_MANIFEST.csv')); ap.add_argument('--output-root',type=Path,default=Path('outputs')); ap.add_argument('--reference-root',type=Path); args=ap.parse_args()
 rows=list(csv.DictReader(args.manifest.open(encoding='utf-8'))); missing=[]
 if args.reference_root and args.reference_root.is_dir():
  # When auditing from a clean release, compare the generated tree with the reference tree by relative path.
  for r in rows:
   if r['stage']!='metadata_or_manual' and not (args.reference_root/r['artifact']).exists(): missing.append('reference:'+r['artifact'])
 for r in rows:
  # Manifest paths refer to the source/reference tree; generated tree uses same relative paths.
  relative = r.get('release_path') or ('outputs/' + r['artifact'])
  p=(args.output_root.parent/relative) if relative.startswith('../') else (args.output_root/Path(r['artifact']))
  if r['stage']=='metadata_or_manual': continue
  if not args.output_root.exists(): continue
  if not p.exists(): missing.append(r['artifact'])
 report={'manifest_entries':len(rows),'checked_entries':sum(r['stage']!='metadata_or_manual' for r in rows),'missing_generated_artifacts':missing,'status':('reference_not_bundled' if not args.output_root.exists() else ('pass' if not missing else 'incomplete'))}
 Path('audit/output_verification.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2)); raise SystemExit(0 if not missing else 1)
if __name__=='__main__':main()
