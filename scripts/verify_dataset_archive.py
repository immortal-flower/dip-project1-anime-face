"""Verify extracted files against ZIP CRCs; optionally extract missing files only."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
import zlib


def verify(archive, destination, extract_missing=False):
    base = Path(destination).resolve()
    missing, mismatched, extracted = [], [], []
    with zipfile.ZipFile(archive) as z:
        entries = [e for e in z.infolist() if not e.is_dir()]
        targets = []
        for entry in entries:
            target = (base / entry.filename).resolve()
            if base not in target.parents or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Unsafe archive entry: {entry.filename}')
            targets.append((entry, target))
        for i, (entry, target) in enumerate(targets):
            if not target.exists():
                if not extract_missing:
                    missing.append(entry.filename)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(entry) as source, target.open('xb') as out:
                    shutil.copyfileobj(source, out)
                extracted.append(entry.filename)
            crc = 0
            with target.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    crc = zlib.crc32(block, crc)
            if target.stat().st_size != entry.file_size or crc != entry.CRC:
                mismatched.append(entry.filename)
            if (i+1) % 2000 == 0:
                print(f'Verified {i+1}/{len(entries)}', flush=True)
    return dict(archive=str(archive), destination=str(base), files=len(entries),
                missing=missing, mismatched=mismatched, extracted=extracted,
                ok=not missing and not mismatched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--extract-missing', action='store_true')
    args = parser.parse_args()
    report = verify(args.archive, args.destination, args.extract_missing)
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'CRC verification: ok={report["ok"]}, files={report["files"]}')
    if not report['ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
