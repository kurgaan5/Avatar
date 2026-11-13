"""Fetch upstream model files and check SHA-256 when the manifest records it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avatar_paths import ROOT, RUNTIME


def run(*command):
    subprocess.run(command, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def download(entry):
    relative, url, expected = entry
    path = RUNTIME / relative
    if path.is_file():
        if not expected or sha256(path) == expected:
            print('Ready:', relative, flush=True)
            return
        raise ValueError(f'Existing file has the wrong SHA-256: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix+'.partial')
    print('Downloading:', relative, flush=True)
    run('curl.exe', '--fail', '--location', '--silent', '--show-error', '--retry', '3',
        '--connect-timeout', '20', '--output', str(partial), url)
    if expected and sha256(partial) != expected:
        raise ValueError(f'SHA-256 mismatch for {relative}; inspect docs/models.md')
    partial.replace(path)


def checkout(entry):
    name, url, revision = entry
    dest = RUNTIME / name
    if not dest.exists():
        run('git', 'init', str(dest))
    run('git', '-C', str(dest), 'fetch', '--depth=1', url, revision)
    run('git', '-C', str(dest), 'checkout', '--detach', revision)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=['portrait', 'speaker', 'voice', 'dialogue', 'all'])
    args = parser.parse_args()
    if not shutil.which('curl.exe'):
        parser.error('curl.exe is required.')
    manifest = json.loads((ROOT/'models.json').read_text(encoding='utf-8'))
    for name, group in manifest.items():
        if args.group != 'all' and name != args.group:
            continue
        for entry in group.get('repositories', []):
            checkout(entry)
        for entry in group['files']:
            download(entry)
