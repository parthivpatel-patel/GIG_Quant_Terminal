"""Verify an exact public GitHub source archive, including sparse checkout paths."""
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

repo = Path.cwd()
out = Path(sys.argv[1])
out.mkdir(exist_ok=False)
sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
skipped = {r[2:].decode() for r in subprocess.check_output(['git', 'ls-files', '-t', '-z']).split(b'\0') if r.startswith(b'S ')}
expected = {}
local_verified = 0
for record in subprocess.check_output(['git', 'ls-tree', '-rz', sha]).split(b'\0'):
    if not record:
        continue
    meta, name = record.split(b'\t', 1)
    mode, kind, oid = meta.decode().split()
    name = name.decode()
    assert kind == 'blob' and mode in {'100644', '100755'}, name
    path = repo / name
    if path.exists():
        content = path.read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        assert actual == oid, name
        assert ('100755' if path.stat().st_mode & 0o111 else '100644') == mode, name
        local_verified += 1
    else:
        assert name in skipped, ('unexpected missing local file', name)
    expected[name] = (oid, mode)
url = f'https://codeload.github.com/parthivpatel-patel/GIG_Quant_Terminal/tar.gz/{sha}'
with urllib.request.urlopen(url, timeout=120) as response:
    archive_bytes = response.read(256*1024*1024+1)
assert len(archive_bytes) <= 256*1024*1024
seen = set()
restored = []
verified_sha256 = {}
root = out / 'restored-source'
root.mkdir()
with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r|gz') as archive:
    prefix = None
    for entry in archive:
        prefix = prefix or entry.name.split('/')[0]
        if entry.isdir():
            continue
        assert entry.isfile() and not entry.issym() and not entry.islnk(), entry.name
        name = entry.name[len(prefix)+1:]
        assert name in expected and '..' not in Path(name).parts, name
        content = archive.extractfile(entry).read()
        oid = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        mode = '100755' if entry.mode & 0o111 else '100644'
        assert (oid, mode) == expected[name], name
        seen.add(name)
        verified_sha256[name] = hashlib.sha256(content).hexdigest()
        if name.split('/')[0] in {'src', 'tests', 'scripts', 'configs', '.github'} or name in {'pyproject.toml', 'setup.py', 'README.md', 'LICENSE'}:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(content)
            target.chmod(int(mode, 8) & 0o777)
            restored.append(name)
assert seen == set(expected)
report = {'state': 'PASS', 'published_head': sha, 'source_url': url,
          'archive_sha256': hashlib.sha256(archive_bytes).hexdigest(),
          'archive_verified_files_and_modes': len(seen), 'local_present_files_verified': local_verified,
          'sparse_paths_verified_against_git_blobs_in_archive': len(skipped),
          'source_test_config_files_restored': len(restored), 'archive_retained_locally': False,
          'method': 'Git tree blob hashes/modes matched against every downloaded entry; present working files verified; sparse paths verified in archive; source/test/config subset restored',
          'external_provider': 'GitHub only'}
(out / 'recovery.json').write_text(json.dumps(report, indent=2)+'\n')
(out / 'archive-file-sha256.json').write_text(json.dumps(verified_sha256, indent=2)+'\n')
print(json.dumps(report, indent=2))
