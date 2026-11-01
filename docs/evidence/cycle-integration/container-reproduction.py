"""Local-only restart verification of the newly installed wheel; no host ports."""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

BASE = "sha256:1931078c61abb38a4ddd2fe68ad671b9828314ee4bea3c27babf5ca14f64a339"
EVIDENCE = Path('/workspace/work/evidence')
WHEEL = Path('/workspace/work/cycle-wheels/gig-0.1.0-cp312-cp312-linux_x86_64.whl')
ENV = {k: v for k, v in os.environ.items() if k not in {
    'DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH'}}
PREFIX = ['docker', '--host=unix:///var/run/docker.sock']
TAG = 'gig-validated-804e96f'
VOLUME = 'gig-cycle-restart-wheel-20261101'


def docker(*args):
    result = subprocess.run(PREFIX + list(args), env=ENV, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[:4000])
    return result.stdout.strip()


def request(container, path, payload=None):
    code = '''import json,urllib.request
path,payload=json.loads(__import__('sys').argv[1])
data=None if payload is None else json.dumps(payload).encode()
request=urllib.request.Request('http://127.0.0.1:8000'+path,data=data,headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=10) as response: print(response.read().decode())
'''
    return json.loads(docker('exec', container, 'python', '-c', code, json.dumps([path, payload])))


def start(name):
    docker('run', '-d', '--name', name, '--network', 'none', '--read-only',
           '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m', '--memory', '1g', '--cpus', '1',
           '--pids-limit', '128', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
           '--mount', f'type=volume,src={VOLUME},dst=/app/data',
           '--mount', 'type=bind,src=/workspace/work/cycle-wheel-installed/gig,dst=/usr/local/lib/python3.12/site-packages/gig,readonly',
           '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=1',
           '-e', 'GIG_DATA_DIR=/app/data', '-e', 'GIG_BROKER_CONNECTIVITY_ENABLED=false',
           TAG, 'python', '-c', 'import uvicorn;from gig.service.app import create_app;uvicorn.run(create_app(),host="127.0.0.1",port=8000)')
    for _ in range(100):
        try:
            if request(name, '/healthz')['ok']:
                return
        except (subprocess.CalledProcessError, KeyError, RuntimeError):
            time.sleep(.1)
    raise RuntimeError('Container health deadline exceeded')


assert docker('image', 'inspect', 'gig-validated-804e96f', '--format', '{{.Id}}') == BASE
docker('volume', 'create', VOLUME)
docker('run', '--rm', '--user', '0:0', '--network', 'none', '--cap-drop', 'ALL',
       '--cap-add', 'CHOWN', '--mount', f'type=volume,src={VOLUME},dst=/app/data',
       TAG, 'python', '-c', 'import os;os.chown("/app/data",10001,10001)')
names = ['gig-cycle-wheel-write', 'gig-cycle-wheel-read']
now = '2026-11-01T15:00:00Z'
query = '?mode=local&engine=execution-engine&as_of=2026-11-01T15%3A00%3A00Z&max_age_seconds=60'
try:
    start(names[0])
    assert docker('exec', names[0], 'id', '-u') == '10001'
    def operation(name, **values):
        result = request(names[0], '/api/v1/workspace/execution-engine',
                         dict(operation=name, now=now, **values))
        assert result['state'] == 'AVAILABLE', result
        return result
    operation('prepare', client_id='container-one', symbol='SYN001', side='buy', quantity='10',
              reference_price='100', price_asof=now, max_notional='1000',
              max_position_quantity='10', limit_price='100')
    ack = operation('dispatch')
    assert ack['data']['positions'] == {}
    fill = operation('fill', client_id='container-one', quantity='2', price='100')
    assert fill['data']['positions']['SYN001'] == '2'
    operation('reconcile')
    operation('kill-switch', engaged=True)
    before = {name: request(names[0], '/api/v1/terminal/' + name + query)
              for name in ('execution', 'portfolio', 'risk')}
    assert all(value['state'] == 'AVAILABLE' for value in before.values())
    docker('stop', '--time', '10', names[0])
    docker('rm', names[0])
    start(names[1])
    after = {name: request(names[1], '/api/v1/terminal/' + name + query) for name in before}
    assert before == after
    assert after['portfolio']['data']['rows'][0]['quantity'] == '2'
    assert after['risk']['data']['metrics']['kill_switch']
    assert after['risk']['data']['metrics']['nav'] is None
    report = {'result': 'PASS', 'base_image': BASE, 'image_id': docker('image', 'inspect', TAG, '--format', '{{.Id}}'),
              'wheel_sha256': hashlib.sha256(WHEEL.read_bytes()).hexdigest(),
              'architecture': 'x86_64', 'uid': 10001, 'network': 'none', 'published_ports': [],
              'read_only_root': True, 'persistent_restart_views': list(before),
              'content_hashes_before_after': {name: value['content_sha256'] for name, value in after.items()},
              'ack_position_count': 0, 'partial_fill_quantity': '2', 'kill_switch_persisted': True,
              'nav': None, 'broker_connections': 0,
              'scope': 'cached validated base with read-only bind of independently installed native wheel; not a new Docker image or ARM64 build'}
    (EVIDENCE / 'cycle-container-runtime.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
finally:
    for name in names:
        subprocess.run(PREFIX + ['stop', '--time', '10', name], env=ENV, capture_output=True)
