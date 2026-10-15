"""Standard-library-only installer and launcher; no manual pip commands needed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['full', 'smoke', 'verify'], default='full',
                        help='full: refit all models; smoke: two origins; verify: check supplied results')
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error('This submission needs 64-bit Python 3.12 or newer for the pinned thesis packages. '
                     'Install a recent Python 3, then run this file again.')
    if sys.maxsize <= 2**32:
        parser.error('Please use a 64-bit Python installation for the scientific packages.')
    environment = ROOT / '.venv'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        print('Creating a private Python environment in .venv ...', flush=True)
        try:
            venv.EnvBuilder(with_pip=True).create(environment)
        except Exception as error:
            raise RuntimeError('Could not create .venv. Use a writable, extracted folder and a full '
                               'Python installation with venv and pip support. Details: ' + str(error)) from error
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['MPLBACKEND'] = 'Agg'
    env['MPLCONFIGDIR'] = str(ROOT / '.runtime' / 'matplotlib')
    env['XDG_CACHE_HOME'] = str(ROOT / '.runtime' / 'cache')
    env['PIP_CACHE_DIR'] = str(ROOT / '.runtime' / 'pip-cache')
    # These also prevent machine-dependent thread oversubscription during fitting.
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[name] = '1'
    requirements = ROOT / 'requirements.txt'
    stamp = environment / '.thesis-requirements.json'
    expected = hashlib.sha256(requirements.read_bytes()).hexdigest()
    installed = json.loads(stamp.read_text()) if stamp.exists() else {}
    # Check actual installed versions as well as the installation stamp. Once
    # installed, subsequent runs work without an Internet connection.
    probe = """import importlib.metadata as m, pathlib, sys
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
 line=line.split('#',1)[0].strip()
 if not line: continue
 if ';' in line:
  line,marker=line.split(';',1)
  if 'win32' in marker and sys.platform != 'win32': continue
 name,want=line.strip().split('==')
 if m.version(name.strip()) != want.strip(): raise SystemExit(1)
"""
    ready = subprocess.run([str(python), '-c', probe, str(requirements)], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not ready or installed.get('requirements_sha256') != expected:
        print('Installing the pinned dependencies (Internet needed on first run) ...', flush=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
                        '--only-binary=:all:', '-r', str(requirements)], check=True, cwd=ROOT, env=env)
        subprocess.run([str(python), '-m', 'pip', 'check'], check=True, cwd=ROOT, env=env)
        stamp.write_text(json.dumps({'requirements_sha256': expected}, indent=2) + '\n')
    else:
        print('The private environment already has the required dependencies.', flush=True)
    print('Starting thesis analysis: ' + args.mode, flush=True)
    return subprocess.run([str(python), str(ROOT / 'run_thesis.py'), '--mode', args.mode],
                          cwd=ROOT, env=env).returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print('\nSetup failed: ' + str(error), file=sys.stderr)
        print('Check Internet access, the Python version, and that the folder is extracted and writable.',
              file=sys.stderr)
        raise SystemExit(1)
