"""OpenAIs Codex was used to help with the code"""
"""Portable entry point: verify inputs, reproduce monthly and combined results."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_submission import check_integrity, check_reference, compare_predictions


def run_stage(label, command, log_dir):
    print('\n' + label, flush=True)
    with (log_dir / (label.split(':')[0] + '.log')).open('w', encoding='utf-8') as log:
        log.write('Command: ' + repr(command) + '\n')
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise RuntimeError(label + ' failed; see ' + str(log_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['full', 'smoke', 'verify'], default='full')
    args = parser.parse_args()
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime' / 'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.runtime' / 'cache'))
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    started = datetime.now(timezone.utc)
    log_dir = ROOT / 'logs' / (started.strftime('%Y%m%dT%H%M%S') + '_' + args.mode)
    log_dir.mkdir(parents=True, exist_ok=False)
    print('Checking submitted files and reconstructing data from the supplied raw sources ...', flush=True)
    record = dict(mode=args.mode, integrity_files_checked=check_integrity(), reference_checks=check_reference())
    if args.mode != 'verify':
        # A new destination for every run avoids mixing complete and partial runs.
        output = ROOT / 'results' / ('smoke' if args.mode == 'smoke' else 'reproduced') / started.strftime('%Y%m%dT%H%M%S')
        monthly = output / 'monthly'
        combined = output / 'combined'
        core = [sys.executable, '-u', str(ROOT / 'scripts' / 'current_scope' / 'run.py'),
                '--project', str(ROOT), '--output', str(monthly)]
        if args.mode == 'smoke':
            core += ['--limit', '2']
        run_stage('01_monthly: original, Paye, baseline and macro models', core, log_dir)
        command = [sys.executable, '-u', str(ROOT / 'scripts' / 'run_combined.py'),
                   '--input-dir', str(monthly if args.mode == 'full' else ROOT / 'results' / 'reference' / 'monthly'),
                   '--output', str(combined)]
        if args.mode == 'smoke':
            command += ['--limit', '2']
        run_stage('02_combined: 108-predictor extension', command, log_dir)
        generated_monthly = monthly / 'smoke_all_2' if args.mode == 'smoke' else monthly
        record['monthly_comparison'] = compare_predictions(generated_monthly, ROOT / 'results' / 'reference' / 'monthly', args.mode == 'smoke')
        record['combined_comparison'] = compare_predictions(combined, ROOT / 'results' / 'reference' / 'combined', args.mode == 'smoke')
        record['generated_results'] = str(output.relative_to(ROOT))
        print('\nNew results: ' + str(output), flush=True)
    record['completed_utc'] = datetime.now(timezone.utc).isoformat()
    record['elapsed_seconds'] = (datetime.now(timezone.utc) - started).total_seconds()
    record['status'] = 'passed'
    record['integrity_files_checked_after_run'] = check_integrity()
    (log_dir / 'validation.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('\nCompleted successfully. Validation: ' + str(log_dir / 'validation.json'), flush=True)
    print('Submitted reference results were preserved.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, AssertionError) as error:
        print('\nAnalysis failed: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
