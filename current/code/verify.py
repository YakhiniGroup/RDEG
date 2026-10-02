"""Verify integrity and optionally reproduce the release in a separate directory."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--full',action='store_true');ap.add_argument('--out',type=Path);ap.add_argument('--inputs',type=Path,help='private archive with data/, results/ and manuscript/');args=ap.parse_args()
    count=0
    for line in (ROOT/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);p=ROOT/name
        if ROOT not in p.resolve().parents:raise ValueError('Manifest path outside package')
        got=hashlib.sha256(p.read_bytes()).hexdigest()
        if got!=digest:raise RuntimeError('Checksum mismatch: '+name)
        count+=1
    print(f'PASS: {count} file checksums',flush=True)
    if not args.full:return
    if args.out is None or args.inputs is None:ap.error('--full requires --out and --inputs; manuscript data are not distributed here')
    inputs=args.inputs.expanduser().resolve()
    for folder in ('data','results','manuscript'):
        if not (inputs/folder).is_dir():ap.error('Missing private input directory: '+str(inputs/folder))
    out=args.out.resolve()
    if out==ROOT or ROOT in out.parents or out==inputs or inputs in out.parents:ap.error('--out must be outside the code package and private input archive')
    out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['RDEG_ANALYSIS_ROOT']=str(inputs);env['PYTHONDONTWRITEBYTECODE']='1';env['MPLCONFIGDIR']=str(out/'matplotlib-cache');env['XDG_CACHE_HOME']=str(out/'cache')
    commands=[['reproduce.py','--all','--out',str(out/'analysis')],['validate_linear.py','--out',str(out/'algorithm')],['plot_results.py','--out',str(out/'figures')],['plot_timings.py','--out',str(out/'figures')]]
    for command in commands:
        subprocess.run([sys.executable,str(ROOT/'code'/command[0]),*command[1:]],check=True,cwd=out,env=env)
    print('PASS: full reproduction and independent checks',flush=True)
    (out/'STATUS.json').write_text(json.dumps({'status':'PASS','verified_manifest_files':count,'commands':commands},indent=2)+'\n')
if __name__=='__main__':main()
