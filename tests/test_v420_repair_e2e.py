import json,pathlib,subprocess,sys,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v420_dress as d

class V420RepairE2E(unittest.TestCase):
    def test_non_dress_run_seed_rows_merge_in_smoke_mode(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);rows=root/'rows';rows.mkdir();out=root/'summary.json';raw=root/'raw.jsonl'
            for j,f in enumerate(d.FAMILIES):
                row=d.run_seed(f,seed_override=613000+j,challenge_B=9,search_budget=2)
                (rows/f'{f}.json').write_text(json.dumps(row,sort_keys=True,allow_nan=False))
            cp=subprocess.run([sys.executable,str(ROOT/'merge_v420_dress.py'),'--rows',str(rows),'--out',str(out),'--raw',str(raw),'--smoke'],cwd=ROOT,text=True,capture_output=True)
            self.assertEqual(cp.returncode,0,msg=cp.stdout+'\n'+cp.stderr)
            summary=json.loads(out.read_text());self.assertTrue(summary['pass']);self.assertTrue(summary['trigger_non_gating']);self.assertTrue(summary['all_search_arms_ran'])

if __name__=='__main__':unittest.main()
