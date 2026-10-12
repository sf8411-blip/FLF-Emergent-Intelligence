import pathlib,sys,tempfile,json,subprocess,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v420_dress as d
from flf_ecf import v420_repair as r

class V420DressRepairTests(unittest.TestCase):
    def test_dress_ranges_are_seven_distinct_unseen_banks(self):
        sets=[set(range(b,b+32)) for b in d.SEED_BASE.values()]
        self.assertEqual(len(sets),7)
        for i,a in enumerate(sets):
            self.assertEqual(len(a),32)
            for b in sets[i+1:]:self.assertTrue(a.isdisjoint(b))

    def test_typed_grammar_has_no_hidden_template_and_text_is_safe(self):
        self.assertTrue(set(r.FORBIDDEN_TEMPLATE_OPS).isdisjoint(r.PRIMITIVES))
        rng=np.random.default_rng(7)
        for _ in range(100):
            e=r.random_expr(('x0','x1','u0','u1'),rng)
            self.assertTrue(r.structure_is_clean(e));self.assertIsInstance(e.text(),str)

    def test_actual_corruption_is_circular_shift_and_zero_oracle(self):
        for fam,basefam,seed in [('corrupt_circular','circular',610001),('corrupt_rank','rank',610002)]:
            e0,y0,h0,m0=r.make_world(basefam,seed);e,y,h,m=r.make_world(fam,seed)
            self.assertTrue(np.allclose(np.sort(y),np.sort(y0)))
            self.assertTrue(np.allclose(h,0.0));self.assertEqual(m['corruption'],'circular_shift_y')
            for k in m0:self.assertAlmostEqual(float(m[k]),float(m0[k]))

    def test_run_seed_and_validator_share_non_gating_core(self):
        seed=611001
        row=d.run_seed('corrupt_circular',seed_override=seed,challenge_B=9,search_budget=3)
        core=r.run_seed_core('corrupt_circular',seed,challenge_B=9,search_budget=3)
        self.assertFalse(row['trigger_is_gate']);self.assertFalse(core['trigger_is_gate'])
        self.assertAlmostEqual(row['trigger_observed'],core['trigger_observed'])
        self.assertEqual(row['search_counts'],core['search_counts'])
        self.assertEqual(row['guided_program'],core['guided']['program'])

    def test_every_family_executes_all_search_arms_on_non_dress_smoke(self):
        for j,f in enumerate(d.FAMILIES):
            row=d.run_seed(f,seed_override=612000+j,challenge_B=9,search_budget=2)
            self.assertFalse(row['trigger_is_gate'])
            self.assertTrue(all(int(x)>0 for x in row['search_counts'].values()))

if __name__=='__main__':unittest.main()
