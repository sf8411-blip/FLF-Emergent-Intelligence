import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v420_dev as v
from flf_ecf import v420_dress as d
from flf_ecf import v420_eval as ev

class V420DressTests(unittest.TestCase):
    def test_dev_ranges_are_distinct_and_exactly_32(self):
        sets=[set(range(b,b+32)) for b in d.SEED_BASE.values()]
        self.assertEqual(len(sets),6)
        for i,a in enumerate(sets):
            for b in sets[i+1:]:self.assertTrue(a.isdisjoint(b))

    def test_corruption_preserves_y_marginal_but_breaks_alignment(self):
        for fam,basefam,seed in [('corrupt_circular','circular',323000),('corrupt_rank','rank',324000)]:
            env0,y0,h0,m0=v.make_world(basefam,seed)
            env,y,h,m=d.make_dress_world(fam,seed)
            self.assertTrue(np.allclose(np.sort(y),np.sort(y0)))
            self.assertTrue(np.allclose(h,0.0))
            for k in m0:self.assertEqual(m[k],m0[k])
            self.assertIn('shift',m)

    def test_fresh_corrupt_challenge_keeps_latent_meta(self):
        env,y,h,m=d.make_dress_world('corrupt_circular',323001)
        env2,y2,h2,m2=d._fresh_challenge_world('corrupt_circular',m,999001)
        self.assertAlmostEqual(m['period'],m2['period'])
        self.assertTrue(np.allclose(h2,0.0))
        self.assertGreater(np.std(env2['u0']),1.2*np.std(env['u0']))

    def test_b2a_b2b_are_selected_from_one_random_bank(self):
        env,y,h,m=v.make_world('null',322002);base=d._gated_base(env,y,h)
        a,b,sigs=d._select_random_controls(env,y,base,777)
        self.assertIn(ev.program_signature(a[0]),sigs)
        self.assertIn(ev.program_signature(b[0]),sigs)
        self.assertEqual(len(sigs),d.CANDIDATES)

if __name__=='__main__':unittest.main()
