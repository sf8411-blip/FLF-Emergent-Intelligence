import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v420_dev as v
from flf_ecf import v420_protocol as p


class V420ProtocolTests(unittest.TestCase):
    def test_continuous_constant_search_and_mdl_charge(self):
        env,y,h,meta=v.make_world('circular',204001)
        base=v.b0_and_oracle(env,y,h)
        prog=v.Program('moddiff','x0',p=.91)
        fitted=p.tune_constants_continuous(prog,env,base['residual'],base['tr'])
        self.assertIsInstance(fitted.p,float)
        self.assertEqual(fitted.constants(),1)
        self.assertAlmostEqual(fitted.mdl_cost(),1.5)
        self.assertFalse(any(abs(fitted.p-x)<1e-12 for x in v.OLD_GRID_PERIODS))

    def test_same_world_challenge_preserves_mechanism_and_changes_action_amplitude(self):
        for fam,seed in [('circular',204101),('rank',204102)]:
            env,y,h,meta=v.make_world(fam,seed,action_scale=1.0)
            env2,y2,h2,meta2=p.make_trace_with_meta(fam,seed+900000,meta,action_scale=1.6)
            self.assertEqual(meta,meta2)
            # Different independent trace and a materially larger intervention scale.
            self.assertFalse(np.array_equal(env['u0'],env2['u0']))
            self.assertGreater(np.std(env2['u0']),1.35*np.std(env['u0']))

    def test_oracle_feature_passes_same_world_block_permutation_challenge(self):
        for fam,seed in [('circular',204201),('rank',204202)]:
            env,y,h,meta=v.make_world(fam,seed)
            prog=p.oracle_program(fam,meta)
            pv,gain,meta2=p.challenge_same_world(prog,fam,meta,seed+910000,B=999)
            self.assertEqual(meta,meta2)
            self.assertLessEqual(pv,.01)
            self.assertGreater(gain,0.0)

if __name__=='__main__':unittest.main()
