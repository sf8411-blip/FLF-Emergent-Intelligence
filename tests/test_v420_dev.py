import pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v420_dev as v

class V420InstrumentTests(unittest.TestCase):
    def test_constant_charge_and_continuous_world_constants(self):
        self.assertEqual(v.MDL_CONSTANT_COST,.5)
        for fam in ('circular','rank'):
            _,_,_,m=v.make_world(fam,171234)
            if fam=='circular':self.assertTrue(all(abs(m['period']-x)>1e-9 for x in v.OLD_GRID_PERIODS))
            else:
                self.assertTrue(all(abs(m['window']-x)>1e-9 for x in v.OLD_GRID_WINDOWS))
                self.assertTrue(all(abs(m['quantile']-x)>1e-9 for x in v.OLD_GRID_QUANTILES))
    def test_corruption_preserves_length_not_alignment(self):
        e,y,h,_=v.make_world('circular',171235);ec,yc,hc,_=v.make_world('corrupt_circular',171235)
        self.assertEqual(len(y),len(yc));self.assertAlmostEqual(float(y.mean()),float(yc.mean()),places=12)
        self.assertLess(abs(v.corr(yc,hc)),abs(v.corr(y,h)))
    def test_same_program_sampler_for_controls(self):
        env,_,_,_=v.make_world('rank',171236);r=__import__('numpy').random.default_rng(1)
        p=v.sample_program(env,r);self.assertIn(p.kind,{'moddiff','rankprod','mean','prod','diff','modraw'})
    def test_novelty_threshold_and_challenge_fixed(self):
        self.assertEqual(v.NOVELTY_R2_MAX,.5);self.assertEqual(v.VALIDATION_GAIN_MIN,.05);self.assertEqual(v.CHALLENGE_P_MAX,.01)

if __name__=='__main__':unittest.main()
