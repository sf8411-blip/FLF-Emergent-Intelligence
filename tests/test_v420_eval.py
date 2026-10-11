import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v420_dev as v
from flf_ecf import v420_eval as e

class V420EvalTests(unittest.TestCase):
    def test_pooled_ratio_is_ratio_of_sums(self):
        rows=[{'num':1.,'den':2.},{'num':9.,'den':18.}]
        self.assertAlmostEqual(e.pooled_ratio(rows),.5)
        lo,hi=e.bootstrap_pooled(rows,B=200,seed=7);self.assertLessEqual(lo,.5);self.assertGreaterEqual(hi,.5)
    def test_in_model_ablation_no_refit(self):
        y=np.linspace(0,1,100);base=np.zeros(100);z=y.copy();idx=np.arange(60,100);beta=1.0
        original=(beta,z.copy(),base.copy())
        err=e.in_model_block_ablation(y,base,beta,z,idx,13,block=5)
        self.assertGreaterEqual(err,0.0);self.assertEqual(beta,original[0]);self.assertTrue(np.array_equal(z,original[1]));self.assertTrue(np.array_equal(base,original[2]))
    def test_world_trace_similarity_and_substitutability(self):
        x=np.linspace(-2,2,200);self.assertGreater(e.pairwise_abs_corr([x,2*x+1,-3*x]),.99);self.assertGreater(e.predictive_substitutability(x,-2*x),.99)
    def test_R8_same_bank_and_B3_forced_when_trigger_closed(self):
        env,y,h,_=v.make_world('circular',191001);base=v.b0_and_oracle(env,y,h);base['_y']=y
        a=e.control_candidates(env,base,551,trigger_open=False,count=8)
        b=e.control_candidates(env,base,551,trigger_open=True,count=8)
        self.assertEqual(a['bank_signature'],b['bank_signature'])
        self.assertTrue(a['b3_ran']);self.assertFalse(a['trigger_open_input'])
        self.assertIn(e.program_signature(a['B2a'][1]),a['bank_signature'])
        self.assertIn(e.program_signature(a['B2b'][1]),a['bank_signature'])

if __name__=='__main__':unittest.main()
