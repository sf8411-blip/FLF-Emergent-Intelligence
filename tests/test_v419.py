import inspect, pathlib, sys, unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v419_learner as learner
from flf_ecf.v419_dsl import evaluate,l0_expressions
from flf_ecf.v419_worlds import make_world,hidden_feature,reference_expr

class V419AntiCircularity(unittest.TestCase):
    def test_information_barrier_source(self):
        src=inspect.getsource(learner).lower()
        for bad in ('v419_worlds','hidden_feature','reference_expr','world_family','operator_id','oracle_feature'):
            self.assertNotIn(bad,src)

    def test_reference_depth_at_least_four(self):
        self.assertGreaterEqual(reference_expr('circular').depth(),4)
        self.assertGreaterEqual(reference_expr('rank_interaction').depth(),4)

    def test_l0_nonapproximation_diagnostic(self):
        for fam in ('circular','rank_interaction'):
            errs=[]
            for s in range(1200,1212):
                env,u,y=make_world(fam,s,420);h=hidden_feature(fam,env);exprs=l0_expressions(sorted(env))
                X=np.c_[np.ones(len(h)),*[evaluate(e,env) for e in exprs]]
                c=np.linalg.lstsq(X,h,rcond=None)[0];p=X@c
                errs.append(np.sqrt(np.mean((h-p)**2))/(np.std(h)+1e-9))
            self.assertGreaterEqual(float(np.mean(errs)),.20)

    def test_reference_is_reachable_and_deep(self):
        for fam in ('circular','rank_interaction'):
            e=reference_expr(fam);errs=[]
            for s in range(1300,1312):
                env,u,y=make_world(fam,s,420);a=evaluate(e,env);b=hidden_feature(fam,env)
                errs.append(np.sqrt(np.mean((a-b)**2))/(np.std(b)+1e-9))
            self.assertLessEqual(float(np.mean(errs)),.02)

if __name__=='__main__':unittest.main()
