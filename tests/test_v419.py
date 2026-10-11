import hashlib, inspect, pathlib, sys, unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v419_learner as learner, v419_dsl as dsl, v419_v418_adapter as adapter, v419_runner as runner
from flf_ecf.v419_dsl import evaluate,l0_expressions
from flf_ecf.v419_worlds import make_world,hidden_feature,reference_expr

class V419AntiCircularity(unittest.TestCase):
    def test_information_barrier_source(self):
        src=inspect.getsource(learner).lower()
        for bad in ('v419_worlds','hidden_feature','reference_expr','world_family','operator_id','oracle_feature'):
            self.assertNotIn(bad,src)

    def test_preregistration_hash_is_runtime_computable(self):
        data=(ROOT/'docs/v4.19_preregistration.md').read_bytes()
        self.assertEqual(len(hashlib.sha256(data).hexdigest()),64)

    def test_dsl_has_no_missing_operator_menu(self):
        src=inspect.getsource(dsl).lower()
        for bad in ('circular','unwrap','rank_interaction','oracle_feature','world_family'):
            self.assertNotIn(bad,src)

    def test_v418_adapter_uses_unchanged_v418_machinery(self):
        src=inspect.getsource(adapter)
        self.assertIn('from .v418 import G, Model, rg, mut, cross',src)
        self.assertEqual(runner.DOWNSTREAM_BUDGET,48)

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

    def test_t1_t2_are_reachability_gated(self):
        src=(ROOT/'run_v419.py').read_text()
        self.assertIn("if a.tier in ('T1','T2')",src)
        self.assertIn('Reachability audit missing; T1/T2 blocked.',src)
        self.assertIn('refusing silent downscale',src)

    def test_final_merge_bootstraps_functional_rediscovery_at_seed_level(self):
        src=(ROOT/'merge_v419.py').read_text()
        self.assertIn('def functional_seed_boot_ci',src)
        self.assertIn('ix=r.integers(0,n,n)',src)
        self.assertIn("'functional_similarity_ci':functional_seed_boot_ci(sig)",src)
        self.assertNotIn("boot_ci(sims,'median')",src)

    def test_final_merge_enforces_frozen_t2_integrity_metadata(self):
        src=(ROOT/'merge_v419.py').read_text()
        self.assertIn("r.get('tier')!='T2'",src)
        self.assertIn("not r.get('frozen_test')",src)
        for key in ('preregistration_git_blob_sha','code_commit_shas','seed_shards','candidate_evaluations_total','rng_state_refs'):
            self.assertIn(key,src)

if __name__=='__main__':unittest.main()
