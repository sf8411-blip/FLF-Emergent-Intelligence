import hashlib,inspect,pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v419_learner as learner,v419_dsl as dsl,v419_v418_adapter as adapter,v419_runner as runner
from flf_ecf.v419_dsl import evaluate,l0_expressions,var
from flf_ecf.v419_worlds import make_world,hidden_feature,reference_expr

class V419AntiCircularity(unittest.TestCase):
    def test_information_barrier_source(self):
        src=inspect.getsource(learner).lower()
        for bad in ('v419_worlds','hidden_feature','reference_expr','world_family','operator_id','oracle_feature'):
            self.assertNotIn(bad,src)

    def test_preregistration_hash_is_runtime_computable(self):
        data=(ROOT/'docs/v4.19_preregistration.md').read_bytes();self.assertEqual(len(hashlib.sha256(data).hexdigest()),64)
        self.assertIn(b'Amendment A3',data)

    def test_dsl_has_no_missing_operator_menu(self):
        src=inspect.getsource(dsl).lower()
        for bad in ('circular','unwrap','rank_interaction','oracle_feature','world_family'):
            self.assertNotIn(bad,src)

    def test_v418_adapter_uses_unchanged_v418_machinery(self):
        src=inspect.getsource(adapter);self.assertIn('from .v418 import G, Model, rg, mut, cross',src);self.assertEqual(runner.DOWNSTREAM_BUDGET,48)

    def test_a3_uniform_extension_slot_is_literal(self):
        env,u,y=make_world('rank_interaction',1600,160)
        # Use a generic DSL expression as the required extension; selection must keep it literally.
        required=reference_expr('rank_interaction')
        chosen=adapter.select_three(l0_expressions(sorted(env)),env,y,required=[required])
        self.assertEqual(len(chosen),3);self.assertEqual(chosen[0].text(),required.text())
        b0=adapter.select_three(l0_expressions(sorted(env)),env,y,required=[])
        self.assertEqual(len(b0),3);self.assertNotIn(required.text(),[e.text() for e in b0])

    def test_reference_depth_at_least_four(self):
        self.assertGreaterEqual(reference_expr('circular').depth(),4);self.assertGreaterEqual(reference_expr('rank_interaction').depth(),4)

    def test_l0_nonapproximation_on_a3_diagnostics(self):
        for fam in ('circular','rank_interaction'):
            errs=[]
            for s in range(12000,12012):
                env,u,y=make_world(fam,s,420);h=hidden_feature(fam,env);exprs=l0_expressions(sorted(env))
                X=np.c_[np.ones(len(h)),*[evaluate(e,env) for e in exprs]];c=np.linalg.lstsq(X,h,rcond=None)[0];p=X@c
                errs.append(np.sqrt(np.mean((h-p)**2))/(np.std(h)+1e-9))
            self.assertGreaterEqual(float(np.mean(errs)),.20)

    def test_reference_is_reachable_on_a3_diagnostics(self):
        for fam in ('circular','rank_interaction'):
            e=reference_expr(fam);errs=[]
            for s in range(12000,12012):
                env,u,y=make_world(fam,s,420);a=evaluate(e,env);b=hidden_feature(fam,env)
                errs.append(np.sqrt(np.mean((a-b)**2))/(np.std(b)+1e-9))
            self.assertLessEqual(float(np.mean(errs)),.02)

    def test_t1_t2_are_reachability_gated_and_budget_frozen(self):
        src=(ROOT/'run_v419.py').read_text()
        self.assertIn("if a.tier in ('T1','T2')",src);self.assertIn('Reachability audit missing; T1/T2 blocked.',src)
        self.assertIn("budget!=prereg_budget",src.replace(' ',''));self.assertIn('override refused',src)

    def test_a3_fresh_seed_ranges_replace_retired_a2_ranges(self):
        src=(ROOT/'run_v419.py').read_text()
        for token in ('81000','82000','83000','91000','91100','1600','10000','91200','91600'):self.assertIn(token,src)
        for retired in ('61000','62000','63000','71000','71100'):self.assertNotIn(retired,src)
        rsrc=(ROOT/'run_v419_reachability.py').read_text();self.assertIn('range(12000,12012)',rsrc);self.assertNotIn('range(8000,8012)',rsrc)

    def test_residual_trigger_is_exactly_three_preregistered_components(self):
        src=inspect.getsource(learner.trigger_components)
        self.assertIn('_compression_gap_z',src);self.assertIn('score=ac+inc+comp',src.replace(' ',''));self.assertNotIn('sign_changes',src)
        rng=np.random.default_rng(419);actions=rng.normal(size=(480,2));res=np.zeros(480);eps=rng.normal(scale=.25,size=480)
        for i in range(1,len(res)):res[i]=.78*res[i-1]+.45*actions[i,0]+eps[i]
        score,ac,inc,comp=learner.trigger_components(res,actions,123.0)
        self.assertAlmostEqual(score,ac+inc+comp,places=12);self.assertGreater(ac,0.0);self.assertGreater(inc,0.0);self.assertGreater(comp,0.0)

    def test_final_merge_bootstraps_functional_rediscovery_at_seed_level(self):
        src=(ROOT/'merge_v419.py').read_text();self.assertIn('def functional_seed_boot_ci',src);self.assertIn('ix=r.integers(0,n,n)',src)
        self.assertIn("'functional_similarity_ci':functional_seed_boot_ci(sig)",src);self.assertNotIn("boot_ci(sims,'median')",src)

    def test_final_merge_enforces_a3_g0_and_frozen_integrity(self):
        src=(ROOT/'merge_v419.py').read_text()
        self.assertIn("vals['G0']",src);self.assertIn('oracle_gain_mean',src);self.assertIn('oracle_gain_ci',src)
        self.assertIn("FAIL (oracle ceiling / benchmark invalid)",src)
        for token in ('81000','82000','83000','91000','91100'):self.assertIn(token,src)
        for key in ('preregistration_git_blob_sha','code_commit_shas','seed_shards','candidate_evaluations_total','rng_state_refs'):self.assertIn(key,src)

if __name__=='__main__':unittest.main()
