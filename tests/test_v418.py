import pathlib,sys,unittest,random
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from flf_ecf.v418 import G,rg,mut,Model,DATA,met,ctl
class TestV418(unittest.TestCase):
 def test_genome_mutation_stays_valid(self):
  r=random.Random(1);g=rg(r);m=mut(g,r);self.assertNotEqual(g,m);self.assertTrue(3<=m.ks<=8 and 3<=m.ka<=8)
 def test_continuous_model_is_finite(self):
  g=G(0,1,0,6,6,0,0,0,1,0);m=Model(g).fit(DATA[0][0]);p,_=met(m,DATA[0][1]);self.assertTrue(0<=p<5)
 def test_control_uses_post_intervention_state(self):
  g=G(0,1,0,6,6,0,0,0,1,0);m=Model(g).fit(DATA[6][0]);self.assertGreater(ctl(1006,m,40),0.5)
if __name__=='__main__':unittest.main()
