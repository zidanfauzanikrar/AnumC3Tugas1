import unittest
from pathlib import Path
from role3_adapter import load_role3, verify_role3, role3_normal
from data_role4 import load_prices
import numpy as np
from solvers_role4 import householder_qr, normal_equations, first_reflection, condition_2
from data_role4 import build_datasets, design


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(71)
        self.A = self.rng.normal(size=(35,6))
        self.x = self.rng.normal(size=6)

    def test_exact_solution(self):
        b = self.A@self.x
        for solver in [lambda a,b:householder_qr(a,b)[0],normal_equations]:
            np.testing.assert_allclose(solver(self.A,b),self.x,rtol=1e-11,atol=1e-12)

    def test_inconsistent_ls_and_Q(self):
        b = self.rng.normal(size=35)
        x,R,c,Q = householder_qr(self.A,b,True)
        np.testing.assert_allclose(Q@R,self.A,atol=1e-12)
        np.testing.assert_allclose(Q.T@Q,np.eye(35),atol=1e-12)
        np.testing.assert_allclose(Q.T@b,c,atol=1e-12)
        np.testing.assert_allclose(self.A.T@(self.A@x-b),0,atol=1e-12)
        np.testing.assert_allclose(x,normal_equations(self.A,b),atol=1e-12)

    def test_first_reflection(self):
        v,H,HA=first_reflection(self.A)
        np.testing.assert_allclose(HA[1:,0],0,atol=1e-12)
        np.testing.assert_allclose(H.T@H,np.eye(35),atol=1e-12)

    def test_rank_deficiency(self):
        A=self.A.copy(); A[:,5]=A[:,0]
        for solver in [householder_qr,normal_equations]:
            with self.assertRaises(ValueError): solver(A,np.ones(35))

    def test_condition(self):
        self.assertAlmostEqual(condition_2(np.diag([1.,2.,8.])),8.)
        A=np.array([[1.,1.],[0.,1.]])
        self.assertAlmostEqual(condition_2(A),(3+np.sqrt(5))/2,places=10)
        self.assertAlmostEqual(condition_2(A.T@A),condition_2(A)**2,places=10)

    def test_regimes(self):
        A,b=design(np.array([.1,-.2,0.,.3]))
        np.testing.assert_allclose(A[0],[0,0,0,1,-.2,.1])
        np.testing.assert_allclose(A[1],[1,0,-.2,0,0,0])

    def test_boundary_no_future_leak(self):
        p=np.array([100.,101.,102.,103.,104.,105.,106.,107.])
        A,b,At,bt,_,idx=build_datasets(p[:5],p[5:])
        self.assertEqual(len(bt),3)
        self.assertAlmostEqual(bt[0],105/104-1)
        self.assertAlmostEqual(At[0,1],104/103-1)
        changed=p[5:].copy(); changed[0]*=1.2
        _,_,At2,_,_,_=build_datasets(p[:5],changed)
        np.testing.assert_allclose(At2[0],At[0])
        self.assertEqual(idx[0],5)

    def test_actual_role3_integration(self):
        folder=Path(__file__).resolve().parents[1]
        api=load_role3(folder/'Peran3_Soal2.ipynb')
        prices,_=load_prices(folder/'data/stock_train.csv')
        test,_=load_prices(folder/'data/stock_test.csv')
        A,b,_,_,_,_=build_datasets(prices,test)
        check=verify_role3(api,prices,A,b)
        self.assertEqual(check['status'],'IDENTICAL_A_AND_B')
        np.testing.assert_allclose(householder_qr(A,b)[0],role3_normal(api,A,b),atol=1e-12,rtol=1e-12)

    def test_independent(self):
        p=np.arange(100.,112.)
        _,_,At,bt,_,_=build_datasets(p[:6],p[6:],False)
        self.assertEqual(At.shape,(3,6))
        self.assertAlmostEqual(bt[0],109/108-1)


if __name__=='__main__':
    unittest.main()
