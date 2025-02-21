import numpy as np
import torch
import unittest

from algo_helpers import eval_regfunc, erm, fixed_grid_npmle, robbins

# Purpose of this file, make sure that things don't break whenever we add more functionality 
# (or at least make sure it compiles). 

class EBTransformerTest(unittest.TestCase):
    # TODO: add all relevant tests, e.g.: 
    # eb_train with very few epochs. 
    # gen_batch (e.g. ensure reproducibility when calling the same seed 3 times). 

    def test_sample(self):
        self.assertEqual(0, 1 - 1)

def test_regfunc_helper(lambdas, mus, newXs):
    outputs = eval_regfunc(lambdas, mus, newXs)
    answer = torch.zeros(len(newXs)) if isinstance(lambdas, torch.Tensor) else np.zeros(len(newXs))
    if isinstance(lambdas, torch.Tensor):
        answer = answer.to(outputs.device)
    for x in newXs:
        # Get the ratio of density between x + 1 to x. 
        exp_neg = lambda lam: (torch.exp(-lam) if isinstance(lam, torch.Tensor) else np.exp(-lam))
        if x == 0:
            fdens_lst = [exp_neg(lam)  * mu for (lam, mu) in zip(lambdas, mus)]
        else:
            fdens_lst = [exp_neg(lam) * (lam ** x) * mu for (lam, mu) in zip(lambdas, mus)]

        fdens = torch.sum(torch.stack(fdens_lst)) if isinstance(lambdas, torch.Tensor) else np.sum(fdens_lst)

        fdens1_lst = [exp_neg(lam) * (lam ** (x + 1)) * mu for (lam, mu) in zip(lambdas, mus)]
        fdens1 = torch.sum(torch.stack(fdens1_lst)) if isinstance(lambdas, torch.Tensor) else np.sum(fdens1_lst)
        answer[x] = torch.tensor(fdens1 / fdens) if isinstance(lambdas, torch.Tensor) else fdens1 / fdens

    if isinstance(lambdas, torch.Tensor):
        torch.allclose(outputs, answer)
    else:
        np.testing.assert_almost_equal(outputs, answer)

class RegFuncTest(unittest.TestCase):
    # Here we just brute force and see if our "smart" way works. 

    def test_regfunc_np(self):
        lambdas = np.array([1.0, 2.0, 3.0])
        mus = np.array([0.4, 0.3, 0.3])
        newXs = np.arange(40)
        test_regfunc_helper(lambdas, mus, newXs)

    def test_regfunc_withzeros(self):
        lambdas = np.array([0.0, 1.0, 2.0])
        mus = np.array([0.4, 0.3, 0.3])
        newXs = np.arange(40)
        test_regfunc_helper(lambdas, mus, newXs)

    # Try torch. 
    def test_regfunc_torch(self):
        lambdas = torch.Tensor([1.0, 2.0, 3.0])
        mus = torch.Tensor([0.4, 0.3, 0.3])
        newXs = torch.arange(40)
        test_regfunc_helper(lambdas, mus, newXs)

    def test_regfunc_torch_withzeros(self):
        lambdas = torch.Tensor([0.0, 1.0, 2.0])
        mus = torch.Tensor([0.4, 0.3, 0.3])
        newXs = torch.arange(40)
        test_regfunc_helper(lambdas, mus, newXs)

    def test_regfunc_torch_cuda(self):
        if not torch.cuda.is_available():
            return
        lambdas = torch.Tensor([1.0, 2.0, 3.0]).cuda()
        mus = torch.Tensor([0.4, 0.3, 0.3]).cuda()
        newXs = torch.arange(40).cuda()
        test_regfunc_helper(lambdas, mus, newXs)


if __name__ == '__main__':
    unittest.main()
