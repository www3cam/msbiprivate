# macrosbi
This is repository that allows one to estimate macro models with the [simulation based inference (SBI) library](https://www.mackelab.org/sbi/)

The code is straightforward to run.  The main function is `sbimacro.sbi_macro()`.  You run it like this:

```
from sbiwrapper import sbi_macro

def testfunct(params):
    xout = params[0]*torch.rand(100).reshape((1,-1))
    return xout

xin = testfunct(torch.tensor([10], dtype= torch.float32)).numpy()
prior1 = torch.distributions.Uniform(torch.tensor([0.0]),torch.tensor([20.0]))
boundmin1 = [.00001]
boundmax1 = [19.99999]
samps, post = sbi_macro(xin, boundmin1, boundmax1, prior1, testfunct, netparams = None, num_rounds = 10, 
              method = 'SNPE', folder = 'save', init_simulations = 30, round_simulations = 25,
              numworkers = None, batch_size = 1000)
```

Note the `init_simulations` and the `round_simulations` are abnormally low for speed reasons.  These numbers should be in the 1000s at least.  
