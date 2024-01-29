# macrosbi
This is a repository that allows one to estimate macro models with the [simulation based inference (SBI) library](https://www.mackelab.org/sbi/)

The code is straightforward to run.  The main function is `sbimacro.sbi_macro()`.  You run it like this:

```
from sbiwrapper.sbimacro import sbi_macro
import torch

def testfunct(params):
    xout = params[0]*torch.rand(100).reshape((1,-1))
    return xout

xin = testfunct(torch.tensor([10], dtype= torch.float32)).numpy()
prior1 = torch.distributions.Uniform(torch.tensor([0.0]),torch.tensor([20.0]))
boundmin1 = [.00001]
boundmax1 = [19.99999]
samps, post = sbi_macro(xin, boundmin1, boundmax1, prior1, testfunct, netparams = None, num_rounds = 10, 
              method = 'SNPE', folder = None, init_simulations = 30, round_simulations = 25,
              numworkers = None, batch_size = 1000)
```

Note the `init_simulations` and the `round_simulations` are abnormally low for speed reasons.  These numbers should be in the 1000s at least.  

This package has been lightly tested.  Feel free to report any errors here.  


#Additional Information:

This repository contains a .ipynb file that also shows how to run the SBI estimation routine on real data and a real model that has been solved.  

Finally, the sbiwork folder contains some of the estimation routines I used to get my results.  
