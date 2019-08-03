
Run the scripts including:
1. To generate Figure 1 in the paper about the different settings (FullGrad, Neg5, Neg5-Boost3, Neg15), you need to run the following scripts:
    1) scriptTorchMultiRun: This script will run twice for different epochs (1 3 5 7 10) for FullGrad, Neg5, Neg5-Boost3.
    2) scriptTorchMultiRun15: Before running this, please change the word2vec.py line 76 to self.data.get_neg_v_neg_sampling(pos_pairs, 15). This script will run for Neg15.
    Note: for each of the two script, you can use the nohup ./script & to run in the back.
    For each script, after you running, you can find the average difference in the output file: resMultiRunl2Boost  resMultiRunl2Neg    resMultiRunl2Neg15  resMultiRunl2NoNeg  

2. To generate Figure 2 in the paper about the timing for different settings (FullGrad, Neg5, Neg5-Boost3, Neg15), you can run the scripts:
    1)scriptTimingRealNeg and scriptTimingRealNeg15 for Neg5, Neg5-Boost3, Neg15. Note before running scriptTimingRealNeg15, you need to change 
    the word2vec.py line 76 to self.data.get_neg_v_neg_sampling(pos_pairs, 15).
    2)scriptTimingFullGrad for full grad.
    Note: We will print out the time during running if you do not use nohup. Or we will write the result (time) to l1BoostingTiminglog  l1Neg15Timinglog  l1Timinglog fullNegativelog.
    
3. To generate one line of Figure 3 in the paper about different boosting factor.
    1) scriptTorchDifferentBoosting: this will try different boosting factor and write the average difference to resRealNegEqualDistl1DiffBoost file. 
    
    
The based code (skip-gram) is Forked from a github project (not ours) https://github.com/Adoni/word2vec_pytorch.


Based on the base code, we implement:
1. 3 Norm: L1, L2, Log

2. Mission impossible (no negative sample, just use all possible negative sample)

3. Generate the derivation result #ij/#i for all pairs or #ij/#i*window (#ij means i j cooccur in a window) for curpus and compare.

4. Auto run and check result in resLoss+{Neg, Boost, NoNeg} represent for Neg5 Neg5Boost3 NoNegative.
        nohup ./scriptTorch &


