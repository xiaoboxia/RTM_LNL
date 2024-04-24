# RTM_LNL
Regularly Truncated M-estimators for Learning with Noisy Labels (PyTorch implementation).

This is the code for the paper:
[Regularly Truncated M-estimators for Learning with Noisy Labels]()      
Xiaobo Xia*, Pengqian Lu*, Chen Gong, Bo Han, Jun Yu, Jun Yu, and Tongliang Liu

## Dependencies
torch==1.10.0

tqdm==4.64.1

numpy==1.23.0

torchvision==0.11.0

## Experiments

To run the experiments for rtcatoni on mnist with 30% symmetric label noise, run 

`python main.py --loss rtcatoni --noise_type symmetric --noise_rate 30 --dataset mnist`

result will be saved in `./results/results.csv`

The loss options for our method is rtcatoni, rtwelschp, rtlogsum.

The noise type options is symmetric, pairflip, ILN.

The noise rate options is 30, 50 but 45 for pairflip.

**Rurrently, we have release the code on four datasets, including mnsit, cifar10, cifar100, svhn and news.**

If you find this code useful in your research, please consider citing:

```
@article{xia2023regularly,
  title={Regularly truncated m-estimators for learning with noisy labels},
  author={Xia, Xiaobo and Lu, Pengqian and Gong, Chen and Han, Bo and Yu, Jun and Liu, Tongliang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```
