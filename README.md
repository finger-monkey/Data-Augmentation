# An Effective Data Augmentation for person re-identification
Code for paper : An Effective Data Augmentation for person re-identification(https://arxiv.org/abs/2101.08533). The paper edition will be updated constantly, please check the latest edition. You can see the Chinese version of this paper in the article 《基于灰度域特征增强的行人重识别方法》-龚云鹏，which is more complete. Additional extensions may be added to the English version of the paper.

By the way, we have implemented a good adversarial defense method of Reid on this basis (see another paper for details:A Person Re-identification Data Augmentation Method with Adversarial Defense Effect(https://arxiv.org/abs/2101.08783, the code is available at:https://github.com/finger-monkey/ReID_Adversarial_Defense))

By providing this code, you can simply verify the validity of the method proposed in the paper.

The baselines we use are the ReID_baseline[10]（see the paper） that the author of CVPR2019 paper [8] opened in the early stage, the strong baseline [1] and FastReID which were recently open sourced by one of the main authors in [1] . We have conducted experiments on three of the largest and most representative datasets, Market-1501[13], DukeMTMC[14], and MSMT17[16], and our method has significantly improved on all three of these baselines. Since the model requires more training epochs to fit than the original, we add 0.5-1.5 times more training epochs to the training process. 

The github of baselines:(You can find the datasets you need from the following links)

ReID_baseline: https://github.com/layumi/Person_reID_baseline_pytorch

strong baseline: https://github.com/michuanhaohao/reid-strong-baseline

FastReID: https://github.com/JDAI-CV/fast-reid


The weight of code：

strong_baseline(resnet50)+RGPR(market)[rank1/mAP:95.1/87.2->reRank:95.9/94.4]：https://drive.google.com/file/d/1aehExt5oZMzUD5Tj8Dau_vhTKy8xj1EB/view?usp=sharing

strong_baseline(resnet50)+RGPR(duke)[rank1/mAP:87.3/77.3->reRank:91/89.4]：https://drive.google.com/file/d/1sAXP2kuUwHTipXAyHZy2tsSsaqye112_/view?usp=sharing

FastReID(sbs_R101-ibn)+RGPR(duke)[rank1/mAP_92.8/84.2->reRank:94.3/92.7]: https://drive.google.com/file/d/13CgLZzLVpPXKAJJV4BH8PA0_1WeyYVEu/view?usp=sharing

FastReID(sbs_R101-ibn)+RGT&RGPR(msmt17)[rank1/mAP_86.2/65.9->reRank-90.3/84.15]: https://drive.google.com/file/d/1vPqEj1THd6KeRK0Jjg2TXSJnIq50AuXG/view?usp=sharing

FastReID(sbs_R101-ibn)+RGT(market)[rank1/mAP_96.5/91.2->reRank-96.9/95.6]: https://drive.google.com/file/d/1Dt1VyLHObZClMpv9uoMrw3k37SGj6XdI/view?usp=sharing


###########Use ReID_baseline to simply verify the effectiveness of our method##################

download the code from the ReID_baseline, and then just do as follow:

(1)download our file 'trans_gray.py' and put it together with the code of ReID_baseline

(2)add the code between lines 23-24 of 'train.py' file:  'from trans_gray import *' 

(3)add the code between lines 76-77 of 'train.py' file:'transforms.RandomGrayscale(0.05)' or 'trans_gray.LGT(0.4)'

(4) change parameter 'num_epochs' to 120 on line 386, 

(5)and use command 'python train.py --train_all' to train according to the tutorial provided by author,finally use 'test.py' to test.

The author’s training accuracy is: r-1:88.84, mAP:71.59. 
You can also set 'transforms.RandomGrayscale(1)' in lines 76-77 and lines 83-84 verify that the contribution of grayscale information to the query task.
