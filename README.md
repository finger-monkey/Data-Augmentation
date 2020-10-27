# An Effective Data Augmentation for ...
Code for paper : An Effective Data Augmentation for xxx. By providing this code, you can simply verify the validity of the method proposed in the paper.

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
