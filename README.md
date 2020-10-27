# Grayscale-Data-Augmentation
Code for paper : An Effective Data Augmentation for Person Re-identification. By providing this code, you can simply verify the validity of the method proposed in the paper.

The baselines we use are the ReID_baseline[10]（see the paper） that the author of CVPR2019 paper [8] opened in the early stage, the strong baseline [1] and FastReID which were recently open sourced by one of the main authors in [1] . We have conducted experiments on three of the largest and most representative datasets, Market-1501[13], DukeMTMC[14], and MSMT17[16], and our method has significantly improved on all three of these baselines. Since the model requires more training epochs to fit than the original, we add 0.5-1.5 times more training epochs to the training process. 
weight：
strong_baseline+RGPR：https://drive.google.com/file/d/1aehExt5oZMzUD5Tj8Dau_vhTKy8xj1EB/view?usp=sharing
