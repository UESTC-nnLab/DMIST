# DMIST-Benchmark

The DMIST benchmark datasets and baseline model implementation of the paper **Towards Dense Moving Infrared Small Target Detection: New Datasets and Baseline**

<img src="/readme/vis.png" width="1000px">


## Benchmark Datasets
- Our proposed dense target datasets DMIST-60 and DMIST-100.
- Datasets are available at [DMIST](https://pan.baidu.com/s/1nKzesU9Glv67qdMosmqyMQ?pwd=bu9t)(code: bu9t) and [IRDST](https://pan.baidu.com/s/1igjIT30uqfCKjLbmsMfoFw?pwd=rrnr)(code: rrnr). Or you can download IRDST directly from the website: [IRDST](https://xzbai.buaa.edu.cn/datasets.html). 

- You need to reorganize these datasets in a format similar to the `DMIST_train.txt` and `DMIST_val.txt` files we provided (`txt files` are used in training).  We provide the `txt files` for DMIST and IRDST.
For example:
```python
train_annotation_path = '/home/LASNet/DMIST_train.txt'
val_annotation_path = '/home/LASNet/DMIST_60_val.txt'
```
- Or you can generate a new `txt file` based on the path of your datasets. `Text files` (e.g., `DMIST_60_val.txt`) can be generated from `json files` (e.g., `60_coco_val.json`). We also provide all `json files` for [DMIST](https://pan.baidu.com/s/1nKzesU9Glv67qdMosmqyMQ?pwd=bu9t) and [IRDST](https://pan.baidu.com/s/1igjIT30uqfCKjLbmsMfoFw?pwd=rrnr).

``` python 
python utils_coco/coco_to_txt.py
```

- The folder structure should look like this:
```
DMIST
├─coco_train.json
├─60_coco_val.json
├─100_coco_val.json
├─images
    ├─train
        ├─data5
            ├─0.bmp
            ├─0.txt
            ├─ ...
            ├─2999.bmp
            ├─2999.txt
            ├─ ...
    ├─test60
        ├─data6
	    	├─0.bmp
            ├─0.txt
            ├─ ...
            ├─398.bmp
            ├─398.txt
    ├─test10
        ├─ ...
```


## Prerequisite

* python==3.10.11
* pytorch==1.12.0
* torchvision==0.13.0
* numpy==1.24.3
* opencv-python==4.7.0.72
* pillow==9.5.0
* scipy==1.10.1
* Tested on Ubuntu 20.04, with CUDA 11.3, and 1x NVIDIA 3090.


## Usage of baseline LASNet

### Train
- Note: Please use different `dataloaders` for different datasets. For example, if you want to use IRDST dataset for training, please change the `dataloader` in `train.py` to: `from utils.dataloader_for_IRDST import seqDataset, dataset_collate`.
```python
CUDA_VISIBLE_DEVICES=0 python train.py
```

### Test
- Usually `model_best.pth` is not necessarily the best model. The best model may have a lower val_loss or a higher AP50 during verification.
```python
"model_path": '/home/LASNet/logs/model.pth'
```
- You need to change the path of the `json file` of test sets. For example:
```python
#Use DMIST-100 dataset for test.
cocoGt_path         = '/home/public/DMIST/100_coco_val.json'
dataset_img_path    = '/home/public/DMIST/'
```
```python
python test.py
```

### Visulization
- We support `video` and `single-frame image` prediction.
```python
# mode = "video" #Predict a sequence
mode = "predict"  #Predict a single-frame image 
```
```python
python predict.py
```

## Results

- PR curve on DMIST and IRDST datasets.
- We provide the results on [DMIST-60](./results/DMIST-60),  [DMIST-100](./results/DMIST-100) and [IRDST](./results/IRDST), and you can plot them using Python.

<img src="/readme/PR1.png" width="500px">
<img src="/readme/PR2.png" width="500px">
<img src="/readme/PR3.png" width="500px">

## Contact
If any questions, kindly contact with Shengjia Chen via e-mail: csj_uestc@126.com.

## References
1. X. Shi, Z. Chen, H. Wang, D.-Y. Yeung, W.-K. Wong, and W.-c. Woo, “Convolutional lstm network: A machine learning approach for precipitation nowcasting,” Advances in Neural Information Processing Systems, vol. 28, 2015.
2. Z. Ge, S. Liu, F. Wang, Z. Li, and J. Sun, “Yolox: Exceeding yolo series in 2021,” arXiv preprint arXiv:2107.08430, 2021.

