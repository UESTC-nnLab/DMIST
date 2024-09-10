# DMIST-Benchmark 
## ***Dense Moving Infrared Small Target Detection***

The DMIST benchmark datasets and baseline model implementation of the **TGRS 2024** paper [**Towards Dense Moving Infrared Small Target Detection: New Datasets and Baseline**](https://ieeexplore.ieee.org/document/10636251)

<img src="/readme/vis.png" width="1000px">


## Benchmark Datasets (bounding box-based)
- We synthesize two dense moving infrared small target datasets **DMIST-60** and **DMIST-100** on DAUB.
- Datasets are available at [DMIST](https://pan.baidu.com/s/1LL4rAFfv0Z8HRV4-w8mJjw?pwd=vkcu)(code: vkcu) and [IRDST](https://pan.baidu.com/s/10So3fntJMQxBy-bdSUUD6Q?pwd=t2ti)(code: t2ti). Or you can download IRDST directly from the website: [IRDST](https://xzbai.buaa.edu.cn/datasets.html). 

- You need to reorganize these datasets in a format similar to the `DMIST_train.txt` and `DMIST_val.txt` files we provided (`txt files` are used in training).  We provide the `txt files` for DMIST and IRDST.
For example:
```python
train_annotation_path = '/home/LASNet/DMIST_train.txt'
val_annotation_path = '/home/LASNet/DMIST_60_val.txt'
```
- Or you can generate a new `txt file` based on the path of your datasets. `Text files` (e.g., `DMIST_60_val.txt`) can be generated from `json files` (e.g., `60_coco_val.json`). We also provide all `json files` for [DMIST](https://pan.baidu.com/s/1LL4rAFfv0Z8HRV4-w8mJjw?pwd=vkcu) and [IRDST](https://pan.baidu.com/s/10So3fntJMQxBy-bdSUUD6Q?pwd=t2ti).

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
│   ├─train
│   │   ├─data5
│   │   │   ├─0.bmp
│   │   │   ├─0.txt
│   │   │   ├─ ...
│   │   │   ├─2999.bmp
│   │   │   ├─2999.txt
│   │   │   ├─ ...
│   │   ├─ ...
│   ├─test60
│   │   ├─data6
│   │   │   ├─0.bmp
│   │   │   ├─0.txt
│   │   │   ├─ ...
│   │   │   ├─398.bmp
│   │   │   ├─398.txt
│   │   │   ├─ ...
│   │   ├─ ...
│   ├─test100
│   │   ├─ ...
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
CUDA_VISIBLE_DEVICES=0 python train_DMIST.py
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
python test_DMIST.py
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
- We optimize old codes and retrain LASNet, achieving slightly better performance results than those reported in our paper.

<table>
  <tr>
    <th>Method</th>
    <th>Dataset</th>
    <th>mAP50 (%)</th>
    <th>Precision (%)</th>
    <th>Recall (%)</th>
    <th>F1 (%)</th>
    <th>Download</th>
  </tr>
  <tr>
    <td align="center">LASNet</td>
    <td align="center">DMIST-60</td>
    <td align="center">76.47</td>
    <td align="center">95.84</td>
    <td align="center">80.07</td>
    <td align="center">87.25</td>
    <td rowspan="3" align="center">
      <a href="https://pan.baidu.com/s/1nOdz29SnwkxUr6liYEYElA?pwd=av6r">Baidu</a> (code: av6r)
      <br><br>
      <a href="https://drive.google.com/drive/folders/13CvH9muxs-9fcgeSZJWraw1StWxE3zek?usp=sharing">Google</a>
    </td>
  </tr>
  <tr>
    <td align="center">LASNet</td>
    <td align="center">DMIST-100</td>
    <td align="center">65.70</td>
    <td align="center">96.52</td>
    <td align="center">68.68</td>
    <td align="center">80.25</td>
  </tr>
  <tr>
    <td align="center">LASNet</td>
    <td align="center">IRDST</td>
    <td align="center">74.50</td>
    <td align="center">89.10</td>
    <td align="center">84.06</td>
    <td align="center">86.51</td>
  </tr>
</table>



- PR curve on DMIST and IRDST datasets in the paper.
- We provide the results on [DMIST-60](./results/DMIST-60),  [DMIST-100](./results/DMIST-100) and [IRDST](./results/IRDST), and you can plot them using Python.

<img src="/readme/PR1.png" width="500px">
<img src="/readme/PR2.png" width="500px">
<img src="/readme/PR3.png" width="500px">

## Contact
If any questions, kindly contact with Shengjia Chen via e-mail: csj_uestc@126.com.

## References
1. S. Chen, L. Ji, J. Zhu, M. Ye and X. Yao, "SSTNet: Sliced Spatio-Temporal Network With Cross-Slice ConvLSTM for Moving Infrared Dim-Small Target Detection," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-12, 2024, Art no. 5000912, doi: 10.1109/TGRS.2024.3350024. 
2. B. Hui et al., “A dataset for infrared image dim-small aircraft target detection and tracking under ground/air background,” Sci. Data Bank, CSTR 31253.11.sciencedb.902, Oct. 2019.

## Citation

If you find this repo useful, please cite our paper. 

```
@ARTICLE{chen2024dmist,
  author={Chen, Shengjia and Ji, Luping and Zhu, Sicheng and Ye, Mao and Ren, Haohao and Sang, Yongsheng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Toward Dense Moving Infrared Small Target Detection: New Datasets and Baseline}, 
  year={2024},
  volume={62},
  number={},
  pages={1-13},
  doi={10.1109/TGRS.2024.3443280}}

```