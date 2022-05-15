# RSTNet: Relationship-Sensitive Transformer Network
This repository contains the reference code for the paper [_RSTNet: Captioning with Adaptive Attention on Visual and Non-Visual Words_ (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_RSTNet_Captioning_With_Adaptive_Attention_on_Visual_and_Non-Visual_Words_CVPR_2021_paper.pdf).

<p align="center">
  <img src="images/RSTNet.png" alt="Relationship-Sensitive Transformer" width="800"/>
</p>

## Tips
Sometimes I may not be able to answer issues in time.  
if you are in a hurry, you can add my wechat: zhangxuying1004 with the remark 'RSTNet'.  

## Environment setup
Clone the repository and create the `m2release` conda environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate m2release
```

Then download spacy data by executing the following command:
```
python -m spacy download en
```

Note: Python 3.6 is required to run our code. 


## Data preparation
To run the code, annotations and visual features for the COCO dataset are needed.  

First, most annotations have been prepared by [1], please download [annotations.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing) and rename the extracted folder as ```m2_annotations```, please download [image_info_test2014.json](http://images.cocodataset.org/annotations/image_info_test2014.zip) and put it into ```m2_annotations```. 

Then, visual features are computed with the code provided by [2]. To reproduce our result, please download the COCO features file [X-101-features.tgz](https://dl.fbaipublicfiles.com/grid-feats-vqa/X-101/X-101-features.tgz) and rename the extracted folder as ```X101-features```. Note that this visual features are huge, you can alternatively save the features as float16 for storage space saving by executing the following command:
```
python switch_datatype.py
```
Finally, in order to solve the shape difference and match the feat shape with region feat shape (`50` regions), please execute the following command to reshape the visual to `49(7x7)` and save all visual features as a h5py file.
```
python feats_process.py
```

Note that, you can also use my processed offline image features [COCO-X-101-grid.hdf5](https://pan.xunlei.com/s/VMyFV3OcrpOj7TdWkt5_amwiA1) with extraction code ```wsvg``` and my processed online image features [X101_grid_feats_coco_test.hdf5](https://pan.xunlei.com/s/VN-YFlVCAGDe_glAUaPkNOg2A1) with extraction code ```qzwm``` for convenience.  

Besides, if you want to extract grid features of your custom image dataset, you can refer to the code [grid-feats-vqa
](https://github.com/facebookresearch/grid-feats-vqa).

## Training procedure
Run `python train_language.py` and `python train_transformer.py` in sequence using the following arguments:

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers, accelerate model training in the xe stage.|
| `--head` | Number of heads (default: 8) |
| `--resume_last` | If used, the training will be resumed from the last checkpoint. |
| `--resume_best` | If used, the training will be resumed from the best checkpoint. |
| `--features_path` | Path to visual features file (h5py)|
| `--annotation_folder` | Path to m2_annotations |

For example, to train our BERT-based language model with the parameters used in our experiments, use
```
python train_language.py --exp_name bert_language --batch_size 50 --features_path /path/to/features --annotation_folder /path/to/annotations
```
to train our rstnet model with the parameters used in our experiments, use
```
python train_transformer.py --exp_name rstnet --batch_size 50 --m 40 --head 8 --features_path /path/to/features --annotation_folder /path/to/annotations
```
The figure below shows the changes of cider value during the training of rstnet. You can also visualize the training details by calling the tensorboard files in ```tensorboard_logs```.
<p align="center">
  <img src="images/train_cider.png" alt="cider changes" width="800"/>
</p>

## Evaluation
Run `python test_transformer.py` to evaluate the rstnet or `python test_language.py` to evaluate the language model using the following arguments:

| Argument | Possible values |
|------|------|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--features_path` | Path to visual features file (h5py) |
| `--annotation_folder` | Path to m2_annotations |

Note that, you can also download our pretrained model files in the ```Pre-trained_Models``` folder to reproduce the our reported results.  

#### References
[1] Cornia, M., Stefanini, M., Baraldi, L., & Cucchiara, R. (2020). Meshed-memory transformer for image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.  
[2] Jiang, H., Misra, I., Rohrbach, M., Learned-Miller, E., & Chen, X. (2020). In defense of grid features for visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.   


#### Acknowledgements
Thank Cornia _et.al_ for their open source code [meshed-memory-transformer
](https://github.com/aimagelab/meshed-memory-transformer), on which our implements are based.  
Thank Jiang _et.al_ for the significant discovery in visual representation [2], which has given us a lot of inspiration.
