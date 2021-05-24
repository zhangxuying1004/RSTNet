# RSTNet:Relationship-Sensitive Transformer Network
This repository contains the reference code for the paper _RSTNet:Captioning with Adaptive Attention on Visual and Non-Visual Words_ (CVPR 2021).

<p align="center">
  <img src="images/RSTNet.png" alt="Meshed-Memory Transformer" width="850"/>
</p>

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
To run the code, annotations and detection features for the COCO dataset are needed.   
Please download the annotations file [annotations.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing) and rename the extracted folder as m2_annotations.  

Visual features are computed with the code provided by [1]. To reproduce our result, please download the COCO features file [x101-coco-features.hdf5](https://dl.fbaipublicfiles.com/grid-feats-vqa/X-101/X-101-features.tgz) and extract it. Note that this visual features are huge, you can use our `switch_datatype.py` to save the features as float16 for storage space saving. Besides, in order to solve the shape difference and match the feat shape with region feat shape (50 regions), please run `feats_process.py` to reshape the visual to `49(7x7)` and save all visual features as a h5py file.    

## Evaluation
To reproduce the results reported in our paper,   
download the pretrained model file [rstnet.pth]() and place it in the code folder.

Run `python test.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |

#### Expected output
Under `output_logs/`, you may also find the expected output of the evaluation code.


## Training procedure
Run `python train_language.py` and `python train_transformer.py` in sequence using the following arguments:

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--m` | Number of memory vectors (default: 40) |
| `--head` | Number of heads (default: 8) |
| `--warmup` | Warmup value for learning rate scheduling (default: 10000) |
| `--resume_last` | If used, the training will be resumed from the last checkpoint. |
| `--resume_best` | If used, the training will be resumed from the best checkpoint. |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |
| `--logs_folder` | Path folder for tensorboard logs (default: "tensorboard_logs")|

For example, to train our BERT-based language model with the parameters used in our experiments, use
```
python train_language.py --exp_name bert_language --batch_size 50 --features_path /path/to/features --annotation_folder /path/to/annotations
```
to train our rstnet model with the parameters used in our experiments, use
```
python train_transformer.py --exp_name rstnet --batch_size 50 --m 40 --head 8 --features_path /path/to/features --annotation_folder /path/to/annotations
```

<p align="center">
  <img src="images/visualness.png" alt="Sample Results" width="850"/>
</p>


#### References
[1] Jiang, H., Misra, I., Rohrbach, M., Learned-Miller, E., & Chen, X. (2020). In defense of grid features for visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10267-10276).
[2] Cornia, M., Stefanini, M., Baraldi, L., & Cucchiara, R. (2020). Meshed-memory transformer for image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10578-10587).

#### Acknowledgements
Thanks Cornia _et.al_ for their open source code [M2 transformer](https://github.com/aimagelab/meshed-memory-transformer), on which our implements are based.
Thanks Jiang _et.al_ for the significant discovery in visual representation, which has given us a lot of inspiration.
