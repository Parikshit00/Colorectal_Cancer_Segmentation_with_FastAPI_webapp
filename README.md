# Description

A FastAPI WebApp for colorectal cancer segmentation with torch version of `Metapolyp` and `AttentionR2Unet`. 

# FastAPI WebApp
The webapp is setup to infer from trained `metapolyp` model checkpoint `best_model.ckpt`. You need to retrain the model to get your own weights and then put it inside `/app/weights/best_model.ckpt`, to use other models, the inference can be adjusted manually at `/app/utils.py`. 
To run the webapp run:  
```
uvicorn main:app --reload
```  
![WebApp](/media/webapp.png)

# Inference and PostProcessing Steps
The input image would be inferred by loading the saved weights.  
The initial mask looks a bit pixelated in the edges. To address this, I have done `GaussianBlur` with `morphological operations` to make it more smoother. This also helps to find accurate contour and bounding boxes. Then the bbox is drawn over original input for detailed visualization:  

![Inference](/media/inferred.png)

# Installation

## Pip
In the root folder of this repo: 
```
pip install -r requirements.txt
```

## Webapp Docker
You can directly build the FastAPI webapp in docker. To do this, first setup your FastAPI webapp `weights`folder as specified above. Then run:  
```
docker build -t coloncare .
docker run -d --name cancercontainer -p 80:80 coloncare
```
Please note you need to manually execute the training script for each folder to train them. I dont provide Docker image for this task Refer to model informations below.

# Dataset 
The dataset of gastrointestinal polyp images and corresponding segmentation masks is from [PraNet](https://github.com/DengPingFan/PraNet), with training set from 50% of Kvasir-SEG and 50% of ClinicDB dataset.

The dataset download link can be found at [Google drive](https://drive.google.com/drive/folders/10SYLHNvO0fSrhhVhj5U-cFgOnTH5uGJf) which consists of `TrainDataset`and `TestDataset` for training and benchmarking respectively. Download the dataset from google drive and place it in `datasets/TrainDataset` and `datasets/TestDataset`.
![Dataset](/media/dataset.png)

# Models
## MetapolyP

Torch implementation of Metapolyp model which achieves state of art results on Kvasir Seg Dataset. 

 
 The code provides an implementation of the Meta-Polyp baseline model for polyp segmentation in `torch`. This model architecture is detailed in the [original paper](https://arxiv.org/pdf/2305.07848v3.pdf).

   The folder `train_metapoly` consistes of model and its layer inside `model.py`and `layers` folder respectively. The original official tensorflow implementation was converted to torch by me. The code loads `CAFormer`pretrained weights as backbone using `Keras Cv Attention Models`package. Check the `train_metapoly/requirements.txt`to setup correct version of `kecam` package in your `pip` or `conda` environment before running `Lightning` style trainer code.
   
   To train the model, just run:  
   ```
   python train.py
   ```
   This will run the training with hyperparameter and implementation of trainer same as [original implementation](https://github.com/huyquoctrinh/MetaPolyp-CBMS2023/tree/main).
   ![model_metapoly](/media/model1.png)
## AttentionR2Unet

The model that has been used is an advanced version of Unet which is a combination of `R2Unet` and `Attention UNet`. It uses a native torch type trainer. The model architecture looks as follows:
  ![model](/media/model.png)  
  To train this model you need execute:  
  ```
  python solver.py --images_folder <path> --masks_folder <path> --epochs <number> --batch_size <number> --out_path <path_to_save_model_weights> --test <True/False> --train <True/False> --resume_training <resume_from_prev_training>
  ```

# Citations
@inproceedings{jha2020kvasir, title={Kvasir-seg: A segmented polyp dataset}, author={Jha, Debesh and Smedsrud, Pia H and Riegler, Michael A and Halvorsen, P{\aa}l and de Lange, Thomas and Johansen, Dag and Johansen, H{\aa}vard D}, booktitle={International Conference on Multimedia Modeling}, pages={451--462}, year={2020}, organization={Springer} }  

@misc{trinh2023metapolyp,
      title={Meta-Polyp: a baseline for efficient Polyp segmentation}, 
      author={Quoc-Huy Trinh},
      year={2023},
      eprint={2305.07848},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
