# Description

This is a project for detecting tumors in colorectal region using `Kvasir-SEG` dataset. It is an open-access dataset of gastrointestinal polyp images and corresponding segmentation masks, manually annotated and verified by an experienced gastroenterologist.

The dataset consists of rgb images of regions of colon and a mask image to highlight the tumor areas.

![Dataset](/media/dataset.png)

The model that has been used is an advanced version of Unet which is a combination of `R2Unet` and `Attention UNet`. The model architecture looks as follows:

![model](/media/model.png)

The model was trained for 50 epochs and produced satisfactory output.

![result1](/media/result1.png)
![result2](/media/result2.png)

##### Note 
This is an ongoing project and various steps towards achieving good accuracy is being done.



# Citations
@inproceedings{jha2020kvasir, title={Kvasir-seg: A segmented polyp dataset}, author={Jha, Debesh and Smedsrud, Pia H and Riegler, Michael A and Halvorsen, P{\aa}l and de Lange, Thomas and Johansen, Dag and Johansen, H{\aa}vard D}, booktitle={International Conference on Multimedia Modeling}, pages={451--462}, year={2020}, organization={Springer} }