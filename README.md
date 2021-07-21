# U2-NET CNN for Diamond Image Segmentation

28/06/2021
*Chenxin Jiang*

**RECSEM REU**

@Joint Institute for Computational Sciences, Oak Ridge National Laboratory & University of Tennessee

## Background

Diamond is a lattice crystal structure in which imperfections often exist.

* impurities / minor deformation of the lattice composition

* Many categories of defects

Defects in gem diamonds affect its appearance and severely degrades its commercial value!

Our Goal is to -- 

**Identifying defects on a diamond visually.**
    *-- what / where*


## Data Preparation

### Raw Data

A full 360-degree image display of a diamond can be captured by *Sarine Loupe scanning technology*.

There are 2067 pieces of diamond records, each with one PNG image of 1200\*1200 pixels and one movie of 2400\*1200 pixels with 400 frames. For each 20 pieces of diamond image, there is an XML file labeled the pixel-level defects from 10 categories (Crystal, Cloud, Twinning wisp, Pinpoint, Feather, Internal graining, Needle, Nick, Pit, Burn mark).

1. The labels distinguish the reflections from the true defects.

2. Currently, all the implementation is based on the image input. We have not utilized movie inputs so far.


### Data pre-processing

**Run `Data_Prepare/dataPrep.py`.**

1.  Generate Masked images (according to the `XML` file)

```python
generate_mask_image.py / genMaskImage(targetFileName,saveDir)
```

generate `-mask` images from the information in `targetFileName` and output it to`saveDir`(The directories need to change to your own file saved path)

* `targetFileName`: the target `XML` file storing the dimond defects information (imageID, width, height, defect labels, points location)
* `saveDir`: the path to save the `-mask` images

2. Delete the crust (black borders) in each image and resize the image to $1200 \times 1200$

```python
clip_and_resize.py / handle_image (filename,label_filename,src_folder,save_folder,tar_size=(1200,1200))
```

3. Clip image to small patches 400\*400​ for training

```python
generate_clipped_image.py / handle_image(filename,label_filename,src_folder,save_folder,tar_size=(400,400),stride=200)
```

Each image is clipped into 25 small patches of size 400\*400 *uniformly*.


**Resulting folders:**

`UNET1/diamond_labels`

* `id.png`: raw image
* `id-mask.png`: masked image

`UNET1/diamond_labels_cutted`: 

* `id.png`: cutted image (after deleting the crust and resizing)
* `id-mask.png`: masked image  (after deleting the crust and resizing)

`UNET1/diamond_labels_clipped`: 

* `id_i_j.png`(i, j=0,1,2,3,4): clipped image. There are 25 clipped images for one id - `id_0_0.png` to  `id_4_4.png`
* `id_i_j-mask.png` :  masked clipped image (same as above)



### Data Split

**Run `Data_Prepare/gen_txt.py`.**

`case = 1`: generate `id.txt` for `UNET1/diamond_labels` folder randomly

`case = 2`: for `UNET1/diamonds_labels_cutted` and `UNET1/diamonds_labels_clipped` folder, generate `id_all.txt`, ` id_val.txt`, `id_test.txt`

* train: test: validation = 6: 3: 1

  

## Model

```python
Apex_codes/utils/utils.py

def buildModel(opt):
    if opt.model_use == 'origin_UNet':
        from .UNet_Architecture import ResNetUNet #U2-Net
        model = ResNetUNet(n_class=opt.n_class) 
    elif opt.model_use == 'UNet':
        from .unet_V2 import UNet #U-Net
        model = UNet(n_class=opt.n_class) 
        model.apply(model_ini)

    return model
```

### U-Net

`.unet_V2.py / UNet`

Original paper: 

*U-Net: Convolutional Networks for Biomedical Image Segmentation* arXiv:1505.04597 [cs.CV]

* Input: 1\*400\*400

* Downsampling (contracting path)
	* Set padding=1, filter=3,
	* Max_pooling: f=2

* Upsampling (expansive path)
	* bilinear interpolation
	* (transpose convolution)
	* concatenate with high-resolution features from the contracting path



### U2-Net

`.UNet_Architecture.py /  ResNetUNet`

Two-level nested U-Net embodied in ReSidual U-blocks


Original paper: 

*U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection* arXiv:2005.09007 [cs.CV]




## Training
**Run `main.py` or `run_main.sh`**

* An Important technique:

```python
from torch.nn.parallel import DistributedDataParallel as DDP
```

## Testing

**Run `/dataPrep.py, case='test'`**

**Run `test.sh`**







