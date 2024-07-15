<!-- #region -->
# 3D Abdominal CT Image Synthesis Using Med-DDPM


## Summary

This repository extends the original med-ddpm repository by including implementations for training the Med-DDPM model to synthesize 3D abdominal CT images.




## Med-DDPM

[ArXiv](https://arxiv.org/pdf/2305.18453.pdf) | [IEEE](https://ieeexplore.ieee.org/document/10493074) | [GitHub](https://github.com/mobaidoctor/med-ddpm/)

Med-DDPM, a conditional diffusion model, establishes a benchmark for semantic 3D brain MRI synthesis. It is capable of converting masks to images in either single MRI modality or multiple modalities simultaneously, addressing data scarcity and privacy concerns. Moreover, it outperforms state-of-the-art methods in generating anatomically accurate images. This model presents significant potential for effective data augmentation and image anonymization.

<table>
  <tr>
    <td align="center">
      <strong>Input Mask</strong><br>
      <img id="img_0" src="images/img_0.gif" alt="Input Mask" width="100%">
    </td>
    <td align="center">
      <strong>Real Image</strong><br>
      <img id="img_1" src="images/img_1.gif" alt="Real Image" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Synthetic Sample 1</strong><br>
      <img id="img_2" src="images/img_2.gif" alt="Synthetic Sample 1" width="100%">
    </td>
    <td align="center">
      <strong>Synthetic Sample 2</strong><br>
      <img id="img_3" src="images/img_3.gif" alt="Synthetic Sample 2" width="100%">
    </td>
  </tr>
</table>
<!-- #endregion -->

## Dataset
The dataset comes from https://www.synapse.org/#!Synapse:syn3193805/wiki/217752.  

Under Institutional Review Board (IRB) supervision, 50 abdomen CT scans of were randomly selected from a combination of an ongoing colorectal cancer chemotherapy trial, and a retrospective ventral hernia study. The 50 scans were captured during portal venous contrast phase with variable volume sizes (512 x 512 x 85 - 512 x 512 x 198) and field of views (approx. 280 x 280 x 280 mm3 - 500 x 500 x 650 mm3). The in-plane resolution varies from 0.54 x 0.54 mm2 to 0.98 x 0.98 mm2, while the slice thickness ranges from 2.5 mm to 5.0 mm.

**Target**: 13 abdominal organs including
  1. Spleen
  2. Right Kidney
  3. Left Kidney
  4. Gallbladder
  5. Esophagus
  6. Liver
  7. Stomach
  8. Aorta
  9. IVC
  10. Portal and Splenic Veins
  11. Pancreas
  12. Right adrenal gland
  13. Left adrenal gland.

**Modality**: CT

**Size**: 30 3D volumes (24 Training + 6 Testing). First 6 samples used in this project

**Challenge**: BTCV MICCAI Challenge

The following figure shows image patches with the organ sub-regions that are annotated in the CT (top left) and the final labels for the whole dataset (right).


![image](https://lh3.googleusercontent.com/pw/AM-JKLX0svvlMdcrchGAgiWWNkg40lgXYjSHsAAuRc5Frakmz2pWzSzf87JQCRgYpqFR0qAjJWPzMQLc_mmvzNjfF9QWl_1OHZ8j4c9qrbR6zQaDJWaCLArRFh0uPvk97qAa11HtYbD6HpJ-wwTCUsaPcYvM=w1724-h522-no?authuser=0)


The image patches show anatomies of a subject, including:
1. large organs: spleen, liver, stomach.
2. Smaller organs: gallbladder, esophagus, kidneys, pancreas.
3. Vascular tissues: aorta, IVC, P&S Veins.
4. Glands: left and right adrenal gland
   


## Usage 

The [Colab Notebook](https://colab.research.google.com/drive/1d7Zh4_bWFyhpKHdsa_zHk1bHM9bTYkh2?usp=sharing) includes the steps to train and test the med-ddpm model on 6 samples of the BTCV dataset.




### Alternatively...
 
Ensure you have the following libraries installed for training and generating images:

- **Torchio**: [Torchio GitHub](https://github.com/fepegar/torchio)
- **Nibabel**: [Nibabel GitHub](https://github.com/nipy/nibabel)

```
pip install -r requirements.txt
```

Get the optimized med-ddpm model weights for both whole-head MRI synthesis and 4-modalities MRI synthesis from the link below:
[Download Model Weights](https://drive.google.com/drive/folders/1t6jk5MrKt34JYClgfijlbNYePIcTEQvJ?usp=sharing)
After downloading, place the files under the "model" directory.


Set the learned weight file path with \`--weightfile\`.

Determine the input mask file using \`--inputfolder\`.


Use the following commands for training and sample generation.

```
train the model on BTCV:$ ./scripts/train_btcv.sh
3D CT abdominal synthesis:$ ./scripts/sample_btcv.sh
```


## Refrences
[1]: Z. Dorjsembe, H. -K. Pao, S. Odonchimed and F. Xiao, "Conditional Diffusion Models for Semantic 3D Brain MRI Synthesis," in IEEE Journal of Biomedical and Health Informatics, vol. 28, no. 7, pp. 4084-4093, July 2024, doi: 10.1109/JBHI.2024.3385504.

[2]: Tang, Y., Yang, D., Li, W., Roth, H.R., Landman, B., Xu, D., Nath, V. and Hatamizadeh, A., 2022. Self-supervised pre-training of swin transformers for 3d medical image analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 20730-20740).

[3]: Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H. and Xu, D., 2022. Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. arXiv preprint arXiv:2201.01266

