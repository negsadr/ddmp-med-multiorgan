This document includes peronal experience while working on this project

### **Setup Challenges and Solutions**
During the setup phase, I encountered an issue with the Pytorch installation. When attempting to install specific packages, I received an error stating that the required version of Pytorch (torch==1.13.1+cu116) was not available. To resolve this, I installed an older version of Pytorch by following the instructions on the [official Pytorch website](https://pytorch.org/get-started/previous-versions/).

### **Dataset Selection Process**
Selecting the appropriate dataset was initially challenging due to my unfamiliarity with 3D image formats, such as point clouds and NifTi files, which are critical for feeding data into models. After researching and learning more about these formats, I chose the LiTS dataset for a Liver Tumor Segmentation task. The LiTS dataset was particularly suitable because it included three labels (liver, tumor, background) and required minimal modification. For additional information, the dataset can be explored further [here](https://colab.research.google.com/drive/1bn6lWjJXKHxgu985ReHYTR2A_SgdtCS8?authuser=1#scrollTo=Vrp4-x7hIGX6). Consequently, I decided to use the BTCV dataset, which aligned well with my project requirements.


### **Data Modification Issue**
I made an error in data preprocessing by passing a scaler argument to the `read_image` function despite it being defined as `False`. This oversight was corrected in subsequent runs.



### **Memory Management Challenges**
I frequently encountered memory issues, typical of high-dimensional image data represented as voxels. In an ideal scenario, reducing the batch size could alleviate this problem, but my batch size was already set to one. To manage this, I utilized RunPod to access greater computing capacity, enabling more efficient data handling. Despite this, I was unable to run inference on the brats dataset with a robust setup (RTX 3090, 32vCPU, 125GB RAM) and had to upgrade my resources further. I also used Colab Pro L4, which proved effective for my needs.
