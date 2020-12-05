# Learning Medical Image Denoising with Deep Dynamic Residual Attention Network

This is the official implementation of state-of-the-art medical image denoising method titled **"DRAN"**. Please consider to cite this paper as follows:

**Full paper can be downloaded from this link:**

# Network Architecture

 <img align="center" src = "https://user-images.githubusercontent.com/15001857/101247318-24858a00-3743-11eb-97eb-1fd5c2f93ce0.png" alt="network">
**Figure:** The overview of proposed network architecture. The proposed network incorporates novel dynamic residual attention blocks, which utilizes dynamic convolution and a noise gate. Also, the network leverage the residual learning along with the learning feature correlation.


# Medical Image Denoising with DRAN </br>
**Qualitative Comparison** </br>

<img align="center" src = "https://user-images.githubusercontent.com/15001857/101258714-93b4ab80-374e-11eb-984d-9f64fd14bf63.png" alt="Results"> </br>
**Figure:** </em> Performance of existing medical image denoising methods in removing image noise at sigma=50.The existing denoising methods immensely failed in addressing a substantial amount of noise removal and susceptible to produce artefacts. (a) Noisy input. (b) Result obtained by BM3D. (c) Result
obtained by DnCNN. (d) Result obtained by Residual MID. (e) Result obtained by DRAN.

    
**Quantitative Comparison** </br>

<img width=800 align="center"  src = "https://user-images.githubusercontent.com/15001857/101259263-007d7500-3752-11eb-8cbc-8a8fa4a56061.png" alt="Results">  </br> 
**Table:** Quantitative comparison between different medical image denoising methods. Results are
obtained by calculating the mean on two evaluation metrics. In all comparing categories, the proposed
method illustrates the consistency and outperforms the existing denoising methods. </br>



# Requirements
```Python 3.8 
Pytoch 1.5 
Torchvision 0.6 
Cuda 10.1  
Opencv 
Scikit
Tensorboard
Etaprogress
Torchsummary
Ptflops
Matplotlib
```

# Installation
```
git clone https://github.com/sharif-apu/MID-DRAN.git
cd MID-DRAN
pip install -r requirement.txt
```
</testpath/testDir>
# Testing
DRAN can be inferenced with pretrained weights and default setting as follows: </br>
```python main.py -i``` </br>

A few testing images are provided in a sub-directory under testingImages (i.e., testingImages/sampleImages/)</br>
Denoised image(s) will be available in modelOutput/sampleImages/ </br>

**To inference with custom setting execute the following command:**</br>
```python main.py -i -s path/to/inputImages -d path/to/outputImages -ns=sigma(s)``` </br>
Here,**-ns** represents noise standard deviation (i.e., -ns=15, 25, 50),**-s** presents root of source images (i.e., testingImages/), **-d** presents destination root (modelOutput/)

# Training
**To train with your own dataset execute:**</br>
```python main.py -ts -e X -b Y```
To specify your trining images path, go to mainModule/config.json and update "trainingImagePath" entity. </br>You can specify the number of epoch with **-e** flag (i.e., -e 5) and number of images per batch with **-b** flag (i.e., b 24).</br>

**For transfer learning execute:**</br>
```python main.py -tr -e -b ```

# Others
**Check model configuration:**</br>
```python main.py -ms``` </br>
**Update configuration file:**</br>
```python main.py ```-c</br>
**Overfitting testing** </br>
```python main.py -to ```</br>

# Contact
For any further query, feel free to contact us through the following emails: sma.sharif.cse@ulab.edu.bd, rizwanali@sejong.ac.kr, or mithun.bishwash.cse@ulab.edu.bd
