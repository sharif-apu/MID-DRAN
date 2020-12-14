# Learning Medical Image Denoising with Deep Dynamic Residual Attention Network

This is the official implementation of state-of-the-art medical image denoising method titled as **"Dynamic Residual Attention Network (DRAN)"**. **[[Click Here](https://www.mdpi.com/2227-7390/8/12/2192/pdf)]** to download the full paper (in PDF).  </br>

**Please consider to cite this paper as follows:**
```
@article{sharif2020learning,
  title={Learning Medical Image Denoising with Deep Dynamic Residual Attention Network},
  author={Sharif, SMA and Naqvi, Rizwan Ali and Biswas, Mithun},
  journal={Mathematics},
  volume={8},
  number={12},
  pages={2192},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
# Overview
<p align="center">
<img width=800 align="center" src = "https://user-images.githubusercontent.com/15001857/101650013-af2cf880-3a65-11eb-9f23-30e85d054011.png" alt="Overview"> </br>
</p>

**Figure:** Overview of proposed medical image denoising method. 

# Network Architecture
<p align="center">
 <img width=800 align="center" src = "https://user-images.githubusercontent.com/15001857/101642681-79841180-3a5d-11eb-9dcf-ae9db1e757f9.png" alt="network"> </br>
</p>

**Figure:** The overview of proposed network architecture. The proposed network incorporates novel dynamic residual attention blocks, which utilizes dynamic convolution and a noise gate. Also, the network leverage the residual learning along with the learning feature correlation.


# Medical Image Denoising with DRAN </br>
**Qualitative Comparison** </br>
<p align="center">
<img width=800 align="center" src = "https://user-images.githubusercontent.com/15001857/101643040-e697a700-3a5d-11eb-8099-e054ae9c7759.png" alt="Results"> </br>
</p>

**Figure:** </em> Performance of existing medical image denoising methods in removing image noise at **sigma = 50**. The existing denoising methods immensely failed in addressing a substantial amount of noise removal and susceptible to produce artefacts. (a) Noisy input. (b) Result obtained by BM3D. (c) Result
obtained by DnCNN. (d) Result obtained by Residual MID. (e) Result obtained by **DRAN**. (f) Reference Image.

    
**Quantitative Comparison** </br>
<p align="center">
<img width=800 align="center"  src = "https://user-images.githubusercontent.com/15001857/101272591-c9da4580-37b7-11eb-8db8-37d7c53ed36c.png" alt="Results"> 
</p>

**Table:** Quantitative comparison between different medical image denoising methods. Results are
obtained by calculating the mean on two evaluation metrics (PSNR and SSIM). In all comparing categories, the proposed
method illustrates the consistency and outperforms the existing denoising methods. </br>



# Prerequisites
```
Python 3.8
CUDA 10.1 + CuDNN
pip
Virtual environment (optional)
```

# Installation
**Please consider using a virtual environment to continue the installation process.**
```
git clone https://github.com/sharif-apu/MID-DRAN.git
cd MID-DRAN
pip install -r requirement.txt
```

# Testing
**DRAN can be inferenced with pretrained weights and default setting as follows:** </br>
```python main.py -i``` </br>

A few testing images are provided in a sub-directory under testingImages (i.e., testingImages/sampleImages/)</br>
In such occasion, denoised image(s) will be available in modelOutput/sampleImages/. </br>

**To inference with custom setting execute the following command:**</br>
```python main.py -i -s path/to/inputImages -d path/to/outputImages -ns=sigma(s)``` </br>
Here,**-ns** specifies the standard deviation of a Gaussian distribution (i.e., -ns=15, 25, 50),**-s** specifies the root directory of the source images
 (i.e., testingImages/), and **-d** specifies the destination root (i.e., modelOutput/).

# Training
**To train with your own dataset execute:**</br>
```python main.py -ts -e X -b Y```
To specify your trining images path, go to mainModule/config.json and update "trainingImagePath" entity. </br>You can specify the number of epoch with **-e** flag (i.e., -e 5) and number of images per batch with **-b** flag (i.e., -b 24).</br>

**For transfer learning execute:**</br>
```python main.py -tr -e -b ```

# Others
**Check model configuration:**</br>
```python main.py -ms``` </br>
**Create new configuration file:**</br>
```python main.py -c```</br>
**Update configuration file:**</br>
```python main.py -u```</br>
**Overfitting testing** </br>
```python main.py -to ```</br>

# Contact
For any further query, feel free to contact us through the following emails: sma.sharif.cse@ulab.edu.bd, rizwanali@sejong.ac.kr, or mithun.bishwash.cse@ulab.edu.bd
