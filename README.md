# Learning Medical Image Denoising with Deep Dynamic Residual Attention Network

This is the official implementation of state-of-the-art medical image denosing method DRAN. Please consider to cite this paper as follows:

Full paper can be downloaded from this link:

Network Architecture

<img src = "https://user-images.githubusercontent.com/15001857/101247318-24858a00-3743-11eb-97eb-1fd5c2f93ce0.png" alt="network">

Denoising with DRAN
<img src = "https://user-images.githubusercontent.com/15001857/101247318-24858a00-3743-11eb-97eb-1fd5c2f93ce0.png" alt="Results">

Requirements
Python 3.8
Pytoch 1.5 
Torchvision 0.6
Cuda 10.1
Opencv
scikit-learn

Installation
git clone https://github.com/sharif-apu/MID-DRAN.git
cd MID-DRAN
pip install -r requirement.txt

Testing
DRAN can be inferenced with pretrained weights and default setting as follows:

python main.py -i
A few testing images are provided in a sub-directory under testingImages (i.e., testingImages/sampleImages/)
Denoised image(s) will be available in modelOutput/sampleImages/ 

To inference with custom setting execute the following command:

To specify input and output path of images please execute the following command:
python main.py -i -s path/to/inputImages -d path/to/outputImages -ns=sigma(s)
Here,-ns represents noise standard deviation (i.e., -ns=15,25,50), -s presents root of source images (i.e., testingImages/), -d presents destination root (modelOutput/)

Training
To train with your own dataset execute:
python main.py -ts -e X -b Y
To specify your trining images path, go to mainModule/config.json and update "trainingImagePath" entity. You can specify the number of epoch with -e flag (i.e., -e 5) and number of images per batch with -b flag (i.e., b 24).

For transfer learning execute:
python main.py -tr -e

Others
Check model configuration:
python main.py -ms
Update configuration file:
python main.py -c
Overfitting testing
python main.py -to

Contact
For any further query, feel free to contact us through the following emails: sma.sharif.cse@ulab.edu.bd, rizwanali@sejong.ac.kr, or mithun.bishwash.cse@ulab.edu.bd
