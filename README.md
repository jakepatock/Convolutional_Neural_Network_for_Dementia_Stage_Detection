<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
## Stages of Dementia
<div style="display: flex; justify-content: flex-end; gap: 20px;">
  <img src="https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection/blob/master/Dataset/Moderate_Demented/moderate_1.jpg" alt="Moderate Demented" style="flex: 25%; margin-left: auto; padding: 10px;">

  <img src="https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection/blob/master/Dataset/Mild_Demented/mild_1.jpg" alt="Mild Demented" style="flex: 25%; margin-left: auto; padding: 10px;">

  <img src="https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection/blob/master/Dataset/Very_Mild_Demented/verymild_1.jpg" alt="Very Mild Demented" style="flex: 25%; margin-left: auto; padding: 10px;">

  <img src="https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection/blob/master/Dataset/Non_Demented/non_1.jpg" alt="Non Demented" style="flex: 25%; margin-left: auto; padding: 10px;">
</div>

&nbsp; &nbsp; &nbsp; Moderate &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Mild &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Very Mild &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Non 

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<h3 align="center">Convolutional Neural Network for Dementia Stage Detection</h3>

  <p align="left">
    This project involves a CNN model used for predicting the stage of dementia in patients based on axial transverse MRI brain images. The dataset used by this model was sourced from Kaggle (link: https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset). This dataset comprises 6,400 128x128-pixel images of transverse brain MRI scans from patients at varying dementia stages. The dataset consists of four classes: Class 1 (mildly demented) includes 896 images; Class 2 (moderately demented) has 64 images; Class 3 (non-demented) comprises 3,200 images; and Class 4 (very mildly demented) contains 2,240 images. Notably, the dataset is imbalanced across its class representation.

The dataset was divided into 80% for training and 20% for validation, and the images were converted into a single grayscale channel tensor. A batch size of 16 images per batch was utilized in the data loader. The CNN structure consisted of 5 convolutional layers with a max-pooling layer interspersed among them. The channel sizes progressed in the order of 1, 64, 128, 256, 512, and 512. The final pooling layer was flattened and connected to three fully connected linear layers. After the first two linear layers, dropout of 0.5 was introduced along with batch normalization to prevent overfitting. The sizes of these linear layers were as follows: 8192 neurons, 1024 neurons, 256 neurons, and finally 4 neurons.

The model utilized cross-entropy loss for multi-class classification and was optimized using the Adam optimizer. Evaluation was performed using a weighted F1 score, biased towards recall due to the medical nature of the data. The model prioritized avoiding false negatives, as they might lead to patients' symptom dismissal and delayed dementia diagnosis, potentially impacting treatment effectiveness. False positives were deemed less damaging, leading to closer patient inspection and continued testing which results in little harm to the pacient.

Training incorporated an early stopper (with a pacients of 50 epochs) based on maximizing the F1 score on the validation set. The final F1 score on the validation set was 0.996875, with an accuracy of 0.996875 and loss of 0.02079408180675042. The model achieved an accuracy and loss of 0.9986328125 and 0.003162659881454785 on the training data. The model reach its max F1 score at epcoh 124 and this is the model state saved in the model.pth file. The figure "CNN_Eval_Plot_norm_batch16.png" in the data directory showcases a comparison plot between the model's training and validation accuracy and loss, along with a plot of the F1 scores during training. The red dot on the plot shows the saved models final statistics. The plot is shown below. The two other figures in the plots directory show the results of training the model with batch sizes of 32 and 64 images istead of 16. The microbatching and batch normalization stabalized the model's learning and was ultimately used to created the saved model. ![Alt text](https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection/blob/master/model_results/plots/CNN_Eval_Plot_norm_batch16.png)
    <br />
    <a href="https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection">View Demo</a>
    ·
    <a href="https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection/issues">Report Bug</a>
  </p>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

* [![Static Badge](https://img.shields.io/badge/Python-3.10.10-yellow?logo=python)][python-url]
* [![Pytorch](https://img.shields.io/badge/Pytorch-2.1.2%2Bcu121-red?logo=pytorch)](https://pytorch.org/)
* [![Static Badge](https://img.shields.io/badge/Scitkit%20Learn-1.3.2-orange?logo=scikit-learn)](https://scikit-learn.org/stable/)
* [![Static Badge](https://img.shields.io/badge/Numpy-1.26.2-blue?logo=numpy&logoColor=blue)](https://numpy.org/)
* [![Static Badge](https://img.shields.io/badge/Matplotlib-3.8.2-black)](https://matplotlib.org/)
* [![Static Badge](https://img.shields.io/badge/Torchmetrics-1.2.1-purple?logo=torchmetrics&logoColor=blue)](https://lightning.ai/docs/torchmetrics/stable/)
* [![Static Badge](https://img.shields.io/badge/PIL-1.2.1-white?logo=pillow&logoColor=blue)](https://pypi.org/project/Pillow/)
* [![Static Badge](https://img.shields.io/badge/ast-green?logo=pillow&logoColor=blue)](https://docs.python.org/3/library/ast.html)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This project requires the installation of a python 3.8-3.10 and few libraries to get started. The dependencies of this project include the libraries of pytorch, scikit learn, numpy, matplotlib, torchmetrics, PIL ,and ast.

### Prerequisites

Pytorch supports python verions of 3.8-3.10. This project uses python 3.10.10 with pip used for package installation. The project used the gpu cuda version of pytorch. Alternatively, you can can use the slower cpu version of pytorch if hardware is not available (cuda is not avaliable on MacOS). Commands for all operating systems can be found on the link below.

[![Static Badge](https://img.shields.io/badge/Torch-Install-red?logo=pytorch)](https://pytorch.org/get-started/locally/)

Examples:
* Pytorch cuda for Windows
  ```sh
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
* Pytorch cuda for Linux
  ```sh
  pip3 install torch torchvision torchaudio
  ```
* Pytorch cpu version for Windows
  ```sh
  pip3 install torch torchvision torchaudio
  ```

The rest of the packages can be installed using pip with the following commands:
* Scikit-Learn
  ```sh
  pip install scikit-learn==1.3.2
  ```
* Numpy
  ```
  pip install numpy==1.26.2
  ```
* Matplotlib
  ```sh
  pip install matplotlib==3.2.2
  ```
* Torchmetrics
  ```sh
  pip install torchmetrics==1.2.1
  ```
* PIL
  ```sh
  pip install Pillow==1.2.1
  ```
* AST
  ```
  pip install ast
  ```

### Installation

1. Initialize a local repo on your machine
  ```sh
  git init
  ```
2. Clone the repo
  ```sh
  git clone https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection.git
  ```
3. Naviagte into the cloned repo directroy
  ```sh
  cd Convolutional_Neural_Network_for_Dementia_Stage_Detection/
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

This project serves as a valuable template for initiating the construction of a CNN designed for medical image classification. The project incorporates auxiliary PyTorch interfaced functions responsible for normalizing the medical dataset, along with early stopper classes that can utilize either loss or F1 score as their evaluation metric. Additionally, it is configured for reproducibility, providing an easy way to scrutinize the impacts of alterations to the model. The model adheres to the standard components of a CNN, encompassing pooling layers, convolutional layers, fully connected layers, ReLU activation, and dropout, with the inclusion of batch normalization between each layer. Despite its simplicity, this project has not undergone extensive testing on data beyond the original dataset, mainly due to the unavailability of private medical imaging. While the model demonstrates commendable performance on the training and validation datasets, its current capacity to generalize beyond these datasets remains uncertain. Nevertheless, this model serves as an excellent educational resource for individuals aiming to grasp convolutional neural networks and PyTorch fundamentals.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Feature Present in Model

- [ ] Pytorch Dataset Loading using ImageFolder
- [ ] Pytorch Image Dataset Preprocessing
    - [ ] Normalization
    - [ ] Tp Tensor
    - [ ] Grayscale Conversion
    - [ ] Resizing
- [ ] Data Partitioning to Traning and Validation Datsets
- [ ] Utilizing Torch Dataloaders
    - [ ] Batch size = 16
- [ ] Convolutional Neural Network Strucutres
    - [ ] Convlutional Layers
    - [ ] Batch Normalization
    - [ ] ReLU Activation
    - [ ] Max Pooling Layers
    - [ ] Fully Connected Layers
    - [ ] Dropout
- [ ] Cross Entropy Loss
- [ ] Adam Optimizer
- [ ] Early Stopper 
- [ ] Plot of Model Training
- [ ] Prediction of Dementia Stage from Transverse Axial MRI Brain Image

See the [open issues](https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection/issues) for a full list of known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Jake Patock - [@jakepatock](https://twitter.com/jakepatock) - jake.r.patock@gmail.com

Project Link: [https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection](https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection.svg?style=for-the-badge
[contributors-url]: https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection.svg?style=for-the-badge
[forks-url]: https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection/network/members
[stars-shield]: https://img.shields.io/github/stars/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection.svg?style=for-the-badge
[stars-url]: https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection/stargazers
[issues-shield]: https://img.shields.io/github/issues/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection.svg?style=for-the-badge
[issues-url]: https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection/issues
[license-shield]: https://img.shields.io/github/license/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection.svg?style=for-the-badge
[license-url]: https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/jake-patock-923089250/
[product-screenshot]: images/screenshot.png
[python]: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue
[python-url]: https://www.python.org
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
