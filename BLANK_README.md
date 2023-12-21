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



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Convolutional_Neural_Network_for_Dementia_Stage_Detection</h3>

  <p align="center">
    This project involves a CNN model used for predicting the stage of dementia in patients based on transverse MRI brain images. The dataset used by this model was sourced from Kaggle (link: https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset). This dataset comprises 6,400 128x128-pixel images of transverse brain MRI scans from patients at varying dementia stages. The dataset consists of four classes: Class 1 (mildly demented) includes 896 images; Class 2 (moderately demented) has 64 images; Class 3 (non-demented) comprises 3,200 images; and Class 4 (very mildly demented) contains 2,240 images. Notably, the dataset is imbalanced across its class representation.

The dataset was divided into 80% for training and 20% for validation, and the images were converted into a single grayscale channel tensor. A batch size of 64 images per batch was utilized in the data loader. The CNN structure consisted of 5 convolutional layers with a max-pooling layer interspersed among them. The channel sizes progressed in the order of 1, 64, 128, 256, 512, and 512. The final pooling layer was flattened and connected to three fully connected linear layers. After the first two linear layers, dropout of 0.5 was introduced to prevent overfitting. The sizes of these linear layers were as follows: 8192 neurons, 1024 neurons, 256 neurons, and finally 4 neurons.

The model utilized cross-entropy loss for multi-class classification and was optimized using the Adam optimizer. Evaluation was performed using a weighted F1 score, biased towards recall due to the medical nature of the data. The model prioritized avoiding false negatives, as they might lead to patients' symptom dismissal and delayed dementia diagnosis, potentially impacting treatment effectiveness. False positives were deemed less damaging, leading to closer patient inspection and continued testing which results in little harm to the pacient.

Training incorporated an early stopper (with a pacients of 50 epochs) based on maximizing the F1 score on the validation set. The final F1 score on the validation set was [insert F1 score], with an accuracy of [insert accuracy] and loss of [insert loss]. The model achieved an accuracy and loss of [insert accuracy] and [insert loss] on the training data. A figure in the data directory showcases a comparison plot between the model's training and validation accuracy and loss, along with a plot of the F1 scores during training.
    <br />
    <a href="https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection">View Demo</a>
    ·
    <a href="https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection/issues">Report Bug</a>
    ·
    <a href="https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection/issues">Request Feature</a>
  </p>
</div>



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

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Here's a blank template to get started: To avoid retyping too much info. Do a search and replace with your text editor for the following: `jakepatock`, `Convolutional_Neural_Network_for_Dementia_Stage_Detection`, `twitter_handle`, `jake-patock-923089250/`, `gmail`, `jake.r.patock`, `Convolutional Neural Netowrk For Dementia Stage Detection`, `This project involves a CNN model used for predicting the stage of dementia in patients based on transverse MRI brain images. The dataset used by this model was sourced from Kaggle (link: https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset). This dataset comprises 6,400 128x128-pixel images of transverse brain MRI scans from patients at varying dementia stages. The dataset consists of four classes: Class 1 (mildly demented) includes 896 images; Class 2 (moderately demented) has 64 images; Class 3 (non-demented) comprises 3,200 images; and Class 4 (very mildly demented) contains 2,240 images. Notably, the dataset is imbalanced across its class representation.

The dataset was divided into 80% for training and 20% for validation, and the images were converted into a single grayscale channel tensor. A batch size of 64 images per batch was utilized in the data loader. The CNN structure consisted of 5 convolutional layers with a max-pooling layer interspersed among them. The channel sizes progressed in the order of 1, 64, 128, 256, 512, and 512. The final pooling layer was flattened and connected to three fully connected linear layers. After the first two linear layers, dropout of 0.5 was introduced to prevent overfitting. The sizes of these linear layers were as follows: 8192 neurons, 1024 neurons, 256 neurons, and finally 4 neurons.

The model utilized cross-entropy loss for multi-class classification and was optimized using the Adam optimizer. Evaluation was performed using a weighted F1 score, biased towards recall due to the medical nature of the data. The model prioritized avoiding false negatives, as they might lead to patients' symptom dismissal and delayed dementia diagnosis, potentially impacting treatment effectiveness. False positives were deemed less damaging, leading to closer patient inspection and continued testing which results in little harm to the pacient.

Training incorporated an early stopper (with a pacients of 50 epochs) based on maximizing the F1 score on the validation set. The final F1 score on the validation set was [insert F1 score], with an accuracy of [insert accuracy] and loss of [insert loss]. The model achieved an accuracy and loss of [insert accuracy] and [insert loss] on the training data. A figure in the data directory showcases a comparison plot between the model's training and validation accuracy and loss, along with a plot of the F1 scores during training.`

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/jakepatock/Convolutional_Neural_Network_for_Dementia_Stage_Detection/issues) for a full list of proposed features (and known issues).

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



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

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
[Next.js]: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue
[Next-url]: https://www.python.org
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
