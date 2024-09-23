# Responsible ML with Insurance Applications

Welcome to our lecture. It covers the following main topics:

- Statistical learning, model comparison, and calibration assessment (Christian)
- Explainability (Michael)

From time to time, we will update the material linked below. You can also clone the repository with 

> "git clone https://github.com/lorentzenchr/responsible_ml_material.git"

## Christian's Material

### Slides

[Slides (pdf)](https://github.com/lorentzenchr/responsible_ml_material/blob/main/lecture_slides.pdf)

### Main reference

Tobias Fissler, Christian Lorentzen, and Michael Mayer. “Model Comparison and Calibration Assessment: User Guide for Consistent Scoring Functions in Machine Learning and Actuarial Practice”. In: (2022). [doi: 10.48550/ARXIV.2202.12780](https://doi.org/10.48550/ARXIV.2202.12780)

[Python and R code for the tutorial](https://github.com/actuarial-data-science/Tutorials/tree/master/11%20-%20Model%20Comparison%20and%20Calibration%20Assessment)

## Michael's Material

### Slides

[Slides XAI (pdf)](https://github.com/lorentzenchr/responsible_ml_material/blob/main/slides_xai.pdf)

### Lecture notes

Note that the Python and R outputs differ.

Python notebooks (ipynb)

1. [Introduction](py/xai_1_introduction.ipynb)
2. [Explaining Models](py/xai_2_explaining_models.ipynb)
3. [Improving Explainability](py/xai_3_improving_explainability.ipynb)

R output (HTML)

1. [Introduction](https://lorentzenchr.github.io/responsible_ml_material/xai_1_introduction.html)
2. [Explaining Models](https://lorentzenchr.github.io/responsible_ml_material/xai_2_explaining_models.html)
3. [Improving Explainability](https://lorentzenchr.github.io/responsible_ml_material/xai_3_improving_explainability.html)

#### Setup

Python: We use Python 3.11 and the packages specified [here](py/requirements.txt).

(Note for R users: We use R 4.3 and the packages loaded in the notebooks.)

## Additional Literature

### Model evaluation and scoring functions

- T. Gneiting. “Making and Evaluating Point Forecasts”. In: Journal of the American Statistical Association 106.494 (2011), pp. 746–762. doi: 10.1198/jasa.2011.r10138. [arXiv: 0912.0902](https://doi.org/10.48550/arXiv.0912.0902)
- T. Gneiting and A. E. Raftery. “Strictly Proper Scoring Rules, Prediction, and Estimation”. In: Journal of the American Statistical Association 102 (2007), pp. 359–378. doi: 10.1198/016214506000001437. url: http://www.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf
- A. Buja, W. Stuetzle, and Y. Shen. Loss Functions for Binary Class Probability Estimation and Classification: Structure and Applications. Tech. rep. University of Pennsylvania, 2005. url: http://www-stat.wharton.upenn.edu/~buja/PAPERS/paper-proper-scoring.pdf

### Explainability

- C. Lorentzen and M. Mayer. “Peeking into the Black Box: An Actuarial Case Study for Interpretable Machine Learning”. In: SSRN Manuscript ID 3595944 (2020). [doi: 10.2139/ssrn.3595944](https://doi.org/10.2139/ssrn.3595944).
- M. Mayer, D. Meier, and M. V. Wüthrich. “SHAP for Actuaries: Explain Any Model”. In: SSRN Manuscript ID 4389797 (2023) [doi: 10.2139/ssrn.4389797](https://doi.org/http://dx.doi.org/10.2139/ssrn.4389797).
- Christoph Molnar. Interpretable Machine Learning. 1st ed. Raleigh, North Carolina: Lulu.com, 2019. isbn: 978-0-244-76852-2. url: https://christophm.github.io/interpretable-ml-book

### Books on responsible ML or AI

- Alyssa Simpson Rochwerger and Wilson Pang. Real World AI: A Practical Guide for Responsible Machine Learning. Lioncrest Publishing, 2021
- Patrick Hall, James Curtis, and Parul Pandey. Machine Learning for High-Risk Applications. O’Reilly Media, Inc., 2022
