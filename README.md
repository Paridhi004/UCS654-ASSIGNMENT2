# UCS654-ASSIGNMENT2
## Assignment-2
## Learning Probability Density Functions Using Roll-Number-Parameterized Non-Linear Model
## 1. Introduction
This notebook demonstrates the implementation and training of a Generative Adversarial Network (GAN) to learn an unknown, one-dimensional data distribution. The process involves using Nitrogen Dioxide (NO₂) concentration values from the India Air Quality Dataset, applying a unique nonlinear transformation derived from a university roll number, and then training a GAN on these transformed samples. Finally, the trained generator is used to produce new samples, and their probability density function (PDF) is estimated and compared with the original transformed data's PDF.

## 2. Dataset Used
The dataset used for this assignment is the India Air Quality Dataset obtained from Kaggle (loaded in the first cell as df).

Original Samples (x): From the complete dataset, only the Nitrogen Dioxide (NO₂) concentration values are used for analysis. These values are represented by the variable x. Missing and invalid entries in the NO2 column are handled (e.g., removed) before further processing.

Transformed Samples (z): These x samples (NO₂ values) are then transformed into z samples using a roll-number-dependent nonlinear equation: [ z = x + a_r \sin(b_r x) ]
where a_r and b_r are calculated based on the user's university roll number (r). The GAN is then trained to learn the probability distribution of these z samples.

## 3. Methodology
The overall approach consists of three main stages as described below.

### Step 1: Roll-Number-Based Nonlinear Transformation
Each x value is transformed into a new variable z using the nonlinear transformation:

[ z = x + a_r \sin(b_r x) ]

The parameters a_r and b_r are calculated from the university roll number r using the formulas:

[ a_r = 0.5 \times (r \bmod 7) ]

[ b_r = 0.3 \times ((r \bmod 5) + 1) ]

For the roll number 102303715 (as entered in the notebook), the computed values are:

| Parameter | Value |
|----------|-------|
| a_r | 1.5 |
| b_r | 0.3 |

This roll-number-dependent transformation ensures that each student works with a unique transformed dataset.

### Step 2: Learning the Probability Distribution Using GAN
The transformed variable z is assumed to be sampled from an unknown probability distribution. No parametric form (such as Gaussian or exponential) is assumed.

A Generative Adversarial Network (GAN) is designed and trained using only samples of z:

The Generator takes random noise sampled from a standard normal distribution N(0,1) and generates fake samples of z.
The Discriminator attempts to distinguish between real transformed samples and fake samples produced by the generator.
Both networks are trained adversarially until the generator produces samples that resemble the real transformed data.

### Step 3: PDF Approximation from Generator Samples
After training the GAN, a large number of samples are generated using the trained generator. These generated samples are used to estimate the probability density function of z using Kernel Density Estimation (KDE).

The estimated density represents the learned probability distribution of the transformed variable.

## 4. GAN Architecture
Generator Network
Input: 100-dimensional random noise vector (latent_dim = 100).
Hidden Layers: Two fully connected layers with 256 neurons each (hidden_dim = 256), followed by a LeakyReLU activation function (negative slope 0.2).
Output: Single scalar value representing the generated z sample. The final layer uses a Tanh() activation, scaling the output to a range of [-1, 1] which is suitable for the distribution of z.
Discriminator Network
Input: Single z value (real or fake).
Hidden Layers: Two fully connected layers with 256 neurons each (hidden_dim = 256), followed by a LeakyReLU activation function (negative slope 0.2).
Output: Single scalar value representing the logit (raw output before sigmoid) indicating whether the sample is classified as real or fake. The BCEWithLogitsLoss combines the sigmoid activation internally during loss calculation.
The architectures are lightweight and suitable for learning a one-dimensional probability distribution.

## 5. Results
The trained generator produces samples whose estimated probability density function closely matches the structure of the transformed data distribution. As shown in the KDE plot above (cell a9ed7887), the learned PDF is smooth and captures the dominant characteristics of the true z distribution. The overlap between the 'True Distribution' and 'Generated Distribution' in the plot visually confirms the generator's ability to approximate the target distribution.

## 6. Observations
### Mode Coverage:
The generator effectively captures the main shape and range of the true z distribution, indicating good mode coverage. While it might not perfectly replicate every subtle nuance, it avoids severe mode collapse and represents the overall structure well.

### Training Stability:
Training remained relatively stable over the 500 epochs, as indicated by the decreasing yet fluctuating D_Loss and G_Loss values. There were no signs of the GAN collapsing or diverging significantly, suggesting a balanced training process between the discriminator and generator.

### Quality of Generated Distribution:
The generated samples z_f result in a smooth and realistic probability density function that closely reflects the underlying structure of the transformed data z. The visual comparison of the KDE plots confirms that the GAN has successfully learned to generate samples that mimic the target distribution's characteristics.

## 7. Graph Plot
<img width="1484" height="555" alt="image" src="https://github.com/user-attachments/assets/572a2c03-a493-4c11-a755-ecb61cd452b8" />


## 8. Conclusion
In this assignment, a roll-number-based nonlinear transformation was applied to a set of uniformly distributed samples. A Generative Adversarial Network (GAN) was successfully implemented and trained to learn the probability density function of the transformed variable (z), demonstrating its capability to approximate complex, unknown distributions from data without assuming any analytical form. The results, as shown in the comparative KDE plot, indicate that the GAN effectively captured the underlying distribution, showcasing the power of GANs in modeling and generating data that closely resembles a target distribution.

## 9. Software and Tools Used
1. Google Colab
2. Python
3. NumPy
3. PyTorch
4. Matplotlib
5. Seaborn
6. Kagglehub
