### GAN
---
- suppose we are given images of dogs
- We want to learn a probability distribution P(x) such that
`Generation`

`Density estimation`

`Unsupervised representation learning`

### Basic Discrete Distributions
---

- Bernoulli distribution

- Categorical distribution 
```
Modeling an RGB joint distribution(of a single pixel)

number of cases? 256*256*256

Example:

supppose we have X1...Xn of n binary pixels( a binary image)

how many possible states? 2*2*...2 = 2^n

sampling from p(x1...xn) generates an image

How many parameters to specify p(x1...xn)? 2^n-1

```
---
- conditional Independence
- Three important rules

- chain rules:
    - parameters? 2^n-1

- Bayes's rule:

- Conditional Independence: i번째 픽셀은 i-1번째에만 dependent 하고 나머지에는 independent 하다 라고 가정한다 
    - parameters? 2n-1 

---

Auto-regressive Modele

- suppose we have 28*28binaary pixels

- our goal to learn p(x)=p(x1....x784) 

- how can we parametrize p(x)?
---
### NADE: Neural Autogressive Density Estimator
---

- The probability distribution of i-th pixel 

- Nade is an explicit model that compute the density of the given inputs
 
- how can ew compute the density of the given image?
    - 확률을 명확하게 계산할 수 있다. 
---
### Pixel RNN
---

- We can also use RNNS to define an auto-regressibe model 

- for example for an n*n RGB image 

- There are two model architectures in pixel RNN based on the ordering of chain 
    - Row Lstm
    - Diagonal BiLSTM
---

### Practical 한 Generative model 인 Variational Auto Encoder 와 Generative Adversarial Network 를 이용하여 Latent variable model 다루기 
---
- Further Reading

    - (1시간만에 GAN 완전 정복하기)[https://www.youtube.com/watch?v=odpjk7_tGY0&t=69s]
    - (An introduction to varaiational Autoencoders)[https://arxiv.org/abs/1906.02691]
---

### variational Auto-encoder
- variational inference(VI)

    - the goal of VI is to optimize the variational distribution that best mathes the posterior distribution
        
        - Posterior distribution
        - Variational distribution
    - In particular we want to find the varaiational distribution that minimizes the KL divergence between the true posterior

    `타겟을 모르는데 어떻게 loss function을 구할 수 있을까???`

    - 수식을 이해해보기 

    - ELBO can further be decomposed into ~

    - ELBO 를 MAXIMIZE하는 것이 어떤 loss 를 최소화한다는걸 증명하는 수식 

    - Key limitation : It is a intractable model( hard to evaluate likelihood)
    - ther prior fitting must be differentiable, hence it is hard to use diverse latent prior distribution
    - in most cases we use an `isotropic Gaussian`

`Adversarial Auto-encoder`-> 학습하기 (더 찾아보기 중요한 개념)

---
### GAN
---

- two player minimax game between generator and discriminator

- for discriminator

- plugging in the optimal discriminator we get ~(수식) `2*Jenson-Shannon Divergenece(JSD)`

---
1. DCGAN

2. Info-GAN

3. Text2Image->Dalle(가장 최신 논문인 달리와 비슷하다)

4. puzzle_GAN->subpatch 를 통해 원래 이미지를 복원하는

5. CycleGAN(중요)-> GAN 구조가 2개가 들어가게 된다 

6. Star-GAN-> 이미지를 단순히 다른 도메인으로 바꾸는 것이 아니라 도메인을 정할 수 있다.

7. Progessive-GAN -> 고차원의 이미지를 만들 수 있는 방법론 4*4 -> 16*16 -> 1024*1024 로 저차원에서 고차원으로 이미지를 조금씩 복원하는 방법 



