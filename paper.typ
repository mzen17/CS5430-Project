= MINST Number Generation with VAE vs Langevin Sampling Expectation Maximization

Mike Zeng \<mzeng5\@uiowa.edu\>

== Introduction
This report compares the use of variational autoencoders to an EM-algorithm like approach using Langevin sampling. I compare latent space and the final output of the generated images using the MINST dataset comprised of 67000 28x28 images of numbers from 0-9 colored in grayscale. 
#image("images/demo.png")
== Design and Methods
=== Variational Autoencoder
We tested with a CVAE, a class-conditioned variational autoencoder. The encoding model takes in an linear vector of 784 floats scaled between 0 and 1 alongside a onehot class tensor representing digits 0-9. 

The variational autoencoder uses 3 layers for the encoding model of dimensions 784x256, 256x64, and 64x8. Notice the latent dimension is $8$. The decoder uses 2 layers of 8x64, 64x784. The variance is fixed at 0. Initally, the model was tested with learned variance but training ended up highly instable with the KLD exploding. It outputs fixed log variance of $0$ instead, aka variance = I.

Let the ELBO be defined as $log p_theta (x, z) - D_"kl" [ q(z|x) || p(z)] ~ D_"kl" [q(z|x) || p_theta (z|x)]$, where both are Guassians. We note that we simplify the KL term, since both are Guassians. From homework 1, we recall that the KLD between two single variate Gaussians for a latent dimension $j$ is:
$ ln s_z/sigma_theta - 1/2 + (sigma_theta^2 + (mu_theta -mu_z)^2)/(2sigma_z^2)  $
Since $p(z)$ has variance of I, our formula becomes:
$ -1/2ln sigma_theta - 1/2 + 1/2 (sigma_theta^2 + (mu_theta -mu_z)^2)  $
$ - 1/2 (1 - sigma_theta^2- mu_theta^2 - mu_z^2 +2 mu_theta mu_z +ln sigma_theta)  $
Since $mu_z = 0$ because $p(z) in N(0, 1)$, this becomes:
$ - 1/2 (1 - sigma_theta^2 - mu_theta^2 +ln sigma_theta) $
Doing a summation of this over all latents gives us our KL term:
$ sum_j - 1/2 (1 - sigma_j^2 - mu_j^2 +ln sigma_j)  $

Now to obtain $log p_theta (x)$, we note that because $p_theta (x)$ is a Guassian with $mu$ and fixed variance $I$, we have:
$
p_theta (x | mu, I)
=
frac(1, sqrt((2 pi)^d det(I)))
exp(
  -frac(1, 2)
  (x - mu)^T (I)^(-1) (x - mu)
)
$

$
p_theta (x | mu, I)
=
frac(1, sqrt((2 pi)^d))
exp(
  -frac(1, 2)
  (x - mu)^2
)
$
$
log p_theta (x | mu, I)
=
d/2 log(2pi) 
  -frac(1, 2)
  (x - mu)^2
)
$

This gives us our reconstruction term, where $x$ is our data and $mu$ is taken from the decoder head. Our loss is reconstruction term + kl term.

=== EM Algorithm
Here, we remove the encoder as rather than training a model for $q(z|x)$, we use an E-step like approach for finding an approximate representation of $p(z|x)$. The decoder is trained with the latents from the E-step and its parameters are hence updated in the M-step.

=== Langevin Sampling for E-step
Formally, we replace the encoder $q_lambda (z|x)$ with Langevin sampling to approximate the posterior $p(z|x)$. Given a latent representation, we can find a better representation as follows:
$ z_(t+1) = z_t + nabla log p(z|x) + sqrt(2 eta )epsilon $
Thus, our sampling iteration becomes:
$ z_(t+1) = z_t + nabla log p(z) + nabla log p(x|z) + sqrt(2 eta) epsilon $ 

If we let $p(z)$ be a Gaussian of $z in N(0, I)$, we have that 

$ log p(z) = -n/2 log(2pi) - 1/2 ||mu-z|| $
$ nabla log p(z) =  -1/2  2 (mu-z)nabla (mu-z) $
$ nabla log p(z) =  - (0-z)(-1) $
$ nabla log p(z) =  - z $

hence the first term becomes $-z$. The second term relies on our decoder model $p(x|z)$, so we obtain this through the gradient function of Pytorch. This can be done by feeding the latent through the decoder model. After our decoder model generates the mean and variance, we obtain $p_theta (x|z)$, and hence we can obtain the negative log-likelihood of observing $x$, then use `autograd.grad` to obtain the gradient. Hence, our update step becomes:
$ z_(t+1) = z_t - z + nabla log p(x|z) + sqrt(2 eta) epsilon $ 
Note that the $-z$ pulls the update closer to 0 while the $nabla log p(x|z)$ pulls the log-likelihood of generating the data $x$ higher.

I found that using randomly initialized vectors for the starting $z$ resulted in the model having poor performance, so I trained the EM-approximation setup with $z$ initialized from the encoding model with its weights frozen.

=== M-step Update
Using the finalized latent $z$, we update our decoder. Using the batch group $b$, we take the negative gaussian likelihood of our batch as our loss. The decoder is updated using the Adam optimizer on the loss. 




== Results
=== Generated Images
We list the batch generations for each configuration. More generations can be found in https://github.com/mzen17/CS5430-Project/examples.


==== EM-approximation with Langevin, 10000 minibatch descends with 64 image batch, 150 iterations
#image("images/lem-10k.png")

==== VAE with 10000 minibatch descends, 64 image batch
#image("images/vae10k.png")
=== Internal Representations
I visualize 1000 latents from the finalized z-bank for the Langevin sampling approach doing another 150 steps. VAE encoder latents are generated by plugging the MINST into the encoder.
#image("images/latents.png")
#pagebreak()
=== Log likelihood Approximations
log p(x|z) was approximated using 10000 z samples. We visualize the likelihood here:
#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 10pt,
  align: (right, right, right, right, right),
  table.header(
    [*image*], [*label*], [*VAE log p(x)*], [*LEM log p(x)*], [*LEM - VAE*],
  ),
  [0], [7], [-731.0604], [-730.8820], [0.1784],
  [1], [2], [-743.6703], [-743.3235], [0.3468],
  [2], [1], [-725.8842], [-725.9359], [-0.0517],
  [3], [0], [-735.3418], [-735.9449], [-0.6031],
  [4], [4], [-734.7455], [-735.2046], [-0.4590],
  [5], [1], [-725.4033], [-725.5439], [-0.1406],
  [6], [4], [-740.4456], [-740.6231], [-0.1776],
  [7], [9], [-740.7876], [-740.9551], [-0.1675],
  [*mean*], [-], [-735.8259], [-735.8308], [-0.0049],
)
=== Training Graphs and Time
The models were trained on an NVIDIA RTX 5070 Ti. The LEM model took around 3 minutes to train for 10000 steps while the VAE took about 20 seconds to train.

== Analysis
Overall, the likelihood of the data had little changes regardless of Langevin sampling VS VAE encoding. However, the latents produced by the VAE seemed to be a lot more clustered, wherevers the latents by the Langevin sampling was more randomly distributed.

The picture quality of the numbers were relatively similar. The Langevin sampling did produce sharper pictures, although it is not seen here because it occured by random chance. At lower training minibatches, Langevin sampling had much better and sharper images. However, the most noticeable thing was the training time. Using the Langevin sampling, it was around a 10x slowdown compared to training the VAE.