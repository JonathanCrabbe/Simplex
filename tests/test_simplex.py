from simplexai.explainers.simplex import Simplex
from simplexai.models.image_recognition import MnistClassifier
from simplexai.experiments.mnist import load_mnist

def test_sanity():
# Get a model
    model = MnistClassifier() # Model should have the BlackBox interface

# Load corpus and test inputs
    corpus_loader = load_mnist(subset_size=100, train=True, batch_size=100) # MNIST train loader
    test_loader = load_mnist(subset_size=10, train=True, batch_size=10) # MNIST test loader
    corpus_inputs, _ = next(iter(corpus_loader)) # A tensor of corpus inputs
    test_inputs, _ = next(iter(test_loader)) # A set of inputs to explain

# Compute the corpus and test latent representations
    corpus_latents = model.latent_representation(corpus_inputs).detach()
    test_latents = model.latent_representation(test_inputs).detach()

# Initialize SimplEX, fit it on test examples
    simplex = Simplex(corpus_examples=corpus_inputs,
                      corpus_latent_reps=corpus_latents)
    simplex.fit(test_examples=test_inputs,
                test_latent_reps=test_latents,
                reg_factor=0)

# Get the weights of each corpus decomposition
    weights = simplex.weights

    assert weights.shape == (10, 100)
