# MNIST-Verification-Tutorial
Contains tutorial python scripts for verifying performance of a small MLP on MNIST digit classification.

In particular, uses interval bound propagation (WIP), linear programming (WIP), or mixed integer linear programming to verify that the MLP will correctly classify all images in an L-infinity ball--that is, all images where each pixel can be tweaked independently within plus or minus epsilon--to the correct label. The mixed integer linear programming formulation is also able to identify a counterexample in cases where the network fails. 

## Instructions to run
1. Create a virtual environment using `python3 -m venv venv_mnist_verification`.
1. Run the command `source venv_mnist_verification/bin/activate`.
1. Run the command `pip install -r requirements.txt`.
1. In `mnist.py`, edit the device to one supported on your system. This should be one of `{"cpu", "mps", "cuda"}` or another accelerator backend supported by torch 2.0. I use mps as I ran this on a MacBook with M2 silicon.
1. Run the command `python mnist.py`. This will train an MLP on the MNIST dataset, downloading the dataset if necessary, and save the model state dictionary to `mnist_model.pt`.
1. Run one of: `python verify_mnist_IBP.py`, `python verify_mnist_LP.py`, `python verify_mnist_MILP.py`.

You can also import the `MNISTModelVerifier` class and use it as desired. 

I created and ran this project in Python 3.11, but it should work for lower versions. I have not tested this, but I'd guess 3.9 or even 3.8 would work.

## TODO
- finish updating README, with sources, further instructions
- create tutorial
- account for batchnorm for increased performance and test for improved verifiability
- comment/document the code
- identify and fix problem in LP or IBP formulation

