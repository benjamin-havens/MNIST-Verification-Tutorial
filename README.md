# MNIST-Verification-Tutorial
Contains tutorial python scripts for verifying performance of a small MLP on MNIST digit classification.

In particular, uses interval bound propagation (WIP), linear programming (WIP), or mixed integer linear programming to verify that the MLP will correctly classify all images in an L-infinity ball--that is, all images where each pixel can be tweaked independently within plus or minus epsilon--to the correct label. The mixed integer linear programming formulation is also able to identify a counterexample in cases where the network fails. 

## Instructions to run
1. Create a virtual environment using `python3 -m venv venv_mnist_verification`.
2. Run the command `source venv_mnist_verification/bin/activate`.
3. Run the command `pip install -r requirements.txt`.
4. Run the command `python mnist.py`. This will train an MLP on the MNIST dataset, downloading the dataset if necessary, and save the model state dictionary to `mnist_model.pt`. It will use CUDA if available, else MPS, else CPU.
5. Run one of: `python verify_mnist_IBP.py`, `python verify_mnist_LP.py`, `python verify_mnist_MILP.py`.

You can also just import the `MNISTModelVerifier` class and use it as desired. The last step in the above instructions just runs the example usage from the bottom of each file.

I created and ran this project in Python 3.11. Lower versions may also work.

## TODO
- create tutorial
- account for batchnorm for increased performance and test for improved verifiability

## Citations
Salman, Hadi, et al. "A convex relaxation barrier to tight robustness verification of neural networks." Advances in Neural Information Processing Systems 32 (2019).

Tjeng, Vincent, Kai Xiao, and Russ Tedrake. "Evaluating robustness of neural networks with mixed integer programming." arXiv preprint arXiv:1711.07356 (2017).

LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.
