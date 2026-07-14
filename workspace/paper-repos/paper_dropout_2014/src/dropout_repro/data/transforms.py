"""
data/transforms.py
==================
Data preprocessing pipelines.

MNIST normalization constants (mean=0.1307, std=0.3081) are the standard
values computed over the full MNIST training set. The paper applies standard
normalization for MNIST (Appendix B.1: inputs are 28×28 pixel images).

Windows compatibility note:
    torchvision.transforms.Lambda uses a lambda function which cannot be
    pickled by Windows multiprocessing (spawn method). We use a named
    callable class FlattenTransform instead, which is fully picklable.

Paper: Srivastava et al. (2014) JMLR 15:1929-1958, Appendix B.1.
"""

from torchvision import transforms


class FlattenTransform:
    """
    Picklable flatten transform: [C, H, W] → [C*H*W] 1-D vector.

    Replaces transforms.Lambda(lambda x: x.view(-1)) to support
    Windows multiprocessing DataLoader workers (spawn method requires
    all transform objects to be picklable; lambdas are not).
    """
    def __call__(self, x):
        return x.view(-1)

    def __repr__(self) -> str:
        return "FlattenTransform()"


def get_mnist_transforms(
    mean: float = 0.1307,
    std: float = 0.3081,
) -> tuple:
    """
    Return (train_transform, test_transform) for MNIST.

    Both transforms flatten the 28×28 image to a 784-dim vector, as required
    by the fully-connected DropoutNet. The paper uses the permutation-invariant
    setting (no 2D spatial structure exploited).

    Appendix B.1: "The MNIST data set consists of 28×28 pixel handwritten digit
    images." No special augmentation is used for the permutation-invariant setting.

    Args:
        mean: Pixel intensity mean for normalization (default: MNIST standard).
        std:  Pixel intensity std  for normalization (default: MNIST standard).

    Returns:
        train_transform: torchvision Transform for training data.
        test_transform:  torchvision Transform for test/validation data.
    """
    # Permutation-invariant setting: flatten to vector, normalize.
    # FlattenTransform is used instead of Lambda to support Windows
    # multiprocessing DataLoader workers (lambdas are not picklable).
    base = transforms.Compose([
        transforms.ToTensor(),               # [0,255] → [0,1], shape [1,28,28]
        transforms.Normalize((mean,), (std,)),  # zero-mean, unit-ish variance
        FlattenTransform(),                  # [1,28,28] → [784] flat vector
    ])

    # No augmentation for permutation-invariant MNIST (Section 6.1.1)
    train_transform = base
    test_transform = base

    return train_transform, test_transform
