"""
data/download_mnist.py
======================
Download and verify the MNIST dataset.

MNIST is publicly available and downloaded automatically by torchvision.
This script provides explicit download with integrity checking and progress output.

Usage:
    python data/download_mnist.py --data-dir ./data

Paper: Srivastava et al. (2014) JMLR 15:1929-1958.
    Dataset: MNIST — "A standard toy data set of handwritten digits" (Section 6).
    Training: 60,000 examples | Test: 10,000 examples (Table 1).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MNIST dataset")
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory to store downloaded MNIST data"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    mnist_dir = data_dir / "MNIST"
    if mnist_dir.exists() and (mnist_dir / "raw").exists():
        raw_files = list((mnist_dir / "raw").glob("*.gz"))
        if len(raw_files) >= 4:
            print(f"MNIST already downloaded at: {mnist_dir}")
            print("Skipping download.")
            return

    print(f"Downloading MNIST to: {data_dir}")
    print("Expected size: ~11 MB")
    print("Files: train-images, train-labels, t10k-images, t10k-labels")

    try:
        from torchvision.datasets import MNIST
        from torchvision import transforms

        transform = transforms.Compose([transforms.ToTensor()])

        print("\nDownloading training set (60,000 samples)...")
        train = MNIST(root=data_dir, train=True, download=True, transform=transform)
        print(f"  Training set: {len(train):,} samples")

        print("Downloading test set (10,000 samples)...")
        test = MNIST(root=data_dir, train=False, download=True, transform=transform)
        print(f"  Test set:     {len(test):,} samples")

        # Quick integrity check: sample a batch
        from torch.utils.data import DataLoader
        loader = DataLoader(train, batch_size=32, shuffle=False)
        x, y = next(iter(loader))
        assert x.shape == (32, 1, 28, 28), f"Unexpected shape: {x.shape}"
        assert y.shape == (32,), f"Unexpected label shape: {y.shape}"
        assert x.min() >= 0.0 and x.max() <= 1.0, "Pixel values out of [0,1] range"

        print(f"\nMNIST downloaded and verified successfully!")
        print(f"  Location: {data_dir / 'MNIST'}")
        print(f"  Pixel range: [{x.min():.3f}, {x.max():.3f}]")
        print(f"  Label range: [{y.min().item()}, {y.max().item()}]")
        print(f"\nReady to train. Run:")
        print(f"  python train.py --config configs/mnist_3layer_1024.yaml")

    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        print("\nManual download instructions:")
        print("  Visit: http://yann.lecun.com/exdb/mnist/")
        print("  Or: https://storage.googleapis.com/cvdf-datasets/mnist/")
        print("  Download the 4 .gz files to ./data/MNIST/raw/")
        sys.exit(1)


if __name__ == "__main__":
    main()
