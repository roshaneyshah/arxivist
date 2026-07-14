"""
inference.py
============
Single-image inference entrypoint for the Dropout reproduction.

Runs a trained DropoutNet on a single input image and outputs the predicted
digit class along with softmax probabilities.

Usage:
    python inference.py --checkpoint checkpoints/dropout_repro/best.pt \\
                        --image path/to/digit.png \\
                        --config configs/mnist_3layer_1024.yaml

    # Monte-Carlo model averaging (Section 7.5) — sample k dropout masks:
    python inference.py --checkpoint checkpoints/dropout_repro/best.pt \\
                        --image path/to/digit.png --monte-carlo 50

Paper: Srivastava et al. (2014) JMLR 15:1929-1958.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn.functional as F

from dropout_repro.models.dropout_net import DropoutNet
from dropout_repro.utils.config import DropoutConfig, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained Dropout checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved .pt checkpoint")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image (28x28, grayscale PNG/JPG)")
    parser.add_argument("--config", type=str, default="configs/mnist_3layer_1024.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--monte-carlo", type=int, default=0,
                        help=(
                            "If > 0, use Monte-Carlo model averaging with k sampled "
                            "dropout masks instead of weight-scaling approximation. "
                            "Section 7.5: k≈50 matches weight-scaling accuracy."
                        ))
    parser.add_argument("--device", type=str, default=None,
                        help="Device override")
    return parser.parse_args()


def load_image(image_path: str, mean: float, std: float) -> torch.Tensor:
    """
    Load and preprocess a single image for inference.

    Args:
        image_path: Path to PNG or JPG image (28x28 pixels expected).
        mean:       Normalization mean.
        std:        Normalization std.

    Returns:
        Preprocessed tensor of shape [1, 784].
    """
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    img = Image.open(image_path)
    x = transform(img).unsqueeze(0)  # [1, 784]
    return x


def predict_with_weight_scaling(
    model: DropoutNet,
    x: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Standard inference: model.eval() disables dropout (weight-scaling approximation).

    Section 7.5: "The efficient test time procedure that we propose is to do an
    approximate model combination by scaling down the weights of the trained neural
    network." (PyTorch inverted dropout achieves this automatically.)

    Returns:
        Dict with 'predicted_class', 'probabilities', 'confidence'.
    """
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)  # [1, 10]
        probs = F.softmax(logits, dim=1).squeeze(0)  # [10]
        predicted = probs.argmax().item()
    return {
        "predicted_class": predicted,
        "probabilities": probs.cpu().tolist(),
        "confidence": probs[predicted].item(),
        "method": "weight_scaling",
    }


def predict_with_monte_carlo(
    model: DropoutNet,
    x: torch.Tensor,
    device: torch.device,
    k: int = 50,
) -> dict:
    """
    Monte-Carlo model averaging: sample k thinned networks and average predictions.

    Section 7.5: "An expensive but more correct way of averaging the models is to
    sample k neural nets using dropout for each test case and average their predictions.
    As k→∞, this Monte-Carlo model average gets close to the true model average."

    Figure 11: k≈50 matches the weight-scaling approximation performance.

    Args:
        model: DropoutNet (dropout active during forward pass in train mode).
        x:     Input tensor [1, 784].
        k:     Number of sampled dropout masks.

    Returns:
        Dict with 'predicted_class', 'probabilities', 'confidence', 'std'.
    """
    # Set to train mode to ENABLE dropout masks
    model.train()
    x = x.to(device)

    all_probs = []
    with torch.no_grad():
        for _ in range(k):
            logits = model(x)  # [1, 10] — different mask each forward pass
            probs = F.softmax(logits, dim=1).squeeze(0)  # [10]
            all_probs.append(probs)

    # Average predictions across k sampled networks
    all_probs_tensor = torch.stack(all_probs)  # [k, 10]
    mean_probs = all_probs_tensor.mean(dim=0)  # [10]
    std_probs = all_probs_tensor.std(dim=0)    # [10]

    predicted = mean_probs.argmax().item()

    # Return to eval mode
    model.eval()

    return {
        "predicted_class": predicted,
        "probabilities": mean_probs.cpu().tolist(),
        "confidence": mean_probs[predicted].item(),
        "std": std_probs[predicted].item(),
        "method": f"monte_carlo_k{k}",
        "k": k,
    }


def main() -> None:
    args = parse_args()

    config = DropoutConfig.from_yaml(args.config)
    if args.device:
        config.hardware.device = args.device
    device = get_device(config.hardware.device)

    # --- Load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_mc = ckpt.get("config", {}).get("model", {})

    model = DropoutNet(
        input_dim=saved_mc.get("input_dim", config.model.input_dim),
        hidden_dims=saved_mc.get("hidden_dims", config.model.hidden_dims),
        num_classes=saved_mc.get("num_classes", config.model.num_classes),
        p_hidden=saved_mc.get("p_hidden", config.model.p_hidden),
        p_input=saved_mc.get("p_input", config.model.p_input),
        activation=saved_mc.get("activation", config.model.activation),
        use_dropout=saved_mc.get("use_dropout", config.model.use_dropout),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # --- Load image ---
    x = load_image(args.image, config.data.normalize_mean, config.data.normalize_std)

    # --- Predict ---
    if args.monte_carlo > 0:
        result = predict_with_monte_carlo(model, x, device, k=args.monte_carlo)
    else:
        result = predict_with_weight_scaling(model, x, device)

    # --- Display results ---
    print(f"\n{'='*40}")
    print("INFERENCE RESULT")
    print(f"{'='*40}")
    print(f"  Image:          {args.image}")
    print(f"  Method:         {result['method']}")
    print(f"  Predicted digit: {result['predicted_class']}")
    print(f"  Confidence:      {result['confidence']*100:.2f}%")
    if "std" in result:
        print(f"  Std dev (k={result['k']}): {result['std']*100:.2f}%")
    print(f"\n  Class probabilities:")
    for digit, prob in enumerate(result["probabilities"]):
        bar = "█" * int(prob * 40)
        marker = " ← predicted" if digit == result["predicted_class"] else ""
        print(f"    {digit}: {prob*100:5.2f}% {bar}{marker}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
