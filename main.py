import argparse

from src.train import train
from src.eval import evaluate
from src.inference import predict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", type=str, required=True, choices=["train", "eval", "predict"]
    )

    parser.add_argument(
        "--viz", action="store_true", help="Save visualization during evaluation"
    )

    parser.add_argument(
        "--image", type=str, help="Path to input image (required for predict mode)"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "eval":
        evaluate(save_viz=args.viz)

    elif args.mode == "predict":
        if args.image is None:
            raise ValueError("❌ Please provide --image path for prediction")
        predict(args.image)


if __name__ == "__main__":
    main()
