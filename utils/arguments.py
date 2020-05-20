import argparse

parser = argparse.ArgumentParser(description="this sets parameters for experiments")
parser.add_argument(
    "--model", default="DistMult", choices=["DistMult", "Complex", "ConvE"]
)
parser.add_argument(
    "--embdim",
    default=200,
    type=int,
    help="this is the dimensionality of the embeddings",
)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batches", default=100, type=int)
parser.add_argument("--lr", default=0.01, type=float)


args = parser.parse_args()


print(args)
