from models.DistMult import DistMult
from models.Complex import Complex
from models.ConvE import ConvE, ConvE_args

from utils.loaders import load_data, get_onehots
from utils.evaluation_metrics import SRR, auprc_auroc_ap
from utils.path_manage import get_files

import torch
import numpy as np
from sklearn.utils import shuffle
import argparse


def main(model, optimiser, test_data, epochs, batches):
    mean_reciporical_rank = 0
    break_condition = False
    for epoch in range(epochs):
        # training stage
        model.train()
        objects, subjects, relationships = load_data(test_data, batches)

        for index in range(batches):

            obj = torch.LongTensor(objects[index])
            rel = torch.LongTensor(relationships[index])
            subj = torch.squeeze(torch.LongTensor(subjects[index]))

            optimiser.zero_grad()
            pred = model.forward(e1=obj, rel=rel)
            loss = model.loss(pred, subj)
            loss.backward()
            optimiser.step()

        # evaluation stage

        if epoch % 10 == 0:

            model.eval()
            objects, subjects, relationships = load_data(test_data, batches)
            total_sum_reciporical_rank = torch.zeros(1)

            for index in range(batches):

                obj = torch.LongTensor(objects[index])
                rel = torch.LongTensor(relationships[index])
                targets = torch.LongTensor(subjects[index])

                predictions = model.forward(e1=obj, rel=rel)
                srr = SRR(predictions, targets)
                total_sum_reciporical_rank = total_sum_reciporical_rank + srr

            print("mean reciporical rank is...")
            print(total_sum_reciporical_rank / len(test_data))
            MRR = total_sum_reciporical_rank / len(test_data)
            if MRR < (mean_reciporical_rank * 1.05):
                torch.save(
                    model,
                    "Model_{model}_Epoch_{epoch}_MRR_{MRR}.pickle".format(
                        model=args.model, epoch=epoch, MRR=MRR
                    ),
                )
                break_condition = True
            else:
                mean_reciporical_rank = MRR

        if epoch % 399 == 0:
            torch.save(
                model,
                "Model_{model}_Epoch_Final_MRR_{MRR}.pickle".format(
                    model=args.model, MRR=MRR
                ),
            )

        if break_condition == True:
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="this sets parameters for experiments")
    parser.add_argument(
        "--model", default="DistMult", choices=["DistMult", "Complex", "ConvE"]
    )
    parser.add_argument(
        "--embdim",
        default=100,
        choices=[100, 256, 625, 1024, 1600, 2025],
        type=int,
        help="this is the dimensionality of the embeddings",
    )
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batches", default=5, type=int)
    parser.add_argument("--lr", default=0.01, type=float)

    args = parser.parse_args()
    print(args)

    ConvE_args = ConvE_args()

    data, lookup, ASD_dictionary, BCE_dictionary, _, __ = get_files()
    entities = int(len(lookup) / 2)

    x = shuffle(data)
    test_data = x[:50]

    # split data here into train, eval, test? Make sure eval fits into memory for AUROC?

    if args.model == "DistMult":
        model = DistMult(
            args=ConvE_args,
            embedding_dim=args.embdim,
            num_entities=entities,
            num_relations=4,
        )
    elif args.model == "Complex":
        model = Complex(
            args=ConvE_args,
            embedding_dim=args.embdim,
            num_entities=entities,
            num_relations=4,
        )
    else:
        model = ConvE(
            args=ConvE_args,
            embedding_dim=args.embdim,
            num_entities=entities,
            num_relations=4,
        )

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    main(
        model=model,
        optimiser=optimiser,
        test_data=test_data,
        epochs=args.epochs,
        batches=args.batches,
    )
