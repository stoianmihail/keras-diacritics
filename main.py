# Instructions:
# Training: you should have a text which is written with diacritics.
# Within the implementation, a new text without diacritcs is created, which will be the baselin for training.
# For that: python main.py train --train dataset/train.txt --test dataset/validation_data.txt --epochs 200

# Transform into diacritcs: python main.py serve --weights output.hf --test dataset/test.txt
# We changed a bit the structure, but --test refers both to the validation_data and the text to be analyzed

import argparse
import os
from ann import NeuralNetwork
from utils import single_class_accuracy
from serve import serve
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("action", help="Test or Train", type=str)

parser.add_argument(
    "--weights", help="A filename where to save our weights", default="output.hf", type=str)

parser.add_argument(
    "--epochs", help="The number of epochs to train the neural network for.", default=100, type=int)

parser.add_argument(
    "--timeseries", help="Size of a single timeseries", default=30, type=int)

parser.add_argument(
    "--lstmSize", help="The size of our first LSTM layer", default=128, type=int)

parser.add_argument(
    "--batchSize", help="Batch size count used for training data.", default=1024, type=int)

parser.add_argument("--dropout", help="Percent Dropout",
                    default=0.25, type=float)

parser.add_argument("--train", help="Train dataset", default='dataset/train', type=str)
parser.add_argument("--test", help="Test dataset", default='dataset/test', type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    ann = NeuralNetwork(
        tsSize=args.timeseries,
        lstmSize=args.lstmSize,
        dropout=args.dropout,
    )

    if os.path.isfile(args.weights):
        ann.model.load_weights(args.weights)

    if args.action == "train":
        ann.fit(args.weights, args.train, args.test,
                epochs=args.epochs, batch_size=args.batchSize)

    if args.action == "serve":
        serve(ann, args.test)

    if args.action == "cli":
        while 1:
            raw = input('Sent > ')
            print(ann.predict(raw))
