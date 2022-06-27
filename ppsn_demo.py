import argparse
from os import path

import torch
import numpy
import random
import torch.nn.functional as F
from util.ucr_data_loader import UnivariateDataset
from model.shapelet_discovery import ShapeletDiscover
from model.position_shapelet import LearningPShapeletsModel
from util.log import Log

import warnings
warnings.filterwarnings("ignore", category=numpy.VisibleDeprecationWarning)


def initialize(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def basic_smooth_crossentropy(pred, target, smoothing=0.1):
    n_class = pred.size(1)

    true_dist = torch.full_like(pred, fill_value=smoothing / (n_class - 1)).to(pred.device)
    true_dist.scatter_(dim=1, index=target.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return torch.mean(torch.sum(-true_dist * log_prob, dim=1))


def main(args):
    device = torch.device(args.device)
    initialize(seed=42)

    print("Dataset: %s" % args.dataset_name)
    dataset_path = path.join(*['dataset','UCRArchive_2018', args.dataset_name])

    train_file_path = path.join(dataset_path, '{}_TRAIN.tsv'.format(args.dataset_name))
    test_file_path = path.join(dataset_path, '{}_TEST.tsv'.format(args.dataset_name))

    training_set_loader = UnivariateDataset(train_file_path, batch_size=args.batch_size, is_train=1)
    testing_set_loader = UnivariateDataset(test_file_path, batch_size=args.batch_size, is_train=0)

    len_of_ts = len(training_set_loader.data[0])
    num_classes = len(numpy.unique(training_set_loader.labels))

    # Change batch_size based on number of instance
    if training_set_loader.number_of_instance < 100:
        args.batch_size = 16
    elif training_set_loader.number_of_instance < 200:
        args.batch_size = 32
    elif training_set_loader.number_of_instance < 400:
        args.batch_size = 64
    elif training_set_loader.number_of_instance < 800:
        args.batch_size = 128
    else:
        args.batch_size = 256

    training_set_loader.update_batch_size(args.batch_size)
    testing_set_loader.update_batch_size(2048)

    shapelet_discovery = ShapeletDiscover(window_size=args.window_size, num_pip=args.num_pip, processes=args.processes)
    print("Extracting shapelet candidate!")
    shapelet_discovery.extract_candidate(train_data=training_set_loader.data)
    print("Shapelet discovery for window_size = %s" % args.window_size)
    shapelet_discovery.discovery(train_data=training_set_loader.data, train_labels=training_set_loader.labels)
    shapelets_info = shapelet_discovery.get_shapelet_info(number_of_shapelet=args.num_shapelet)

    shapelets = []
    for si in shapelets_info:
        sc = training_set_loader.data[int(si[0]),int(si[1]):int(si[2])]
        shapelets.append(sc)

    model = LearningPShapeletsModel(shapelets_info=shapelets_info, shapelets=shapelets,
                                    len_ts=len_of_ts, num_classes=num_classes, sge=args.sge,
                                    window_size=args.window_size, bounding_norm=args.bounding_norm).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
    log = Log(log_each=3)
    for epoch in range(args.epochs+1):
        model.train()
        log.train(len_dataset=len(training_set_loader))

        for b in range(len(training_set_loader)):
            batch = training_set_loader[b]
            inputs, targets = (b.to(device) for b in batch)

            optimizer.zero_grad()
            predictions = model(inputs,epoch)
            loss = basic_smooth_crossentropy(predictions, targets, smoothing=args.smoothing)
            loss.mean().backward()
            optimizer.step()

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), args.lr)

        if epoch % args.sep == 0:
            model.eval()
            log.eval(len_dataset=len(testing_set_loader))

            with torch.no_grad():
                for b in range(len(testing_set_loader)):
                    batch = testing_set_loader[b]
                    inputs, targets = (b.to(device) for b in batch)
                    predictions = model(inputs,epoch)

                    loss = basic_smooth_crossentropy(predictions, targets, smoothing=args.smoothing)
                    correct = torch.argmax(predictions, 1) == targets
                    log(model, loss.cpu(), correct.cpu())

    log.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Optimizer parameters
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 5e-4)')

    # General parametersƯ
    parser.add_argument("--dataset_name", default="ECGFiveDays", type=str, help="dataset name")
    parser.add_argument("--num_shapelet", default=0.2, type=float, help="number of shapelets")
    parser.add_argument("--window_size", default=10, type=float, help="window size")
    parser.add_argument("--num_pip", default=0.3, type=float, help="number of pips")
    parser.add_argument("--sge", default=1, type=int, help="stop-gradient epochs")
    parser.add_argument("--processes", default=10, type=int, help="number of processes for extracting shapelets")
    parser.add_argument("--bounding_norm", default=100, type=int, help="Fixed at 100")
    parser.add_argument("--max_acc", default=0.0, type=float, help="Fixed at 0")

    parser.add_argument("--batch_size", default=16, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=1000, type=int, help="Total number of epochs.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    parser.add_argument('--device', type=str, default="cuda:0", help='Device for training model')
    parser.add_argument("--sep", default=1, type=int, help="Number of CPU threads for dataloaders.")

    args = parser.parse_args()
    print(args)
    main(args)



