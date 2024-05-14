import argparse
from train import train_baseline, train_fastAA
from config import device


def parse_arguments():
    '''
    Parses arguments for Fast AutoAugment
    '''
    parser = argparse.ArgumentParser(description="Argument Parser for Fast AutoAugment")    
    parser.add_argument(                                                                        # Dataset name
        "--dataset", type=str, default="None", help="Name of the dataset (default: None)"
    )
    parser.add_argument(                                                                        # Number of folds
        "--K", type=int, default=5, help="Number of folds for Fast AutoAugment (default: 5)"
    )
    parser.add_argument(                                                                        # Number of best policies
        "--N", type=int, default=1, help="Number of best policies (default: 1)"
    )
    parser.add_argument(                                                                        # Number of iterations
        "--T", type=int, default=5, help="Number of iterations (default: 5)"
    )
    parser.add_argument(                                                                        # Number of samples per fold    
        "--B", type=int, default=50, help="Number of samples per fold (default: 50)"
    )
    parser.add_argument(                                                                        # Number of epochs for model training
        "--epochs", type=int, default=200, help="Number of epochs (default: 200)"
    )
    parser.add_argument(                                                                        # Batch size for model training
        "--batch_size", type=int, default=8, help="Batch size (default: 8)"
    )
    parser.add_argument(                                                                        # Compare FastAA results with baseline results                          
        "--compare", action="store_true", help="Compare FastAA results with baseline results"
    )
    parser.add_argument(                                                                        # Number of runs for standard deviation calculation
        "--runs", type=int, default=1, help="Number of runs for standard deviation calculation (default: 1)"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    K = args.K
    N = args.N
    T = args.T
    B = args.B
    epochs = args.epochs
    comparison = args.compare
    dataset_name = args.dataset
    batch_size = args.batch_size
    runs = args.runs

    print("\n#################################################\n")
    print(
        "Performing Fast AutoAugment on the {} dataset".format(dataset_name),
        "computing on device {}".format(device),
    )
    print(
        "Using {}-fold bagging (K)".format(K),
        "to find the N={} best policies".format(N),
        "over T={} iterations".format(T),
        "with B={} samples per fold".format(B),
    )
    print("\n#################################################\n")

    if comparison:
        print("Comparing FastAA results with baseline results")
        print("Training the baseline model on the {} dataset ...".format(dataset_name))
        print("\n#################################################\n")
        train_baseline(dataset_name, epochs, batch_size, runs)
        print("\n#################################################\n")
        print("Baseline training completed")
        print("Training FastAA on the {} dataset ...".format(dataset_name))
        print("\n#################################################\n")
        train_fastAA(dataset_name, epochs, batch_size, K, N, T, B, runs)
        print("\n#################################################\n")
        print("FastAA training completed")
        
    else:
        print("Training FastAA on the {} dataset ...".format(dataset_name))
        print("\n#################################################\n")
        train_fastAA(dataset_name, epochs, batch_size, K, N, T, B, runs)
        print("\n#################################################\n")
        print("FastAA training completed")

if __name__ == "__main__":
    main()