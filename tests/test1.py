import argparse
import sys
sys.path.append('FastAA')
from train import train_baseline, train_fastAA
from config import device

def main():

    K = 5
    N = 1
    T = 5
    B = 50
    epochs = 200
    dataset_name = 'Adiac'
    batch_size = 39
    runs = 5
    aug = False

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

    if aug:
        print("Training FastAA on the {} dataset ...".format(dataset_name))
        print("\n#################################################\n")
        train_fastAA(dataset_name, epochs, batch_size, K, N, T, B, runs)
        print("\n#################################################\n")
        print("FastAA training completed")
        
    else:
        print("Training the baseline model on the {} dataset ...".format(dataset_name))
        print("\n#################################################\n")
        train_baseline(dataset_name, epochs, batch_size, runs)
        print("\n#################################################\n")
        print("Baseline training completed")

if __name__ == "__main__":
    main()