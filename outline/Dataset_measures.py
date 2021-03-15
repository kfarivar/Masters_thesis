from torch.utils.data import Dataset, DataLoader

# At the moment there doesn't seem to be need for a class

def concentration_measure(data:DataLoader):
    # input: DataLoader
    # 1. calculate knn (preliminary.py)
    # 2. run the proposed algorithm that finds a robust error region (main_infinity.py)
    # return a number