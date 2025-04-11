import math
from collections import Counter

def calc_entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((count/total) * math.log2(count/total) for count in counts.values())

def split_dataset(dataset, value, feature):
    left, right = [], []
    for row in dataset:
        if row[feature] < value:
            left.append(row)
        else:
            right.append(row)

    return left, right

class Node:
    def __init__(self, threshold = None, value = None, feature = None, left = None, right = None):
        self.value = value
        self.feature = feature
        self.left = left
        self.right = right
        self.threshold = threshold


def optimal_div(dataset):
    labels = [row[-1] for row in dataset]
    base_entropy = calc_entropy(labels)
    max_gain = 0
    max_value = None
    max_feature = None

    for feature in range(len(dataset[0]) - 1):
        values = [row[feature] for row in dataset]
        for value in values:
            left, right = split_dataset(dataset, value, feature)
            if not left or not right:
                continue
            gain = base_entropy - len(left)/len(dataset) * calc_entropy([row[-1] for row in left]) - len(right)/len(dataset) * calc_entropy([row[-1] for row in right])
            if gain > max_gain:
                max_gain, max_value, max_feature = gain, value, feature

    return max_gain, max_value, max_feature

def build_tree(dataset, min_samples = 2, depth = 0):
    labels = [row[-1] for row in dataset]
    num_labels = len(labels)
    
    if (labels == [dataset[0][-1] for _ in dataset]):
        return Node(value = labels[0])

    if (num_labels < min_samples):
        majority_class = Counter(labels).most_common(1)[0][0]
        return Node(value=majority_class)
    
    gain, value, feature = optimal_div(dataset)
    if gain == 0:
        majority_class = Counter(labels).most_common(1)[0][0]
        return Node(value=majority_class)
    
    left_data, right_data = split_dataset(dataset, value, feature)
    left = build_tree(left_data, min_samples, depth + 1)
    right = build_tree(right_data, min_samples, depth + 1)


    return Node(threshold = value, feature = feature, left = left, right = right)

def predict(tree, sample):
    if tree.value is not None:
        return tree.value
    
    if sample[tree.feature] < tree.threshold:
        return predict(tree.left, sample)
    else:
        return predict(tree.right, sample)


dataset = [
    [2.7, 2.5, 0],
    [1.3, 1.9, 0],
    [3.1, 3.0, 1],
    [2.0, 1.8, 0],
    [3.0, 2.9, 1],
    [7.6, 2.7, 1]
]

tree = build_tree(dataset)
test_sample = [3.0, 3.0]
print("Predicted class for sample", test_sample, "is", predict(tree, test_sample))