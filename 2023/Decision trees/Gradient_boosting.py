import math
from collections import Counter
import random

# CONSTS
MAX_DEPTH = 4
NUM_OF_TREES = 2
LEARNING_RATE = 0.4


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

    if (num_labels < min_samples or depth >= MAX_DEPTH):
        majority_class = find_average(labels)
        return Node(value=majority_class)
    
    gain, value, feature = optimal_div(dataset)
    if gain == 0:
        majority_class = find_average(labels)
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

def calc_loss(observed, predicted):
    return 1/2 * (observed - predicted) ** 2

def calc_diff_loss(observed, predicted):
    return -(observed - predicted)

def find_average(arr):
    ans = sum(arr) / len(arr)
    return ans

def build_grad_boost(dataset):
    labels = [row[-1] for row in dataset]
    average = find_average(labels)
    fs = [[average for _ in range(len(labels))]]
    model = [average]

    for m in range(1, NUM_OF_TREES):
        current_dataset = []
        for i in range(len(labels)):
            # current_dataset.append(dataset[i])
            # current_dataset[i][-1] = -1 * calc_diff_loss(labels[i], fs[m-1][i])
            residual = labels[i] - fs[m-1][i]
            current_dataset.append(dataset[i][:-1] + [residual]) 


        tree = build_tree(current_dataset)
        fs_m = [fs[m-1][i] + LEARNING_RATE * predict(tree, row[:-1]) for row in dataset]
        fs.append(fs_m)

        model.append(tree)
    return model

def predict_from_grad_boost(sample, model):
    ans = model[0]
    for tree in model[1:]:
        ans += LEARNING_RATE * predict(tree, sample)
    return ans



dataset = [
    [1.6, 0, 0, 88],
    [1.6, 1, 1, 76],
    [1.5, 0, 1, 56],
    [1.2, 1, 1, 56]
]

model = build_grad_boost(dataset)
ans = predict_from_grad_boost([1.8, 0, 0], model)
print(ans)
