import math
import random
from collections import Counter

# CONST
ROWS_IN_BOOTSTRAPED_DATASET = 4
NUMBER_OF_TREES = 10
USE_EXCLUSIVE_FETURES = True

def calc_entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((count/total) * math.log2(count/total) for count in counts.values())

class Node:
    def __init__(self, value = None, feature = None, porog = None, left = None, right = None):
        self.value = value
        self.feature = feature
        self.porog = porog
        self.left = left
        self.right = right

# class Tree:
#     def __init__(self):
        

def split_dataset(dataset, value, feature):
    left, right = [], []
    for row in dataset:
        if row[feature] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def best_split(dataset, candidate_features):
    base_entropy = calc_entropy([row[-1] for row in dataset])

    max_gain, max_value, max_feature = 0, None, None

    for feature in candidate_features:
        for row in dataset:
            value = row[feature]
            left, right = split_dataset(dataset, value, feature)
            if not left or not right:
                continue
            c_gain = base_entropy - len(left)/len(dataset) * calc_entropy([row[-1] for row in left]) - len(right)/len(dataset) * calc_entropy([row[-1] for row in right])
            if c_gain > max_gain:
                max_gain, max_value, max_feature = c_gain, value, feature

    return max_gain, max_value, max_feature


def build_random_tree(dataset, min_split = 2, num_candidate_features = 2, excl_feature = None):
    labels = [row[-1] for row in dataset]
    if len(set(labels)) == 1:
        return Node(value=labels[0])
    
    if len(labels) < min_split:
        majorclass = Counter(labels).most_common(1)[0][0]
        return Node(value=majorclass)
    
    if excl_feature is None: excl_feature = []

    if len(excl_feature) + num_candidate_features > len(dataset[0])-1:
        majorclass = Counter(labels).most_common(1)[0][0]
        return Node(value=majorclass)
    
    candidate_features = []
    while len(candidate_features) != num_candidate_features:
        k = random.randint(0, len(dataset[0])-2)
        if k not in excl_feature:
            candidate_features.append(k)
    
    gain, value, feature = best_split(dataset, candidate_features)
    excl_feature.append(feature)

    if gain == 0:
        majorclass = Counter(labels).most_common(1)[0][0]
        return Node(value=majorclass)

    left, right = split_dataset(dataset=dataset, value=value, feature=feature)
    if USE_EXCLUSIVE_FETURES:
        left = build_random_tree(left, num_candidate_features=num_candidate_features, excl_feature=excl_feature)
        right = build_random_tree(right, num_candidate_features=num_candidate_features, excl_feature=excl_feature)
    else:
        left = build_random_tree(left, num_candidate_features=num_candidate_features, excl_feature=None)
        right = build_random_tree(right, num_candidate_features=num_candidate_features, excl_feature=None)
    
    return Node(feature=feature, porog=value, left=left, right=right)

def predict(tree, sample):
    if tree.value is not None:
        return tree.value
    if sample[tree.feature] < tree.porog:
        return predict(tree.left, sample)
    return predict(tree.right, sample)

def bootstrap_dataset(dataset, n):
    indices = [random.randint(0, len(dataset) - 1) for _ in range(n)]
    bt_dataset = [dataset[i] for i in indices]
    oob_indices = set(range(len(dataset))) - set(indices)
    oob = [dataset[i] for i in oob_indices]
    return bt_dataset, oob

def generate_random_forest(dataset, n):
    best_accuracy = 0
    best_forest = None
    for num_candidate_features in range(1, len(dataset[0]) - 1):
        forest, correct, total_oob = [], 0, 0
        for _ in range(n):
            bt_dataset, oob = bootstrap_dataset(dataset, ROWS_IN_BOOTSTRAPED_DATASET)
            tree = build_random_tree(bt_dataset, num_candidate_features = num_candidate_features)
            forest.append(tree)
            for sample in oob:
                if (predict(tree, sample[:-1]) == sample[-1]):
                    correct += 1
                total_oob += 1
        current_accuracy = correct / total_oob if total_oob > 0 else 0
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_forest = forest

    return best_forest

def predict_forest(forest, sample):
    predictions = [predict(tree, sample) for tree in forest]
    votes = Counter(predictions)
    majority_class = votes.most_common(1)[0][0]
    return votes, majority_class

dataset = [
    [2.7, 2.5, 0],
    [1.3, 1.9, 0],
    [3.1, 3.0, 1],
    [2.0, 1.8, 0],
    [3.0, 2.9, 1],
    [7.6, 2.7, 1]
]

forest = generate_random_forest(dataset, NUMBER_OF_TREES)
test_sample = [3.0, 3.0]

_, ans = predict_forest (forest, test_sample)
print(ans)