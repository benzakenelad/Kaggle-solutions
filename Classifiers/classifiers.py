import utils
import math as m


class classification_algorithm:
    '''
    abstract class that unified all the shared data
    '''

    def __init__(self):
        self.X = None
        self.y = None
        self.classes = None
        self.classes_label_counter = None
        self.attributes_len = 0
        self.X_size = 0

    def predict(self, X_test):
        '''
        predicting set of samples
        :param X_test: samples
        :return: classifications list
        '''
        predictions = []
        for x in X_test:
            predictions.append(self._predict_single_sample(x))
        return predictions

    def _predict_single_sample(self, sample):
        '''
        abstract method
        :param sample: predicting label for sample
        :return: the classification
        '''
        raise NotImplementedError()


class KNN(classification_algorithm):
    '''
    K-nearest neighbor classifier
    '''

    def __init__(self, k=5):
        classification_algorithm.__init__(self)
        self.k = k
        self.classes_map = None

    def fit(self, X, y):
        '''
        fit a KNN model to the data
        :param X: data
        :param y: label
        '''
        self.X = list(X)
        self.y = list(y)
        self.classes, self.classes_labels_counter = utils.extract_unique_labels_and_labels_counter(self.y)
        self.attributes_len = len(X[0])
        self.classes_map = {k: v for v, k in enumerate(self.classes)}

    def _predict_single_sample(self, sample):
        '''
        :param sample: predicting label for sample
        :return: the classification
        '''
        if self.X is None:
            raise Exception('KNN data was not initialize with fit method.')
        if self.attributes_len != len(sample):
            raise ValueError('sample len is not as DATA\'s len')

        hamming_distances = []
        for x in self.X:
            hamming_distances.append(self.__distance(sample, x))
        argsort = utils.argsort(hamming_distances)[:self.k]
        predictions = [self.y[i] for i in argsort]
        predictions_class_counter = [0] * len(self.classes)

        for p in predictions:
            predictions_class_counter[self.classes_map.get(p)] += 1

        return self.classes[utils.argmax(predictions_class_counter)]

    def __distance(self, sample_a, sample_b):
        '''
        calculating hamming distance between sample_a and sample_b
        :param sample_a:
        :param sample_b:
        :return: the distance
        '''
        hamming_distance = 0
        for i in range(self.attributes_len):
            if sample_a[i] != sample_b[i]:
                hamming_distance += 1
        return hamming_distance


class NaiveBayes(classification_algorithm):
    '''
    naive bayes classifer
    '''

    def __init__(self):
        classification_algorithm.__init__(self)
        self.classes_amount = 0
        self.smoothing_k = None

    def fit(self, X, y):
        '''
        fitting the data naive bayes model
        :param X: data
        :param y: labels
        '''
        self.X = list(X)
        self.y = list(y)
        self.attributes_len = len(X[0])
        self.X_size = len(self.X)
        self.classes, self.classes_labels_counter = utils.extract_unique_labels_and_labels_counter(self.y)
        self.classes_amount = len(self.classes)
        self.smoothing_k = [0] * self.attributes_len
        for i in range(self.attributes_len):
            sub_attribute, temp = utils.extract_unique_labels_and_labels_counter([x[i] for x in self.X])
            self.smoothing_k[i] = len(sub_attribute)

    def _predict_single_sample(self, sample):
        '''
        :param sample: predicting label for sample
        :return: the classification
        '''
        attribute_instances_counter_by_label = [[0] * self.classes_amount for i in range(self.attributes_len)]
        for i, X_sample in enumerate(self.X):
            for j, cls in enumerate(self.classes):
                if self.y[i] == cls:
                    for k in range(self.attributes_len):
                        if sample[k] == X_sample[k]:
                            attribute_instances_counter_by_label[k][j] += 1
                    break

        for i in range(self.attributes_len):
            for j in range(self.classes_amount):
                attribute_instances_counter_by_label[i][j] = (attribute_instances_counter_by_label[
                                                                  i][j] + 1) / (
                                                                 self.classes_labels_counter[j] + self.smoothing_k[i])

        classes_prior = [x / self.X_size for x in self.classes_labels_counter]

        for i in range(len(self.classes)):
            for j in range(self.attributes_len):
                classes_prior[i] *= attribute_instances_counter_by_label[j][i]

        return self.classes[utils.argmax(classes_prior)]


class DecisionTree(classification_algorithm):
    '''
    decision tree classifier
    '''

    def __init__(self):
        classification_algorithm.__init__(self)
        self.attributes = None
        self.prior = None
        self.tree_root = None
        self.attributes_map = None

    def fit(self, X, y, attributes):
        '''
        fitting the data decision tree model
        :param X: data
        :param y: labels
        :param attributes: attributes list (for tree printing)
        '''
        self.X = list(X)
        self.y = list(y)
        self.X_size = len(self.X)
        self.attributes = attributes
        self.attributes_len = len(self.X[0])
        self.classes, self.classes_labels_counter = utils.extract_unique_labels_and_labels_counter(self.y)
        self.prior = self.classes[utils.argmax(self.classes_labels_counter)]
        self.tree_root = self.__DTL(self.X, self.y, self.attributes, self.prior)
        self.attributes_map = {att: i for i, att in enumerate(attributes)}

    def _predict_single_sample(self, sample):
        '''
        :param sample: predicting label for sample
        :return: the classification
        '''
        tree = self.tree_root
        while True:
            if tree.classification is not '':
                return tree.classification
            current_attribute_index = self.attributes_map.get(tree.subtrees[0].attribute)
            for sub_tree in tree.subtrees:
                if sub_tree.value == sample[current_attribute_index]:
                    tree = sub_tree
                    break
            else:
                return self.prior

    def __information_gain(self, attribute_data, labels):
        '''
        calculating the information gain using entropy
        :param attribute_data:
        :param labels:
        :return:
        '''
        if len(attribute_data) == 0 or len(attribute_data) != len(labels):
            raise Exception('illegal x,y size')
        n = len(attribute_data)
        sub_attribute_values, attributes_counter = utils.extract_unique_labels_and_labels_counter(attribute_data)
        labels_values, labels_counter = utils.extract_unique_labels_and_labels_counter(labels)
        sub_attributes_classes_counter = [[0] * len(labels_values) for i in range(len(sub_attribute_values))]
        sub_attribute_map = {k: v for v, k in enumerate(sub_attribute_values)}
        classes_map = {k: v for v, k in enumerate(labels_values)}

        for i, x in enumerate(attribute_data):
            sub_attributes_classes_counter[sub_attribute_map.get(x)][classes_map.get(labels[i])] += 1

        sub_attributes_entropy = [0] * len(sub_attribute_values)
        conditioned_entropy = 0
        for i in range(len(sub_attribute_values)):
            sub_attributes_entropy[i] = self.__entropy(sub_attributes_classes_counter[i])
            conditioned_entropy += (sum(sub_attributes_classes_counter[i]) / n) * sub_attributes_entropy[i]

        labels_entropy = self.__entropy(labels_counter)
        return labels_entropy - conditioned_entropy

    def __entropy(self, x):
        '''
        calculating vector x entropy
        :param x:
        :return:
        '''
        n = sum(x)
        return -sum([(a / n) * m.log(a / n, 2) for a in x if a != 0])

    def __DTL(self, data, labels, attributes, default):
        '''
        DTL method recursively building the decision tree nodes
        :param data:
        :param labels: labels list
        :param attributes: attributes list
        :param default: default classification
        :return:
        '''
        tree = Tree()

        if len(data) == 0:
            tree.classification = default
            return tree
        if len(set(labels)) == 1:
            tree.classification = labels[0]
            return tree
        if len(attributes) == 0:
            label_values, label_values_counter = utils.extract_unique_labels_and_labels_counter(labels)
            tree.classification = label_values[utils.argmax(label_values_counter)]
            return tree

        best_att_value, best_att_index = self.__choose_attribute(data, labels, attributes)
        best_att_sub_values, best_att_sub_values_counter = utils.extract_unique_labels_and_labels_counter(
            [x[best_att_index] for x in data])

        for i, sub_att in enumerate(best_att_sub_values):
            new_data = []
            new_labels = []

            for j, sample in enumerate(data):
                if sample[best_att_index] == sub_att:
                    new_data.append(data[j])
                    new_labels.append(labels[j])

            new_data = utils.remove_column(new_data, best_att_index)
            new_attributes = list(attributes)
            new_attributes = [new_attributes[i] for i in range(len(new_attributes)) if i != best_att_index]
            classes, classes_labels_counter = utils.extract_unique_labels_and_labels_counter(labels)
            default = classes[utils.argmax(classes_labels_counter)]
            sub_tree = self.__DTL(new_data, new_labels, new_attributes, default)
            sub_tree.value = sub_att
            sub_tree.attribute = best_att_value

            tree.subtrees.append(sub_tree)
        return tree

    def __choose_attribute(self, data, labels, attributes):
        '''
        select the attribute with the maximum information gain
        :param data:
        :param labels:
        :param attributes:
        :return: the attribute value and the attribute index where the information gain is maximized
        '''
        info_gain = [0] * len(attributes)
        for i, att in enumerate(attributes):
            info_gain[i] = self.__information_gain([x[i] for x in data], labels)
        return attributes[utils.argmax(info_gain)], utils.argmax(info_gain)

    def print_tree(self, output=None):
        '''
        print the tree to output file
        :param output:
        :return:
        '''
        if self.tree_root is None:
            raise Exception('tree is not exist, please run fit method')

        self.tree_root.subtrees.sort(key=lambda x: x.value)
        for sub_tree in self.tree_root.subtrees:
            self.__sub_tree_print(sub_tree, 0, output)

    def __sub_tree_print(self, tree, tabs_amount, output):
        '''
        print the subtree to output file
        :param tree:
        :param tabs_amount:
        :param output:
        :return:
        '''
        line = ''
        for i in range(tabs_amount): line += '\t'
        classifications = []
        for sub_tree in tree.subtrees:
            if sub_tree.classification == '':
                break
            classifications.append(sub_tree.classification)
        else:
            classifications = set(classifications)
            if len(classifications) == 1:
                if tabs_amount > 0:
                    output.write('{}|{}={}:{}\n'.format(line, tree.attribute, tree.value, classifications.pop()))
                else:
                    output.write('{}{}={}:{}\n'.format(line, tree.attribute, tree.value, classifications.pop()))
                return
        if tree.classification is '':
            if tabs_amount > 0:
                line = '{}|{}={}\n'.format(line, tree.attribute, tree.value)
            else:
                line = '{}{}={}\n'.format(line, tree.attribute, tree.value)
        else:
            if tabs_amount > 0:
                line = '{}|{}={}:{}\n'.format(line, tree.attribute, tree.value, tree.classification)
            else:
                line = '{}{}={}:{}\n'.format(line, tree.attribute, tree.value, tree.classification)
        if output is None:
            print(line)
        else:
            output.write(line)
        tree.subtrees.sort(key=lambda x: x.value)
        for sub_tree in tree.subtrees:
            self.__sub_tree_print(sub_tree, tabs_amount + 1, output)


class Tree:
    '''
    tree node class
    '''

    def __init__(self):
        self.subtrees = []
        self.value = ''
        self.attribute = ''
        self.classification = ''
