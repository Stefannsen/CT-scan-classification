import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import accuracy_score


class KnnClassifier:

    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def calculate_dists(self, test_image, metric):
        if metric == 'l2':
            pwr = np.power(self.train_images - test_image, 2)
            distances = np.sum(pwr, axis=1)
            return np.sqrt(distances)
        else:
            return np.sum(np.abs(self.train_images - test_image), axis=1)

    def classify_image(self, test_image, num_neighbours=3, metric='l2'):
        dists = self.calculate_dists(test_image, metric)

        indexes = np.argsort(dists)

        neighbours = indexes[:num_neighbours]
        n_labels = []

        for i in neighbours:
            n_labels.append(train_labels[i])

        votes = np.bincount(n_labels)

        return np.argmax(votes)


test_file = open("ML-concurs/test.txt", "r")
train_file = open("ML-concurs/train.txt", "r")
test_output_file = open("sub1_knn3.csv", "w")
test_output_file.write("id,label\n")
# validation_fie = open("ML-concurs/validation.txt", "r")

train_images = []
train_labels = []
test_images = []
test_labels = []
test_images_names = []

for line in train_file.readlines():
    pair = line.split(",")
    image = Image.open("ML-concurs/train/" + pair[0])
    image_array = np.array(image).ravel()
    train_images.append(image_array)
    train_labels.append(int(pair[1]))

for line in test_file.readlines():
    pair = line.split("\n")
    test_images_names.append(pair[0])
    image = Image.open("ML-concurs/test/" + pair[0])
    image_array = np.array(image).ravel()
    test_images.append(image_array)




classifier = KnnClassifier(train_images, train_labels)
predicted = np.zeros(len(test_images))

for index, image in enumerate(test_images):
    predicted[index] = classifier.classify_image(image)
    test_output_file.write(test_images_names[index] + "," + str(int(predicted[index])) + "\n")

test_file.close()
train_file.close()
test_output_file.close()
