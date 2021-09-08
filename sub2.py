import numpy as np
from PIL import Image
from sklearn import preprocessing
from sklearn.svm import SVC


# extragere features cu ajutorul unui dictionar de frecventa
class BagOfWords:

    def __init__(self):
        self.vocabulary = dict()
        self.words = list()

    def build_vocabulary(self, data):
       id = 1
       self.vocabulary['UNK'] = 0
       self.words.append('UNK')
       for lista in data:
            for word in lista:
                 if word not in self.words:

                       self.vocabulary[word] = id
                       id += 1
                       self.words.append(word)

    def get_features(self, data):
        features = np.zeros((len(data), len(self.vocabulary)))
        for i in range(len(data)):
            sample = data[i]
            for word in sample:
                if word in self.words:
                    features[i][self.vocabulary[word]] += 1
                else:
                    features[i][0] += 1
        return features


def normalize_data(train_data, test_data, type=None):

    if type is None:
        return train_data, test_data
    if type == 'standard':
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_data)
        return scaler.transform(train_data), scaler.transform(test_data)
    if type == 'minmax':
        return (preprocessing.minmax_scale(train_data, axis=-1),
                preprocessing.minmax_scale(test_data, axis=-1))
    if type == 'l1':
        return (preprocessing.normalize(train_data, norm='l1'),
                preprocessing.normalize(test_data, norm='l1'))
    if type == 'l2':
        return (preprocessing.normalize(train_data, norm='l2'),
                preprocessing.normalize(test_data, norm='l2'))


test_file = open("ML-concurs/test.txt", "r")
train_file = open("ML-concurs/train.txt", "r")
test_output_file = open("sub3_svm.csv", "w")
test_output_file.write("id,label\n")
# validation_fie = open("ML-concurs/validation.txt", "r")

train_images = []
train_labels = []
test_images = []
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

# preprocesare
bow = BagOfWords()
bow.build_vocabulary(train_images)
train_features = bow.get_features(train_images)
test_features = bow.get_features(test_images)
X_train, X_test = normalize_data(train_features, test_features, type='standard')

# for ker in ["linear", "poly", "rbf"]:

# clasificare
model = SVC(kernel='rbf')  # Are deja default C=1
model.fit(X_train, train_labels)
predicted = model.predict(X_test)

# afisare rezultate
for index, image in enumerate(test_images):
    test_output_file.write(test_images_names[index] + "," + str(int(predicted[index])) + "\n")


test_file.close()
train_file.close()
test_output_file.close()