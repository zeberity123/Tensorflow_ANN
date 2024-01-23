from sklearn import preprocessing
import numpy as np

def encode(cities):
    d_cities = {i : city for city, i in enumerate(dict.fromkeys(cities))}
    n_cities = [d_cities[i] for i in cities]
    return n_cities

def label_encoder(cities):
    enc = preprocessing.LabelEncoder()
    enc.fit(cities)
    result = enc.transform(cities)

    # print(enc.inverse_transform(result))
    # print(enc.classes_)
    # print(enc.classes_[result])

    return result


def label_binarizer(cities):
    bin = preprocessing.LabelBinarizer()
    bin.fit(cities)
    result = bin.transform(cities)

    print([list(i).index(1) for i in result])
    print([np.max(i) for i in result])
    print([np.argmax(i) for i in result])
    print(np.argmax(result))
    print(np.argmax(result, axis=1))
    
    # print(result)

cities = ['bali', 'paris', 'london', 'bali', 'london']
print(encode(cities))

print(label_encoder(cities))
label_binarizer(cities)