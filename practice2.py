import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import keras
from keras.utils import pad_sequences
from keras.preprocessing.text import one_hot
from keras.layers import Dense,Flatten,Embedding
from keras.models import Sequential

data_url = "https://archive.ics.uci.edu/dataset/911/recipe+reviews+and+user+feedback+dataset"
data_dir = r"D:\YouTube Tutorials\Deep Learning\Tensorflow Code Basics Deep Learning\recipe+reviews+and+user+feedback+dataset\Recipe Reviews and User Feedback Dataset.csv"
df = pd.read_csv(data_dir)
df.head()

df['stars'].unique()
df['text'][:10]
x = df['text'][:50]
y = df['stars'][:50]


for i in x:
    print(i)

vocab_size = 50000
encoded_reviews= [one_hot(d,vocab_size) for d in x]
encoded_reviews[1]

max_len = 110
padded_reviews = pad_sequences(encoded_reviews,maxlen=max_len,padding="post")
padded_reviews[16]

Embedded_vector_size = 5
model = Sequential([
    Embedding(vocab_size,Embedded_vector_size,input_length=max_len,name="embedding"),
    Flatten(),
    Dense(1,activation="sigmoid")
])

model.summary()

model.compile(optimizer='adam',loss="categorical_crossentropy",metrics='acc')

model.fit(padded_reviews,y,epochs=100)

# its fucked up