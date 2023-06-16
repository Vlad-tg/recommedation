import pa as pa
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


movies_data = pd.read_csv('db_movies.csv')
users_data = pd.read_csv('db_user.csv')
y = pd.read_csv('db_ratings.csv')

y = y.to_numpy()

num_user_features = users_data.shape[1] - 2  # remove userid, rating count and ave rating during training
num_item_features = movies_data.shape[1] - 1

uvs = 3  # user genre vector start
ivs = 3  # item genre vector start
u_s = 2  # start of columns to use in training, user
i_s = 1  # start of columns to use in training, items

scalerItem = StandardScaler()
scalerItem.fit(movies_data)
item_train = scalerItem.transform(movies_data)

scalerUser = StandardScaler()
scalerUser.fit(users_data)
user_train = scalerUser.transform(users_data)


scalerTarget = MinMaxScaler((-1, 1))
scalerTarget.fit(y.reshape(-1, 1))
y = scalerTarget.transform(y.reshape(-1, 1))

X_train, X_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)

print(X_train.shape[0])

y_train, y_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)

y, y_t = train_test_split(y, train_size=0.80, shuffle=True, random_state=1)

num_outputs = 32

tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs, activation='linear'),
])

item_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs, activation='linear'),

])

input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

output = tf.keras.layers.Dot(axes=1)([vu, vm])

model = tf.keras.Model([input_user, input_item], output)

model.summary()

tf.random.set_seed(1)
cost_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,
              loss=cost_fn)

tf.random.set_seed(1)
model.fit([X_train[:, u_s:], y_train[:, i_s:]], y, epochs=30)

model.evaluate([X_test[:, u_s:], y_test[:, i_s:]], y_t)
model.save('model_recommended_system.h5')