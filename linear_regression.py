import tensorflow as tf

x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random.normal([1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

def hypothesis(x):
    return x * W + b

# Define the cost function (loss function)
def cost_function(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Define the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Training function
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = hypothesis(x)
        loss = cost_function(y, y_pred)
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    return loss

# Training loop
for step in range(2001):
    loss = train_step(x_train, y_train)
    if step % 20 == 0:
        print(step, loss.numpy(), W.numpy(), b.numpy())