import matplotlib.pyplot as plt
import os
import tensorflow as tf
import splitfolders
import numpy as np


np.random.seed(2022)
tf.random.set_seed(2022)


# code to download dataset from zip and split into train, val, test set
original_dir = 'all_images'
new_dir = 'output'
SEED = 10
saved_model = True
test_mode = True
regularize = False

if not os.path.isdir(new_dir):
    splitfolders.ratio(original_dir, output=new_dir, seed=SEED, ratio=(.8, 0.1, 0.1))


train_dir = os.path.join(new_dir, 'train')
validation_dir = os.path.join(new_dir, 'val')
test_dir = os.path.join(new_dir, 'test')


BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)


class_names = train_dataset.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
#preprocess_input = tf.keras.applications.xception.preprocess_input
#preprocess_input = tf.keras.applications.densenet.preprocess_input

# code to setup pre-trained model

if saved_model:
    model = tf.keras.models.load_model('model')

else:
    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    #base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE,
                                                #include_top=False,
                                                #weights='imagenet')
    #base_model = tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE,
                                                   #include_top=False,
                                                   #weights='imagenet')


    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    base_model.trainable = False

    print(base_model.summary())

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    print(model.summary())

    print(len(model.trainable_variables))

    # code to train feature extraction model

    initial_epochs = 10

    loss0, accuracy0, precision0, recall0 = model.evaluate(validation_dataset)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        validation_data=validation_dataset,
                        callbacks=callback)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # code shows historical training and validation performance

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('feature_extracted.png')

    # code for finetuning model

    base_model.trainable = True


    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at = len(base_model.layers)-50

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable = False

    penalty = 0.01
    regularizer = tf.keras.regularizers.l2(penalty)

    if regularize:
        for layer in base_model.layers[fine_tune_at:]:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)


    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    print(model.summary())

    len(model.trainable_variables)


    fine_tune_epochs = 11
    total_epochs =  initial_epochs + fine_tune_epochs

    start_fine_tune = history.epoch[-1] + 1

    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=start_fine_tune,
                             validation_data=validation_dataset,
                             callbacks=callback)

    # code shows historical training and validation performance

    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([start_fine_tune-1,start_fine_tune-1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('fine_tuned.png')

    model.save('model')



if test_mode:
    # finally we evaluate performance on the test dataset
    loss, accuracy, precision, recall = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)


    external_test_dataset = tf.keras.utils.image_dataset_from_directory('external_test',
                                                               shuffle=True,
                                                               batch_size=BATCH_SIZE,
                                                               image_size=IMG_SIZE)


    ext_loss, ext_accuracy, ext_precision, ext_recall = model.evaluate(external_test_dataset)
    print('External test accuracy :', ext_accuracy)






