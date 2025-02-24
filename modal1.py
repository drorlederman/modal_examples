import modal
#import tensorflow as tf

# Enable GPU usage
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU acceleration enabled")
else:
    print("No GPU found, running on CPU")

app = modal.App(name="cnn-dog-cat-classifier")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "numpy",
    "matplotlib",
    "tensorflow-gpu",
    "pillow"
)

volume = modal.Volume.from_name("cats_vs_dogs")
MODEL_DIR = "/data"
DATA_DIR = "/data/dataset"  # Ensure dataset is stored here in Modal Volume

@app.function(
    volumes={MODEL_DIR: volume},
    image=image,
    gpu=True  # Enable GPU support in Modal
)
def train_model(target_size=(150, 150), batch_size=32, epochs=10):
    """
    Train a CNN model for dog vs cat classification.

    Args:
        target_size (tuple): The target size for the images.
        batch_size (int): Number of images per batch.
        epochs (int): Number of training epochs.
    """
    import os
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    import zipfile

    print(f"Checking dataset at: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        print("Available directories:", os.listdir("/"))
        raise ValueError(f"Data directory '{DATA_DIR}' does not exist!")

    # Extract dataset if it's in a ZIP file
    zip_path = os.path.join(DATA_DIR, "cats_vs_dogs.zip")
    if os.path.exists(zip_path):
        print(f"Extracting dataset from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Dataset extracted successfully!")

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.6),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    with tf.device('/GPU:0'):
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=val_generator,
            validation_steps=val_generator.samples // val_generator.batch_size,
            epochs=epochs
        )

    model.save(os.path.join(MODEL_DIR, 'cnn_dog_cat_classifier.h5'))
    print("Model training completed and saved.")

# Example usage:
# train_model(target_size=(150, 150), batch_size=32, epochs=15)
