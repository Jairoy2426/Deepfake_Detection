import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Set the parameters
IMG_SIZE = 224  # Size to resize frames
SEQ_LENGTH = 100  # Number of frames per video to use

def load_video(path, max_frames=SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(1, frame_count // max_frames)  # Ensure you get at most max_frames frames
    for i in range(0, frame_count, skip_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # Resize and normalize the frame
        frame = cv2.resize(frame, resize)
        frame = frame[:, :, [2, 1, 0]]  # Convert BGR to RGB
        frames.append(frame)
        if len(frames) == max_frames:
            break
    cap.release()
    return np.array(frames)

def prepare_dataset(dataset_dir, label, seq_length=SEQ_LENGTH):
    X = []
    y = []
    for video_file in os.listdir(dataset_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(dataset_dir, video_file)
            frames = load_video(video_path, max_frames=seq_length)
            if len(frames) == seq_length:  # Only consider videos with enough frames
                X.append(frames)
                y.append(label)
    return np.array(X), np.array(y)

# Load real and fake videos
X_real, y_real = prepare_dataset(r'C:\Users\jairo\Downloads\Celeb Dataset\Celeb-DF\real', label=0)  # 0 for real
X_fake, y_fake = prepare_dataset(r'C:\Users\jairo\Downloads\Celeb Dataset\Celeb-DF\fake', label=1)  # 1 for fake

# Combine the data and shuffle it
X = np.concatenate((X_real, X_fake), axis=0)
y = np.concatenate((y_real, y_fake), axis=0)

# Shuffle the dataset
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=42)


# Split the dataset into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Model
def build_model(seq_length=SEQ_LENGTH, img_size=IMG_SIZE):
    model = models.Sequential()

    # TimeDistributed Conv2D layers for each frame
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'),
                                     input_shape=(seq_length, img_size, img_size, 3)))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Flatten()))

    # Add an LSTM layer
    model.add(layers.LSTM(64, return_sequences=False))
    model.add(layers.Dropout(0.5))

    # Output layer for binary classification (real vs fake)
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build and compile model
model = build_model()
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=30, validation_data=(X_val, y_val))

# Save the model after training
model.save('deepfake_detection_model.h5')

print("Model training complete and saved.")
