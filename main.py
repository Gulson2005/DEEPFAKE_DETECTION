import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB4
import os
import numpy as np
from datetime import datetime
from sklearn.utils import class_weight

# Enable mixed precision but keep final layer in float32
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Dataset paths
base_dir = '/kaggle/input/celeb-df-v2-zip/Celeb_V2'
train_dir = os.path.join(base_dir, 'Train')
val_dir = os.path.join(base_dir, 'Val')
test_dir = os.path.join(base_dir, 'Test')

# Image size for EfficientNetB4
img_height, img_width = 224, 224  # Changed to appropriate size for EfficientNetB4
batch_size = 16  # Reduced batch size due to larger model

# Check class distribution
def check_distribution():
    real_train = len(os.listdir(os.path.join(train_dir, 'real')))
    fake_train = len(os.listdir(os.path.join(train_dir, 'fake')))
    print(f"Training - Real: {real_train}, Fake: {fake_train}")
    
    real_val = len(os.listdir(os.path.join(val_dir, 'real')))
    fake_val = len(os.listdir(os.path.join(val_dir, 'fake')))
    print(f"Validation - Real: {real_val}, Fake: {fake_val}")
    
    real_test = len(os.listdir(os.path.join(test_dir, 'real')))
    fake_test = len(os.listdir(os.path.join(test_dir, 'fake')))
    print(f"Test - Real: {real_test}, Fake: {fake_test}")
    
    return real_train, fake_train

real_count, fake_count = check_distribution()

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest',
    validation_split=0.0  # Ensure no validation split here as we already have separate validation set
)

val_test_datagen = ImageDataGenerator(rescale=1.0/255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Calculate class weights
total = real_count + fake_count
weight_for_0 = (1 / real_count) * (total / 2.0)  # Real is usually class 0
weight_for_1 = (1 / fake_count) * (total / 2.0)  # Fake is usually class 1
class_weights = {0: weight_for_0, 1: weight_for_1}
print(f"Class weights: {class_weights}")

# Build EfficientNetB4 model with a proper head for binary classification
def build_efficient_net_model():
    # Load EfficientNetB4 pre-trained on ImageNet without top layers
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=(img_height, img_width, 3)
    )
    
    # Freeze the base model layers initially for better transfer learning
    for layer in base_model.layers:
        layer.trainable = False
    
    # Build the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        
        # Add custom classifier head with proper regularization
        layers.Dense(512, activation='relu', 
                     kernel_regularizer=regularizers.l2(1e-5)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-5)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    
    return model, base_model

# Create model
model, base_model = build_efficient_net_model()
model.summary()

# Use a lower learning rate for transfer learning
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Compile model with comprehensive metrics
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.FalseNegatives(name='fn')
    ]
)

# Two-phase training approach
# Phase 1: Train just the top layers
print("Phase 1: Training only the top layers (base model frozen)")

callbacks_phase1 = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    ),
    ModelCheckpoint(
        'best_model_phase1.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        save_weights_only=False,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        mode='min',
        verbose=1
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=f"logs/phase1_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        histogram_freq=1
    )
]

# Phase 1 training
history_phase1 = model.fit(
    train_generator,
    epochs=10,  # Short initial training
    validation_data=val_generator,
    callbacks=callbacks_phase1,
    class_weight=class_weights,
    verbose=1
)

# Phase 2: Fine-tune the model by unfreezing some layers of the base model
print("\nPhase 2: Fine-tuning model by unfreezing some layers")

# Unfreeze the last 30 layers of the base model
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile with a lower learning rate for fine-tuning
optimizer_phase2 = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(
    optimizer=optimizer_phase2,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.FalseNegatives(name='fn')
    ]
)

callbacks_phase2 = [
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        mode='min',
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0005
    ),
    ModelCheckpoint(
        'best_model_phase2.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        save_weights_only=False,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,
        min_lr=1e-8,
        mode='min',
        verbose=1
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=f"logs/phase2_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        histogram_freq=1
    )
]

# Phase 2 training
history_phase2 = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=callbacks_phase2,
    class_weight=class_weights,
    verbose=1
)

# Combine histories for plotting
history = {}
for key in history_phase1.history:
    if key in history_phase2.history:
        history[key] = history_phase1.history[key] + history_phase2.history[key]
    else:
        history[key] = history_phase1.history[key]

# Evaluate on test data
results = model.evaluate(test_generator, verbose=1)
metric_names = ['loss', 'accuracy', 'auc', 'precision', 'recall', 'tp', 'tn', 'fp', 'fn']

print("\nTest Results:")
for name, value in zip(metric_names, results):
    print(f"{name}: {value}")

# Confusion matrix analysis
y_pred = model.predict(test_generator)
y_pred_classes = (y_pred > 0.5).astype(int)
y_true = test_generator.classes

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_true, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

# Prediction distribution analysis
plt.figure(figsize=(10, 6))
plt.hist(y_pred, bins=50)
plt.title('Prediction Distribution')
plt.xlabel('Prediction Value')
plt.ylabel('Count')
plt.axvline(0.5, color='red', linestyle='--')
plt.savefig('prediction_distribution.png')
plt.show()

# Enhanced training history visualization
plt.figure(figsize=(18, 12))

# Accuracy
plt.subplot(2, 2, 1)
plt.plot(history['accuracy'], label='Train', linewidth=2)
plt.plot(history['val_accuracy'], label='Validation', linewidth=2)
plt.title('Model Accuracy', fontsize=14, pad=20)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim([0, 1])

# Loss
plt.subplot(2, 2, 2)
plt.plot(history['loss'], label='Train', linewidth=2)
plt.plot(history['val_loss'], label='Validation', linewidth=2)
plt.title('Model Loss', fontsize=14, pad=20)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# AUC
plt.subplot(2, 2, 3)
plt.plot(history['auc'], label='Train', linewidth=2)
plt.plot(history['val_auc'], label='Validation', linewidth=2)
plt.title('Model AUC', fontsize=14, pad=20)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim([0.5, 1])

# Precision-Recall
plt.subplot(2, 2, 4)
plt.plot(history['precision'], label='Precision', linewidth=2)
plt.plot(history['recall'], label='Recall', linewidth=2)
plt.title('Precision and Recall', fontsize=14, pad=20)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim([0, 1])

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the final model
model.save('efficient_net_final_model.keras')
print("Model saved successfully.")

# Final model prediction test
print("\nRunning final prediction test to verify model output distribution...")
sample_preds = model.predict(test_generator)
real_preds = sample_preds[test_generator.classes == 0]
fake_preds = sample_preds[test_generator.classes == 1]

print(f"Real images prediction mean: {np.mean(real_preds):.4f}, std: {np.std(real_preds):.4f}")
print(f"Fake images prediction mean: {np.mean(fake_preds):.4f}, std: {np.std(fake_preds):.4f}")

# Plot prediction distributions by class
plt.figure(figsize=(12, 6))
plt.hist(real_preds, bins=30, alpha=0.5, label='Real', color='green')
plt.hist(fake_preds, bins=30, alpha=0.5, label='Fake', color='red')
plt.axvline(0.5, color='black', linestyle='--')
plt.title('Prediction Distribution by Class')
plt.xlabel('Prediction Value')
plt.ylabel('Count')
plt.legend()
plt.savefig('prediction_distribution_by_class.png')
plt.show()