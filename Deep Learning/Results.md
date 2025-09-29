A Basic CNN

The selected architecture consists of three convolutional layers with 32, 64, and 128 filters,
each followed by max-pooling to down sample the spatial dimensions. After feature
extraction, the network includes three fully connected layers: one with 512 units, another
with 128 units, and a final output layer with a number of units equal to the number of
classes (defaulting to 10). ReLU activation functions are applied after each convolutional and
fully connected layer, with the model outputting raw class scores (logits) for classification.
After training the basic CNN model we see that after 8 epoch the training loss was 0.1295,
while the validation loss was approximately 1.914, with a validation accuracy of around
59.77%. When the model was applied to the test dataset, the test accuracy was
approximately 60.17%, and the test loss was about 1.949

<img width="1206" height="1075" alt="image" src="https://github.com/user-attachments/assets/a2aaeb8b-6ab9-4377-aed3-3ef9fa4ba0d7" />

ResNet 18

The chosen architecture for the ResNet-18 model is used for image classification with
PyTorch Lightning. It loads a pretrained ResNet-18, modifies the final layer to output a
custom number of classes, and tracks accuracy using torchmetrics. The model is trained with
cross-entropy loss, incorporating early stopping and model checkpointing to prevent
overfitting and save the best model. Data preprocessing includes resizing and normalization,
with the dataset split into training, validation, and test sets. The training, validation, and test
steps are managed by PyTorch Lightning's Trainer, which automates device handling and logs
key metrics like loss and accuracy.
After training the ResNet-18 model we see that after 8 epoch the training loss was 0.9729,
with a validation loss of approximately 1.0493 and a validation accuracy of around 77.30%.
When tested on the test dataset, the model achieved a test accuracy of about 77.50%, with
a test loss of approximately 0.997.

<img width="1253" height="927" alt="image" src="https://github.com/user-attachments/assets/f9fe9606-d03c-42e1-8367-89a116b59bec" />

Regularization

We selected the ResNet-18 (Res-18) model for its superior performance, achieving an
accuracy of approximately 77%, compared to around 60% for the Basic CNN model.
Additionally, the Res-18 model demonstrated a lower test loss of about 0.99, in contrast to
1.94 for the Basic CNN model. This shows that the Res-18 model not only performs better in
terms of accuracy but also generalizes more effectively, with a lower test loss indicating its
ability to make more accurate predictions on unseen data.
To further enhance the performance of the Res-18 model, we applied regularization
techniques through data augmentation to improve its robustness and generalization
capabilities. These augmentations included Random Horizontal Flip, which flips images
horizontally with a 50% probability, making the model invariant to left-right orientation. We
also applied Random Rotation, allowing the images to rotate by up to ±30 degrees, which
helps the model handle variations in object orientation. Additionally, Color Jitter was
introduced, making random changes to the image's brightness, contrast, saturation, and
hue, simulating various lighting conditions and improving the model's ability to generalize in
different environments.
Furthermore, the images were center-cropped to 160x160 pixels and resized to 64x64 to
maintain consistent image dimensions throughout the dataset. Finally, the images were
normalized using specific statistics from the Imagenette dataset, ensuring standardized pixel
values to help with smoother and faster convergence during training. These augmentations
collectively contribute to making the Res-18 model more resilient to variations in real-world
data, thereby improving its overall performance.
When compared to the standard Res-18 model without regularization, the model with
regularization shows a notable improvement. The accuracy after 6 epoch increased to about
81.07%, compared to 77.50% for the regular Res-18 model, and the test loss dropped to
approximately 0.823, compared to 0.99 for the regular Res-18 model. The Res-18 model with
regularization also outperformed the non-regularized model on the validation set, achieving
a validation accuracy of 81.1% and a validation loss of 0.8388, compared to the regular Res-
18 model’s validation accuracy of 77.30% and validation loss of 1.0493.
Moreover, although the train accuracy for the regularized Res-18 model was slightly lower at
97.05% compared to 97.29% for the non-regularized model, it achieved a lower train loss of
about 0.088, compared to the train loss of the regular Res-18 model, which was higher. This
indicates that the model with regularization not only performs better on validation and test
data but also helps to reduce overfitting, achieving a more generalizable and robust model
overall.

<img width="1338" height="977" alt="image" src="https://github.com/user-attachments/assets/1d0cd496-92bf-4551-b15f-ceefa76b2ea6" />

Transfer Learning
We choose the ResNet-18 model with regression due to its high accuracy of approximately
81%. The model is implemented using the ResNet18TransferLearning class, which is
designed with PyTorch Lightning's `L.LightningModule`. This framework simplifies the
process of training and managing the model. The architecture utilizes ResNet-18, a deep
residual network that addresses common issues like vanishing gradients through residual
blocks and skip connections.
Initially, the ResNet-18 model is loaded without pretrained weights. However, the final fully
connected (fc) layer is replaced with a new one that outputs 10 values, corresponding to the
10 classes in the CIFAR-10 dataset. Pretrained weights from a prior model are then loaded
into the network, excluding the last layer, to enable transfer learning. This approach allows
the model to benefit from prior knowledge, improving convergence and speeding up the
training process.
The training_step and validation_step methods compute the cross-entropy loss and
accuracy, logging these metrics for both the training and validation datasets. The optimizer
used is Adam, a widely adopted optimizer in deep learning, which updates the model's
parameters during training.

Transfer learning

plays a crucial role here, as it allows the model to leverage pretrained
weights, speeding up training and improving generalization, especially when dealing with
smaller datasets like CIFAR-10. This architecture efficiently combines the strengths of deep
residual networks and transfer learning, making it highly effective for the CIFAR-10 task
while maintaining computational efficiency.
We see that in transfer learning after 19 epoch we have test accuracy of about 78.60% with
test loss 0.6422 and it also train loss of about 0.4906 and train accuracy of about 82.63% and
also we have validation accuracy 78.66% and loss about 0.4906.

<img width="1253" height="576" alt="image" src="https://github.com/user-attachments/assets/e3823070-9973-4fcd-9621-74e351b91c78" />

<img width="562" height="681" alt="image" src="https://github.com/user-attachments/assets/1275fcb8-e342-43c0-8add-237aac741c0a" />

