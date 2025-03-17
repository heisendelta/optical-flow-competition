<h1 align="center">Optical Flow Prediction from the DSEC Event Camera Dataset</h1>

***Our model placed 8th in the final standings of the Optical Flow Prediction competition.***

## 1. Introduction

In this assignment, we handle the DSEC dataset, which consists of outputs of two color frame cameras and two high-resolution monochrome event cameras (Gehrig et al., 2021). Unlike RGB cameras, each frame consists of an event expressed using four channels: polarity, time, positional x and positional y coordinates. Our task was to predict optical flow, the components of the distance vectors of features between two consecutive frames (Teed and Deng, 2020), for each positional x, y coordinate. Most existing flow models such as FlowNet or RAFT train on three channel data, and to convert our four channel to three channel data we experiment with autoencoders consisting of convolutional layers and the dimensionality reduction techniques principal component analysis (PCA), sparse PCA and truncated singular value decomposition. We test our baseline model EvFlowNet (Zhu et al., 2018) with the vision transformer or ViT (Dosovitskiy et al., 2021); recurrent all-pairs field transforms or RAFT (Teed and Deng, 2020); pyramidal convolutional LSTM or PCLNet (Guan et al., 2019). Our best performing series of models were the PCLNets using images preprocessed using randomized resized crop and randomized horizontal and vertical flips.

## 2. Baseline Model: Native EvFlowNet

Without any prior image processing, the native EvFlowNet implementation performs up with EPE errors of 10-12. We apply Gaussian blur to reduce any noise and then train for two epochs to obtain an test EPE error of 2.98166. Here, standard or min-max normalization did not impact test error significantly. Global contrast normalization (GCN) and zero-phase component analysis (ZCA) whitening also did not significantly impact train EPE error, and we hypothesize that the sparsity of non-zero values in the frames of the DSEC dataset reduces the increases in efficiency from conventional normalization techniques. The loss function employed in training is end-to-end (EPE) error. The loss function implemented in the native EvFlowNet architecture, the combination of a photometric loss and smoothness loss (Zhu et al., 2018), performs subpar on training the dataset and yields increasing EPE errors after 19 batches. Hence, we train our model on EPE error, but we additionally monitor EvFlowNet’s native loss function to determine when training should be stopped early. EvFlowNet’s native loss function could not, however, be monitored for RAFT and any other non-FlowNet based architectures due to the shapes of flow vectors returned by the models.

## 3. Best Model: Modified Pyramid Convolutional LSTM (PCLNet)

The Pyramid Convolutional LSTM Network (PCLNet) architecture produced the best-performing series of models. The original paper by Guan et al. (2019) proposed this LSTM autoencoder to capture the time series nature of optical flow prediction by recognizing the ability for LSTM layers to store the states of past frames trained on. Guan et al. (2019) proposed the network as an unsupervised framework that did not rely on ground truth labels. However, for the purposes of fine-tuning the weights and biases of the ResNet backbone to the DSEC dataset, we make use of the ground truth provided. Additionally, we upscale the heights and widths of flow outputs by the factors of 4, 4, 6, 16 respectively using torch.nn.functional.interpolate to obtain flow vectors of size [2, 480, 640] that match the input x, y dimensions. We evaluate the eight vectors obtained, four flow vectors for two frames, individually, using EPE loss. However, since the differences seem insignificant and have no particular pattern, we take the element-wise average and use the resultant vector of shape [batch_size, 2, 480, 640] for prediction. It is important to note that the PCLNet architecture takes images with three channels, akin to the color images of RGB video cameras, and to resize our channels, we embed a convolutional layer that converts our four channels into three channels.

For image preprocessing we use randomized resized crop and randomized horizontal and vertical flips with a probability of 50%, as they perform marginally better than Gaussian blur and normalization techniques on the sparse data have little to no positive effect on EPE error. When training, we use a sliding window, an approach widely used in time series forecasting (Yang, 2018), where we use the past n frames as a window to forecast the next m frames. We define a variable window_size=n and use m=1 for all batches. Additionally, we take the element-wise mean for consecutive frames, and the variable window_concat defines the number of consecutive frames used. For example, in our modified PCLNet model with window_size=3 and window_concat=2, we take the past six frames and take the mean of every 2 consecutive frames to make a window of size 3.

## 4. Results

All models with ResNet backbones used pretrained weights from ImageNet. They were trained using the AdamW optimization function with an initial learning rate of 0.00002, a weight decay of 0.0005 an epsilon of 10-8 and the ExponentialLR scheduler with a gamma of 0.9. EvFlowNet and RAFT models were trained with the same optimization function with the same parameters and the OneCycleLR scheduler with total steps of 105.

| Model                                   | Backbone   | Image Preprocessing                                                              | EPE Error |
|-----------------------------------------|------------|----------------------------------------------------------------------------------|-----------|
| EvFlowNet (baseline)                    | ResNet     | Gaussian Blur                                                                    | 2.98166   |
| Modified PCLNet<br>(window_size=3)         | ResNet18   | Gaussian Blur                                                                    | 2.13978   |
| Modified PCLNet<br>(window_size=3)         | ResNet18   | Randomized Resized Crop, Randomized Horizontal and Vertical Flip                 | 2.13906   |
| Modified PCLNet<br>(window_size=3, window_concat=2) | ResNet34   | Randomized Resized Crop, Randomized Horizontal and Vertical Flip                 | 2.13877   |

<!-- <table style="margin: auto;  width: 80%;">
  <thead>
    <tr>
      <th style="padding: 8px;">Model</th>
      <th style="padding: 8px; text-align: center;">Backbone</th>
      <th style="padding: 8px;">Image Preprocessing</th>
      <th style="padding: 8px; text-align: center;">EPE Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px;">EvFlowNet (baseline)</td>
      <td style="padding: 8px; text-align: center;">ResNet</td>
      <td style="padding: 8px;">Gaussian Blur</td>
      <td style="padding: 8px; text-align: center;">2.98166</td>
    </tr>
    <tr>
      <td style="padding: 8px;">Modified PCLNet<br>(window_size=3)</td>
      <td style="padding: 8px; text-align: center;">ResNet18</td>
      <td style="padding: 8px;">Gaussian Blur</td>
      <td style="padding: 8px; text-align: center;">2.13978</td>
    </tr>
    <tr>
      <td style="padding: 8px;">Modified PCLNet<br>(window_size=3)</td>
      <td style="padding: 8px text-align: center;">ResNet18</td>
      <td style="padding: 8px;">Randomized Resized Crop, Randomized Horizontal and Vertical Flip</td>
      <td style="padding: 8px; text-align: center;">2.13906</td>
    </tr>
    <tr>
      <td style="padding: 8px;">Modified PCLNet<br>(window_size=3, window_concat=2)</td>
      <td style="padding: 8px; text-align: center;">ResNet34</td>
      <td style="padding: 8px;">Randomized Resized Crop, Randomized Horizontal and Vertical Flip</td>
      <td style="padding: 8px; text-align: center;">2.13877</td>
    </tr>
  </tbody>
</table> -->


## 5. Conclusion

The RAFT model performed with an EPE error of 10.34389, with no trend of improvement. This was higher than any of the sparse optical flow estimation models, and we hypothesize that the sparsity of the dataset and the dimensionality reduction we employed contributed to the poor performance. The RAFT model also returned three-channel output and we performed TSVD on the output before prediction. Dimensionality reduction techniques are designed to preserve the variance of the features being reduced, but the repeated reduction could have introduced noise or image artifacts. 

PCLNet performed better than the baseline EvFlowNet even though both models shared the same ResNet backbone because a higher number of, and more complex operations were possible with the convolution architectures of PCLNet. The ResNet34-based architecture performed with a lower error than the Resnet18 architecture for similar reasons, that a more complex architecture could capture more complex relationships between the event volume and ground truth flow.

## 6. Further Work

Due to the limited computational resources, we could not test any transformers or attention based mechanisms. The standalone ViT model performed poorly, but if coupled with a wrapper like PCLNet, it may have performed better. We attempted to use Swin Transformer (Liu et al., 2021), for its performance in semantic segmentation and object detection tasks, and overall ability to capture complex relationships, but we ran into computational constraints and problems with upscaling the outputs of the model. ViT, Swin Transformer, and many other transformers train on images of height and width 224 pixels, and resizing the outputs of these models using convolutional layers to the original 480 x 640 proved to be challenging. Integrating transformers as the backbones of networks such as PCLNet may drastically improve their accuracy, but would also escalate the computational time and resources required. 

## Additional Notes

The entire project was run using Jupyter notebooks on Google Colab or Kaggle. As such, the project was never run locally, and we cannot confirm whether the syntax of these files were copied correctly. To replicate the experiments we have conducted, please refer to the “notebooks/” directory. For our highest performing model, the modified PCLNet, please refer to “event_camera_pclnet2.ipynb”.

## Works Cited

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, 
  J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (arXiv:2010.11929). arXiv. https://doi.org/10.48550/arXiv.2010.11929

Gehrig, M., Aarents, W., Gehrig, D., & Scaramuzza, D. (2021). DSEC: A Stereo Event Camera Dataset for Driving Scenarios. IEEE Robotics and Automation 
  Letters, 6(3), 4947–4954. IEEE Robotics and Automation Letters. https://doi.org/10.1109/LRA.2021.3068942

Guan, S., Li, H., & Zheng, W.-S. (2019). Unsupervised Learning for Optical Flow Estimation Using Pyramid Convolution LSTM (arXiv:1907.11628). arXiv. 
  https://doi.org/10.48550/arXiv.1907.11628

Teed, Z., & Deng, J. (2020). RAFT: Recurrent All-Pairs Field Transforms for Optical Flow (arXiv:2003.12039). arXiv. https://doi.org/10.48550/arXiv.2003.
  12039

Yang, R. (2018, January 24). Omphalos, Uber’s Parallel and Language-Extensible Time Series Backtesting Tool. Uber Blog. https://www.uber.com/en-DE/blog/
  omphalos/

Zhu, A. Z., Yuan, L., Chaney, K., & Daniilidis, K. (2018, June 26). EV-FlowNet: Self-Supervised Optical Flow Estimation for Event-based Cameras. 
  Robotics: Science and Systems XIV. https://doi.org/10.15607/RSS.2018.XIV.062
