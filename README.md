# Multiclass U-Net Segmentation for Liver Tumor Detection in CT-Scan Images

This repository dedicated to liver tumor detection in CT-scan images through an advanced multiclass U-Net segmentation approach. Leveraging state-of-the-art techniques such as window leveling, window blending, and one-hot semantic segmentation, the method aims to enhance the accuracy and efficiency of liver tumor identification.

### Key Features:

1. **UNet Architecture:**
   - This repository utilizes the powerful U-Net architecture, a convolutional neural network (CNN) designed for image segmentation. This architecture is particularly effective in handling medical imaging tasks, providing accurate segmentation results. You can see the model in ![Multiclass U-Net](https://github.com/YourUsername/YourRepoName/blob/main/path_to_image/U-Net%20Visualisasi.png)

2. **Multiclass Semantic Segmentation:**
   - Unlike traditional binary segmentation, our approach supports multiclass segmentation. This means the model can distinguish between different classes of tissues, allowing for more nuanced and detailed segmentation, crucial for accurate liver tumor detection.
   - ![Multiclass Semantic Segmentation](/Users/hansimgluck/Desktop/Multiclass_U-Net/U-Net_Model.png)

### How to Use:

1. **Dataset Preparation:**
   - Organize your CT-scan dataset, ensuring proper labeling for liver tumor regions and background. This is crucial for training the model effectively.
   - https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation

2. **Model Training:**
   - Utilize the provided training scripts to train the UNet model on your dataset. Tweak hyperparameters as needed for optimal performance.
   

3. **Inference and Evaluation:**
   - Run the inference scripts on new CT-scan images to detect liver tumors. Evaluate the model's performance using metrics like Dice Coefficient Similarity and Intersection Over Union for multiclass.

4. **Customization:**
   - Feel free to customize the model architecture, training pipeline, or post-processing steps to better suit your specific requirements or dataset characteristics.


# Advancing My Deep Learning Skills for Images with PyTorch! 🖼️

I recently completed an intensive 4-hour advanced course where I deepened my knowledge and skills in image processing and deep learning. Here's a quick summary of what I've learned:

## Image Classification with CNNs 🧠
- **Binary and Multi-class Classification**: Gained in-depth knowledge of both binary and multi-class classification.
- **Convolutional Neural Networks (CNNs)**: Developed powerful models using CNNs.
- **Pre-trained Models**: Leveraged pre-trained models to reduce training time and enhance performance.

## Object Recognition 🔍
- **Bounding Boxes**: Learned to use bounding boxes for object localization.
- **Advanced Models**: Explored robust models like R-CNN and Faster R-CNN.
- **Non-max Suppression (NMS)**: Applied NMS to refine detections.
- **Model Evaluation**: Evaluated models using Intersection over Union (IoU).

## Image Segmentation 🖍️
- **Segmentation Masks**: Worked with segmentation masks to classify pixels within images.
- **Instance Segmentation**: Gained deep insights into instance segmentation with Mask R-CNN.
- **Semantic Segmentation**: Delved into semantic segmentation using U-Net.
- **Panoptic Segmentation**: Tackled the comprehensive task of panoptic segmentation.

## Image Generation with GANs 🎨
- **Generative Adversarial Networks (GANs)**: Built a strong understanding of GANs.
- **Deep Convolutional GANs (DCGAN)**: Implemented and trained DCGANs.
- **Model Training & Evaluation**: Focused on training and evaluation using Frechet Inception Distance (FID).

This advanced course has significantly sharpened my skills in computer vision and deep learning. I’m excited to apply these skills to real-world challenges and look forward to what’s next in this ever-evolving field!
