<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    
</head>
<body>

<h2>Project Overview</h2>
<p>
This project implements a <strong>Convolutional Neural Network (CNN) using MATLAB</strong>
to classify skin lesion images into <strong>seven dermatological categories</strong>:
Melanocytic nevi (<code>nv</code>), Melanoma (<code>mel</code>), Benign keratosis (<code>bkl</code>),
Basal cell carcinoma (<code>bcc</code>), Actinic keratoses (<code>akiec</code>),
Vascular lesions (<code>vasc</code>), and Dermatofibroma (<code>df</code>).
</p>

<h2>Dataset</h2>
<p>
The model is trained using the <strong>HAM10000 dataset</strong>, which is publicly available
on Kaggle and can be downloaded for free from the following link:
</p>
<p>
<a href="https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000" target="_blank">
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
</a>
</p>

<p>
After extracting the dataset, a custom sorting script is used to preprocess the data.
The script reads lesion labels from <code>HAM10000_metadata.csv</code>, matches image IDs
to their corresponding lesion types, and organizes the images into class-specific
directories compatible with MATLAB’s <code>imageDatastore</code>.
</p>

<h3>Dataset Directory Structure</h3>
<pre>
data/
 ├── nv/
 ├── mel/
 ├── bkl/
 ├── bcc/
 ├── akiec/
 ├── vasc/
 └── df/
</pre>

<h2>CNN Architecture</h2>
<ul>
    <li>Input size: <strong>224 × 224 × 3</strong> (RGB images)</li>
    <li>Four convolutional blocks</li>
    <li>Each block consists of:
        <ul>
            <li>Convolution</li>
            <li>Batch Normalization</li>
            <li>ReLU activation</li>
            <li>Max Pooling</li>
        </ul>
    </li>
    <li>Progressive filter sizes: <strong>32 → 64 → 128 → 256</strong></li>
    <li>Global Average Pooling to reduce parameters</li>
    <li>Dropout layer to prevent overfitting</li>
    <li>Fully connected layer with <strong>7 neurons</strong> (one per class)</li>
    <li>Softmax layer for probability distribution</li>
    <li>Classification layer for final prediction</li>
</ul>

</body>
</html>