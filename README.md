# AI Learning & Certification Program: Cycle 4 - Deep Learning

https://circuit.intel.com/content/np/sites/AIInside/Mentoring/cycle-4---deep-learning.html 

## Student: Sean Condon
[sean.condon@intel.com](mailto:sean.condon@intel.com) 

Completion date: 2024-11-15

## Project B - Wikipedia - Image/Caption Matching
[link](https://www.kaggle.com/competitions/wikipedia-image-caption/overview)

The purpose of the project is to build a model that automatically retrieves the text closest to an image.
The model is trained to associate given images with article titles or complex captions, in multiple languages.

The dataset consists of images and their captions from Wikipedia articles,
divided into training and testing sets. 
The training set contains 1,000,000 image-caption pairs, while the testing
set contains 100,000 image-caption pairs. 

The goal is to develop a model that can accurately match input images to their corresponding captions.

> **Note:** The training set is quite huge at over 275 GB far in excess of the available storage
> and processing power on the local machine, or most available cloud machines without serious GPU
> acceleration.
> I make the best effort to show the mechanism of the solution without running the code on the full dataset.

## Image captioning
Image captioning is a challenging task that combines computer vision and natural language processing.

Building a deep learning model for image captioning typically involves using a combination of
Convolutional Neural Networks (CNNs) for image feature extraction and 
Recurrent Neural Networks (RNNs) or Transformers for generating captions. 

A conventional high-level approach would consist of: 

* **Data Preparation**: Preprocess the images and captions. Convert images to tensors and tokenize captions.
* **Feature Extraction**: Use a pre-trained CNN (like ResNet or Inception) to extract features from images.
* **Caption Generation**: Use an RNN (like LSTM or GRU) or a Transformer to generate captions based
  on the extracted features.
* **Training**: Train the model using a suitable loss function (like cross-entropy loss) and an optimizer (like Adam).
* **Evaluation**: Evaluate the model using metrics like BLEU, METEOR, or CIDEr.



## Alternative approach using Vision Transformers (ViT)
In the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2020),"
the authors propose a Vision Transformer (ViT) model that directly applies the Transformer architecture to images.

This is trained on a large dataset of images of as much as 300 million images (JFT-300M), and can give state-of-the-art results on
image classification tasks.

On Hugging Face, the `transformers` library provides pre-trained ViT models like that can be used for image **classification**
tasks - for example https://huggingface.co/docs/transformers/tasks/image_classification

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg" alt="Vision Transformer Architecture" width="500">

## Vision Transformers for Image Captioning
Other work proposes using the [Vision Encoder Decoder Model](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder) 
for image captioning tasks by combining any pretrained Transformer-based vision model
as the encoder (e.g. ViT, BEiT, DeiT, Swin) and any pretrained language model as the 
decoder (e.g. RoBERTa, GPT2, BERT, DistilBERT).

<img src="https://ankur3107.github.io/assets/images/vision-encoder-decoder.png" alt="Vision Encoder Decoder Architecture" width="500">

While it is appealing to try to create captions using this Vision Encoder Decoder model, 
it will never exactly match the captions in the wikipedia caption list (../data/test_caption_list.csv), 
with its specific details and multiple languages. 

## Vision Transformers for Image Classification
What's really needed in this exercise is to classify the images into the correct caption, and 
this can be most easily done by considering the 97000 entries in the test set as a classification task.

The task then becomes to fine tune the Vision Transformer model on the wiki caption list.

This will not be influenced by the language of the captions, as the strings 
are just treated as a class (regardless of language).

## Data preparation

The wiki caption list can be loaded straight from the CSV file with the index of each
used as the classification label.

The training dataset includes:
* Images encoded in base64 corresponding to URLs (`image_data_train`)
* training data linking the URL to the `caption_title_and_reference_description` ( e.g. `train-{0000x}-of-{00005}.tsv`)

Through data processing, these can be combined in to one dataset with the image data and the caption text.

The images will be loaded from the base64 encoded image data and resized to the appropriate
size for the model (3 channel 224x224 pixel for ViT). 

The [ViTImageProcessor](https://huggingface.co/docs/transformers/v4.46.2/en/model_doc/vit#transformers.ViTImageProcessor) 
can be used to perform this task.

Some authors it is often beneficial to use a higher resolution than pre-training
([Touvron et al., 2019](https://arxiv.org/abs/1906.06423)),
([Kolesnikov et al., 2020](https://arxiv.org/abs/1912.11370)). In order to
fine-tune at higher resolution, the authors perform 2D interpolation of the pre-trained
position embeddings, according to their location in the original image.

## Vision Transformer model

Following the original Vision Transformer model proposed in 
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,
some follow-up works have been made to improve the performance of the model.

* [DeiT](https://huggingface.co/docs/transformers/model_doc/deit): Data-efficient image Transformer by Facebook AI
* [BEiT](https://huggingface.co/docs/transformers/model_doc/beit):   BERT pre-training of Image Transformers) by Microsoft Research
* [DINO](https://huggingface.co/models?other=dino): a method for self-supervised training of Vision Transformers) by Facebook AI.
* [MAE](https://huggingface.co/docs/transformers/model_doc/vit_mae): Masked Autoencoders by Facebook AI.

To facilitate fine-tuning, the `transformers` library provides pre-trained models
that are checkpointed at either
1. pre-trained on ImageNet-21k (a collection of 14 million images and 21k classes) only, or 
2. also fine-tuned on ImageNet (also referred to as ILSVRC 2012, a collection of 1.3 million images and 1,000 classes).

For our purposes in this exercise we will use the `vit-base-patch16-224-in21k` model, 
and we will fine tune it ourselves using the wiki caption set as described above.

We will use supervised pre-training, as the original authors found this performs best.

We will load the model in half-precision (e.g. torch.float16 or torch.bfloat16) for best performance.

## Implementing the solution
My preference is for a Python program that can be run from the command line,
rather than a Jupyter notebook, as this is more easily automated.

> Unfortunately I did not get to finish this project, but I hope this document gives a good overview of the approach.

Run `poetry install` to install the required dependencies

Call `poetry run pyython wiki_image_caption.py` to run the program.

The program loads the caption set, loads the model, and fine-tunes the model on the caption set.

The approach will be very similar to that in the example notebook available at
https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb 

Instead of the Euronet Dataset in that article we will use the wiki training set, loaded from the TSV files.

The training of the model will focus only on the last layer of the model, as
the model is already pre-trained on ImageNet-21k, with a similar loss function
and SGD optimizer and learning rate as the given example


## Running the inference
Once the model has been trained, inference can be done on the test images.

As each one is suplied to the model, a prediction will be made as to the caption,
which can then be passed to a `softmax()` function and the top 5 captions can be selected
using the `torch.topk()` function.

In this way the output dataset of a 2 column CSV file can be created with the image ID 
and the top 5 captions that apply.

## Conclusion
This approach provides a novel approach to the Wikipedia Image Caption project, using 
a Vision Transformer model that was not considered in the competition results at the time

