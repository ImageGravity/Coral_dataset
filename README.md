# Towards an automated classification and detection method for bleaching corals 

The goal of this project is to create a pipeline to detect coral reefs under thermal distress and quantify the level of damage.

## Detecting corals

Images from the [BENTHOZ-2015](https://figshare.com/articles/BENTHOZ-2015_public_data_set/1524165) public dataset were utilized to fine-tune a CNN model based on [VGG-16](https://neurohive.io/en/popular-networks/vgg16/) and pre-trained on [ImageNet](http://www.image-net.org/) to accomplish the task of classifying images from the Great Barrier Reef. When constrained to the 27 classes with more than 1000 sample images, the model raches an F1-Score of 0.67 when employing a confidence threshold of 0.35.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1A0nVi7DRlJP2uMIOBcL5YItHYnOrI6w7">
</p>
<p align="center">
  F1-Score vs. Confidence threshold
</p>

## Quantifying coral bleaching

A small-scale experiment was conducted using *Pocillopora damicornis* as a model organism. Twelve coral specimens were subjected to a higher-than-optimum temperature followed by a period of optimal temperature to induce bleaching. Pictures were obtained and k-means clustering was used to classify the pixels on the images by color into categories background, healthy coral, and bleached coral. The percentage of bleaching on a specimen was quantified as the percentage of pixels that fell on the bleached cluster.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1p1zNO6dC4jOKmv8GN2naVPN74XRfMKS-">
</p>

## Conclusion
This work shows a proof-of-concept to implement a method to automatically survey the state of coral reefs under the effects of climate change.

### Attribution:
Data was sourced from the Integrated Marine Observing System (IMOS). An initiative of the Australian Government being conducted as part of the National Collaborative Research Infrastructure Strategy.
