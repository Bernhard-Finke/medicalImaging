# medicalImaging

To run this code download the Chest X-Ray dataset from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
We then manually merged the train, val, and test split into a folder named all_images, containing all normal images in a folder named NORMAL,
and all pneumonia images into a folder called PNEUMONIA.
The code in main.py then generates a train, val, test split.
We also took an additional, external test dataset from a smaller COVID-related pneumonia dataset found at:
https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets
We then merged all files to create a single test set found in external_test.

The main.py file contains code for feature extraction and fine tuning ImageNet networks on medical images, adapted from:
https://www.tensorflow.org/tutorials/images/transfer_learning

We tested three Keras pre-trained networks: MobileNetV2, Xception, and DenseNet121.
We used feature extraction for 10 epochs and fine-tuning for 11 epochs.
We found that MobileNetV2 performed best both on the train and val set, so continued our experiments with this network.
As a result, we experimented with adding L2 regularization to the MobileNet fine-tune layers, and limited fine-tuning
epochs by adding early stopping after 5 epochs with no improvement in val_accuracy. 
This was found to have no improvement on val_accuracy.
Finally, we experimented with increasing IMAGE_SIZE from (160, 160) to (320, 320).
Again no significant improvements were found in terms of val_accuracy.
Saved models and graphed training trajectories can be found in the folders: denseNet, mobileNetV2, and XCeption.
We report a final test set performance, based on our 21 epoch MobileNet saved model of:
Accuracy: 0.9574, Precision: 0.9855, Recall: 0.9556
And separately for our smaller external COVID-19 pneumonia set of:
Accuracy: 0.9149, Precision: 0.9432, Recall: 0.8830
It should be noted that in a medical context recall is usually a more important metric than precision, so future work
could focus on improving recall over precision.

Citations:
http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

Joseph Paul Cohen and Paul Morrison and Lan Dao. COVID-19 image data collection, arXiv, 2020. https://github.com/ieee8023/covid-chestxray-dataset

https://github.com/JordanMicahBennett/SMART-CT-SCAN_BASED-COVID19_VIRUS_DETECTOR/