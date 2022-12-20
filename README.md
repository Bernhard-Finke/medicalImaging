# medicalImaging

To run this code download the LC25000 dataset from https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images
Then extract the colon_image_sets from the zipped file, and add it to your local directory.

After experimentation with the colon cancer dataset from LC25000, it was determined that the feature augmentation used to increase the size of the dataset, led to the model learning the noise in the entire dataset, including the test set too effectively, resulting in an almost 100% accuracy (see saved model parameters). As a result, it was decided to experiment with a different disease dataset (see master branch).

Citation:
Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). arXiv:1912.12142v1 [eess.IV], 2019
