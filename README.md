Assignment-1-Pattern-Recognition

Dataset: 
Wine (scikit-learn built-in)

Algorithms used:
- PCA (2 components)
- Logistic Regression (original & PCA features)
- k-NN (Euclidean p=2 and Manhattan p=1)

To run:
1. Install dependencies:
   pip install scikit-learn pandas matplotlib seaborn numpy

2. Run the script:
   python assignment1_pr.py

Outputs will be saved into the `outputs/` folder:
- pca_train_scatter.png
- cm_Logistic_Original.png
- cm_Logistic_PCA2.png
- cm_kNN_Euclidean_p5.png
- cm_kNN_Manhattan_p5.png
- cm_kNN_PCA2.png
- summary.txt
- bias_variance_notes.txt
