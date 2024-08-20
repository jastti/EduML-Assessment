# EduML-Assessment
This project focuses on predicting student performance on diagnostic questions using various machine learning algorithms, including KNN, IRT, and Matrix Factorization. The objective is to enhance the accuracy of predicting whether a student will answer a given question correctly based on their previous responses and the responses of others. 

## Project Overview
This repository contains the implementation of machine learning models aimed at predicting student performance on educational diagnostic questions. The project was conducted as part of the CSC311 course at the University of Toronto.

## Algorithms Implemented
- **K-Nearest Neighbors (KNN):** Applied both user-based and item-based collaborative filtering to predict student responses.
- **Item Response Theory (IRT):** Developed a model to estimate student ability and question difficulty.
- **Matrix Factorization:** Used SVD and ALS techniques for collaborative filtering.
- **Ensemble Model:** Combined predictions from KNN, IRT, and Matrix Factorization models to enhance accuracy.

## Getting Started
To replicate the results, follow these steps:

1. Clone the repository to your local machine.
2. Install the required Python libraries: `NumPy`, `Pandas`, `Scipy`, `PyTorch`.
3. Run the provided scripts (`knn.py`, `item_response.py`, `matrix_factorization.py`, `ensemble.py`) to train and evaluate the models.

## **Files in This Repository**

- `knn.py`: Implementation of K-Nearest Neighbors algorithm.
- `item_response.py`: Implementation of Item Response Theory model.
- `matrix_factorization.py`: Implementation of Matrix Factorization methods.
- `ensemble.py`: Implementation of the ensemble model.
- `Final Report.pdf`: The final project report.

## **Contributors**

This project was completed by **Jasmine (Jiaxuan) Tian** and **Junhan Zhang**.
