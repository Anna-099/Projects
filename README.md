📊 Data Science & Healthcare Projects

This repository includes independent data science projects I developed alongside my professional work, exploring health data, cognitive phenomena, and statistical analysis using Python. These projects reflect my personal interest in applying analytical methods to real-world problems in healthcare. 

🔍 Projects Overview

💉1. Diabetes Health Indicators
File: Diabetes_Health_Indicators.ipynb

Summary:
Developed machine learning models to predict diabetes risk using the Behavioral Risk Factor Surveillance System (BRFSS) survey data. The study evaluated Decision Trees, Random Forests, and XGBoost classifiers, identifying the most important predictive risk factors. The Random Forest model with oversampling achieved the best performance with 90% accuracy and a 95% recall for diabetic cases, making it highly effective for identifying at-risk individuals.

Problem:
Early detection of diabetes is critical to prevent complications through timely intervention and provide insights for preventive care strategies.

Methodology:
Data Preprocessing: Applied random oversampling to address class imbalance between diabetic and non-diabetic cases.
Modeling Approaches: Decision Tree Classifier, Random Forest Classifier (with and without hyperparameter tuning), XGBoost Classifier
Evaluation Metrics: Accuracy, F1-score, Precision, Recall, Balanced Accuracy, and ROC-AUC.
Feature Importance: Identified key predictors such as BMI, Age, Physical Health, and High Blood Pressure.
Visualization: Plotted confusion matrices for best-performing models to assess prediction quality.

Tools:
scikit-learn, XGBoost, pandas, NumPy, Seaborn, Matplotlib

Results:
Overall, the Random Forest without hyperparameter tuning on oversampled data emerged as the best-performing model, with an accuracy of 90% and maintaining the same high recall of 95%, making it the most effective model for identifying diabetic patients.
Feature importance analysis highlighted several key predictors strongly associated with diabetes risk, including BMI, Age, Physical Health, General Health and High Blood Pressure.
Confusion matrix analysis confirmed the model’s strength in correctly identifying diabetic patients, ensuring that most high-risk individuals are flagged for further medical evaluation.

👁️2. Self-Motion Illusion Simulation
File: Self_motion_illusion.ipynb

Summary:
This project investigates the train illusion, where stationary observers perceive motion when adjacent trains move, by modeling the effects of noisy vestibular signals on self-motion perception. Using a drift-diffusion-inspired model, we simulated vestibular noise, leaky integration, and decision thresholds to predict motion judgments. Results indicate that higher vestibular noise increases the likelihood of illusory motion, demonstrating that variability in vestibular input can explain perceptual errors.

Research Problem:
How do noisy vestibular estimates of motion lead to illusory self-motion percepts? Understanding this phenomenon is critical for insights into human motion perception and balance.

Methodology:
Vestibular Signal Generation: Modeled vestibular input with Gaussian noise added to acceleration signals corresponding to motion or no motion.
Integration Mechanism: Applied a leaky integrator (analogous to a drift-diffusion process) to simulate accumulation of vestibular evidence over time.
Decision Mechanism: A thresholding step determined whether the integrated evidence exceeded a decision criterion, producing a “motion detected” or “no motion” judgment.
Model Evaluation: Computed the proportion of motion decisions across trials for each parameter combination and analyzed the relationship between vestibular noise and illusion frequency.
Visualization: Multi-panel plots showing motion decision frequency across noise levels for varying leak and threshold parameters.

Tools:
NumPy, SciPy, Matplotlib, Pandas

Results:
The model successfully reproduced train illusion behavior, predicting motion perception even when no self-motion occurred.
Noise-dependent behavior: Higher vestibular noise led to more frequent illusory motion judgments. For example, with σ=50, c=0.0004, thr=1.5:
The relationship between noise and illusion frequency was not strictly linear; it depended on the interaction of noise, integration leak, and threshold.
Parameter sweeps demonstrated that lower leakage and lower thresholds increased motion detection frequency, while higher thresholds and leaks reduced false-positive perceptions.
These results support the notion that noisy vestibular input can explain illusory self-motion, and that the illusion is more frequent when the signal-to-noise ratio is low.

🧠3. Brain Connectome Classification
File: Classifying Brain Connectomes

Summary:
Developed a Graph Convolutional Network (GCN) to predict schizophrenia based on patient data represented as a graph. 

Problem:
Early detection of schizophrenia can significantly improve patient outcomes by enabling timely intervention. 

Methodology:

Data Representation: Patient data converted into a graph structure to capture relationships between features.
Model Architecture: Graph Convolutional Network (GCN) implemented in PyTorch.
Evaluation Metrics: Accuracy on test set, confusion matrix to analyze prediction types.

Tools:
PyTorch, PyTorch Geometric, Matplotlib, Seaborn, NumPy

Results:
Training Loss gradually decreased from 0.6909 → 0.3875 over 100 epochs, indicating effective learning.
Test Accuracy: 88.9%, demonstrating strong predictive performance on unseen data.
Model shows strong discrimination between positive and negative cases, though some misclassifications remain.

🩺 4. Breast Cancer Survival Analysis
File: Cancer_data_survival_analysis

Summary:
Applied survival analysis techniques to study factors affecting breast cancer patient survival. Key predictors such as age, hormone therapy, and chemotherapy were analyzed using Kaplan-Meier curves and Cox proportional hazards models, providing actionable insights for clinical decision-making.

Problem:
Understanding which patient characteristics and treatments influence survival is critical for optimizing treatment strategies, identifying high-risk patient groups and informing clinical guidelines and patient counseling.

Methodology:
Data Exploration: Visualized individual patient survival times using lifespan plots.
Kaplan-Meier Estimation: Estimated survival probabilities for chemo vs. non-chemo groups.
Cox Proportional Hazards Model: Assessed hazard ratios for covariates including age, chemotherapy, and hormone therapy.
Log-Rank Test: Tested whether survival distributions between groups were statistically different.

Tools:
pandas, matplotlib, lifelines

Results:
Higher age significantly increases hazard; older patients have lower survival probabilities.
Hormone therapy has a statistically significant positive effect on hazard.
No statistically significant effect for chemotherapy: (log(HR) near 0; confidence interval crosses 0).
Kaplan-Meier curves show a visual separation between chemo and non-chemo groups; differences not statistically significant according to Cox model.
Log-Rank Test: p-value = 0.000105, confirms significant difference between selected example groups.

🚀How to Run:
Open the notebooks using Jupyter Notebook or Google Colab.
