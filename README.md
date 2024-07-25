## Spam SMS Detection

This project aims to build a machine learning model that can accurately classify SMS messages as spam or ham (not spam). The model is trained on a dataset of SMS messages and utilizes natural language processing (NLP) techniques to extract relevant features.

### Deployment: 
- https://smsoremail-spam-classifier.streamlit.app/

### Project Structure

- **spam.csv:** The dataset containing SMS messages labeled as spam or ham.
- **spam_detection.ipynb:** Jupyter Notebook containing the code for data cleaning, EDA, text preprocessing, model building, and evaluation.
- **app.py:** Streamlit app for real-time spam detection.
- **requirements.txt:** List of required Python libraries.

### Steps Involved

1. **Data Cleaning:**
   - Dropping irrelevant columns.
   - Renaming columns for clarity.
   - Encoding the target variable ('spam' or 'ham') using LabelEncoder.
   - Handling missing values and duplicates.

2. **Exploratory Data Analysis (EDA):**
   - Analyzing the distribution of spam and ham messages.
   - Calculating and visualizing the number of characters, words, and sentences in messages.
   - Identifying correlations between features.

3. **Text Preprocessing:**
   - Converting text to lowercase.
   - Tokenization.
   - Removing special characters, stop words, and punctuation.
   - Lemmatization.
   - Generating word clouds for spam and ham messages.

4. **Model Building:**
   - Extracting features using TF-IDF vectorization.
   - Appending the 'num_characters' feature.
   - Splitting the data into training and testing sets.
   - Training various machine learning models (Naive Bayes, SVM, Logistic Regression, Decision Tree, Random Forest, etc.).
   - Evaluating model performance using accuracy and precision metrics.

5. **Streamlit App:**
   - Building a user-friendly web application using Streamlit.
   - Allowing users to input SMS messages for real-time spam detection.
   - Displaying the predicted classification (spam or ham).

### How to Run

1. **Install Dependencies:**
   - pip install -r requirements.txt
2. **Run Jupyter Notebook:**
   - Open `spam_detection.ipynb` in a Jupyter Notebook environment.
   - Execute the code cells sequentially.

3. **Run Streamlit App:**
   - Access the app in your web browser at the provided local address (usually http://localhost:8501).

### Model Performance

The best performing model was **Bernoulli Naive Bayes** with a **98% accuracy** and **1.0 precision**.

### Future Improvements

- Experiment with different NLP techniques and feature engineering methods.
- Fine-tune hyperparameters for further optimization.
- Implement model deployment for real-world usage.

### Contributing

Contributions are welcome! Feel free to open issues or pull requests.

### License

This project is licensed under the MIT License.
