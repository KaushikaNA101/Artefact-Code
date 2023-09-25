# Importing the Immportant Libraries  
import os
import matplotlib
import pandas as pd
import random
import numpy as np
import re  
import matplotlib.pyplot as plt
import seaborn as sns
import urllib
import string
import urllib.parse
import requests
import nltk
from nltk.corpus import stopwords
import re
import joblib
from joblib import dump
from joblib import load
from urllib.parse import urlparse, parse_qs
matplotlib.use('TkAgg')

# Importing the Scikit-learn text processing packages 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import label_binarize
# importing ML models from Scikit-learn
from sklearn.linear_model import LogisticRegression

# Importing accuracy and metric processes 
from sklearn.metrics import classification_report, confusion_matrix, completeness_score,accuracy_score,f1_score,roc_curve,roc_auc_score


#Creating Tokenizer to extract features from the URL data
def tokenizer(url):
    """Seperating unique features from the raw URL data
    by . / _ and removing common extensions and headers 
    such as https,com and www from the token list
    (url) represents a full length URL
    """
    #Spliting URL Segments by '-' and the'/', and boarder delimiters'_'
    tokens =re.split('[/-]', url)
    #Remvoing Subdomain Gaps and Extension Splits using the splited segments of the URL
    for i in tokens:
        if i.find(".") >= 0:
            dot_split = i.split('.')
            # Removing common extensions that prove no significance.
            if "www" in dot_split:
                dot_split.remove("www")
            if "com" in dot_split:
                dot_split.remove("com")
            if "https" in dot_split:
                dot_split.remove("https")
            tokens += dot_split
    # These subsegments are added to the token list 
    return tokens
    # A Sucessfully Execution Print is given
print ("\n### Defined the Tokenizer ! ###\n")


def train_model():
    print("\n### Packages Loaded ###\n")
    print("Loading the URL Dataset")
    # loading the dataset from the path specified
    # Read Using pandas Library under the 'url_data' object
    url_data = pd.read_csv(r"C:\Users\Kaush\Desktop\malicious_phish.csv")
    # Preprocessing With duplicate removal, NaN check and setting stopwords
    url_data.isnull().sum()
    missing_values_count = url_data.isnull().sum()
    # Displaying Missing Value Count
    print("Missing Value Count")
    print(missing_values_count)
    url_data.dropna(inplace=True)
    # Setting Stop Words to English ("the" "and" etc)
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


    # Testin URL is setup to apply and verifiy the tokenizer 
    # The random URL being the 5th + 1 URL the 'ur' column of the dataset
    test_url = url_data['url'][5]
    print(url_data)

    

    # Setting up a train & Test Split
    # Allocation 20 % for testing and by default 80 % becomes traning data
    test_percentage = 0.2
    # Randomstate is set to 42 to allow reproducibility on each iteration 
    train_data, test_data = train_test_split(url_data, test_size=test_percentage, random_state=42)
    # Target Variables for test and training is set us the 'label' column of the dataset
    labels = train_data['label']
    test_labels = test_data['label']

    print("\n### Testing Data Split ###\n")
    # Displaying the Train and Test Split counts specified by "len"
    print("- Counting Total Splits -")
    print("Training Samples:", len(train_data))
    print("Testing Samples:", len(test_data))

    # Class Type Counts For Testing
    count_test_classes = pd.value_counts(test_data['label'])
    count_test_classes.plot(kind='bar', fontsize=15, colormap='ocean')
    plt.title("Class Type Count (Testing)", fontsize=18)
    plt.xticks(rotation='horizontal')
    plt.xlabel("Type", fontsize=18)
    plt.ylabel("Class Type Count", fontsize=18)
    plt.show()

    # Class Type Counts For Training
    count_train_classes = pd.value_counts(train_data['label'])
    count_train_classes.plot(kind='bar', fontsize=15)
    plt.title("Class Type Count (Training)", fontsize=18)
    plt.xticks(rotation='horizontal')
    plt.xlabel("Type", fontsize=18)
    plt.ylabel("Class Type Count", fontsize=18)
    plt.show()
    
    # Printing the full test URL
    print("\n- Full URL -\n")
    print(test_url)
    # Obtaning the tokenized output of the URL using the defined tokenzier
    print("\n- Tokenized Output -\n")
    tokenized_url = tokenizer(test_url)
    print(tokenized_url)

    print("\n### Vectorizing Complete ###\n")

    # Extracting token counts from the test url terms given in (line 5) as a for Loop
    for token in list(dict.fromkeys(tokenized_url)):
        print("{} - {}".format(tokenized_url.count(token), token))

    # Creating Example Counts for the test URLs Vectorizers
    # The object excec aims to obtain unique text feature token counts for the test url
    print("\n Count Vectorizer (Test URL) -\n")
    exvec = CountVectorizer(tokenizer=tokenizer)
    exx = exvec.fit_transform([test_url])
    print(exx)
    # Prints a List
    print()
    print("=" * 50)
    print()
    # The same object is defined again but this time using TF-IDF and token counts are obtained
    print("\n- TFIDF Vectorizer for (Test URL) -\n")
    exvec = TfidfVectorizer(tokenizer=tokenizer)
    exx = exvec.fit_transform([test_url])
    print(exx)


    # Creating TFIDF Vectorizers for Training URL Data
    print(" Training TFIDF Vectorizer")
    # The vecotrizer is named as tvec
    # The tokenizer argument is set to the pre-defined tokenizer
    tVec = TfidfVectorizer(tokenizer=tokenizer)
    # Fit the TF-IDF vectorizer with training data in the url column of the dataset
    tVec.fit(train_data['url'])
    # Feature vectors are obtained and stored in tfidf_x for training data
    # Fit.Transform Returns a matrix 
    # This matrix represent each URL Row as a Document 
    # Each unique token term as a column
    tfidf_X = tVec.fit_transform(train_data['url'])

    # Creating the TFIDF Vectorizers on the Test URL Data
    print(" - TFIDF Vectorizer")
    # The same logic is applied to the testing data using the pre-learned configuration from above
    test_tfidf_X = tVec.transform(test_data['url'])
    print("\n### TFIDF Vectorizing Complete ###\n")

    # Creating Logistic Regression Model Using TF-IDF
    # Hyperparamet Tuning of setting the solver and max iteration count is apllied from its default 100
    # sag solver provided the highest accuracy for a large dataset with the cost being time to compute
    # Penalty is set to L2 regularization to aid in giving weight to important feature tokens
    LRG_TFIDF = LogisticRegression(solver='sag', max_iter=400, penalty= "l2")  # 100 to 300 provided convergence issues
    # The model is fitted with features obtanined from the traning features of the url data and target variable
    LRG_TFIDF.fit(tfidf_X, labels)
    

    score_LRG_TFIDF = LRG_TFIDF.score(test_tfidf_X, test_labels)
    predictions_LRG_TFIDF = LRG_TFIDF.predict(test_tfidf_X)
    creport_LRG_TFIDF = classification_report(test_labels, predictions_LRG_TFIDF)
    cmatrix_LRG_TFIDF = confusion_matrix(test_labels, predictions_LRG_TFIDF)

    print("\n## Logistic Regression Model Built##\n")

    # Save the logistic regression model to a file
    print("### Saving Logistic Regression Model ###")
    joblib.dump(LRG_TFIDF, 'logisticRGTF_model.joblib')
    print("### Saving the TFIDF Vectorizer Model ###")
    joblib.dump(tVec,'tfidf_vectorizer.joblib')

    # Create a suitable DF for the prediction table
    prediction_table = pd.DataFrame({'Actual Label': test_labels, 'Predicted Label': predictions_LRG_TFIDF})

    # Creating the prediction table
    print("\n### Prediction Table ###\n")
    print(prediction_table)

    # Classification report
    creport_LRG_TFIDF = classification_report(test_labels, predictions_LRG_TFIDF)
    print("\n### Classification Report ###\n")
    print(creport_LRG_TFIDF)

    # CM Report
    cmatrix_LRG_TFIDF = confusion_matrix(test_labels, predictions_LRG_TFIDF)
    print("\n### Confusion Matrix ###\n")
    print(cmatrix_LRG_TFIDF)
    class_labels = ['benign', 'malicious']

    # CM Heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(cmatrix_LRG_TFIDF, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels', fontsize=13)
    plt.ylabel('Actual Labels', fontsize=13)
    plt.title('Confusion Matrix', fontsize=15)
    plt.show()

    #Setting up Prediction metrics for ROC
    test_labels_binary = label_binarize(test_labels, classes=['benign', 'malicious'])

    # Get predicted probabilities for the positive class (malicious)
    predicted_probabilities = LRG_TFIDF.predict_proba(test_tfidf_X)[:, 1]

    fpr, tpr, thresholds = roc_curve(test_labels_binary, predicted_probabilities)
    roc_auc = roc_auc_score(test_labels_binary, predicted_probabilities)

    #ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Accuracy Score
    accuracy_LRG_TFIDF = accuracy_score(test_labels, predictions_LRG_TFIDF)
    print("\nAccuracy Score:", accuracy_LRG_TFIDF)


## Stage 2: Short URL Check, Conversion and Prediction ##
# Regular expression pattern to match common short URL services
short_url_service_pattern = re.compile(
    r'(bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
    r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
    r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
    r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
    r'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
    r'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
    r'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
    r'tr\.im|link\.zip\.net|shorturl\.at)'
)

# Function to check if a URL is short
def short_url_check(url):
    # Check if the URL matches the short URL pattern
    if re.search(short_url_service_pattern, url):
        return True
    # URL Length Check Threshold
    if len(url) < 30:
        return True
    return False
# Extracting the long URL by following potential redirects and return Long URL
def extracting_long_url(url):
    try:
        response = requests.head(url, allow_redirects=True)
        if response.status_code == 200:
            return response.url
        else:
            return url
# Error Exceptions during network and URL unavlability retruns the Long URL
    except requests.exceptions.RequestException:
        return url



# Function to classify a custom URL
def classify_custom_url(custom_url):
    # Check if the user-provided URL is short
    if short_url_check(custom_url):
        long_url = extracting_long_url(custom_url)
        print("Long URL:", long_url)
        return validateCustomURl(long_url)
    else:
        print("The URL is not short.")
        return validateCustomURl(custom_url)
        
# Function to define, vectorize and predict the Long URL Obtained     
def validateCustomURl(url):
    
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('logisticRGTF_model.joblib')
    # Vectorize the URL using the pre-trained vectorizer
    url_vectorized = vectorizer.transform([url])
    # Predict the class label of URL using loaded model
    prediction = model.predict(url_vectorized)
    print("Classification Result:", prediction[0])

## Stage 3 : Polymorphism Check ##
def check_polymorphic_url(user_url):
    # Rule Counter is set to 0 at the start
    Res = 0
    # Dynamic Path check for characters that contain 1 - 10 numbers after every word
    dynamic_path_check = re.compile(r'/\w+/\d{1,10}/')  
    # If regular expression patter is found (true) the res counter is increased by one
    check1 = bool(dynamic_path_check.search(user_url))
    if check1:
        Res += 1
        print("Dynamic Path Detected")
    # Query Parameter Count in URL
    parsed_url = urlparse(user_url)
    query_params = parse_qs(parsed_url.query)
    # If the URL supplied has more than one 2 query parameters the counter is increased by one again
    check2 = any(len(values) > 2 for values in query_params.values())
    if check2:
        Res += 1
        print("Query Parameters Detected")
    # Randomization check for random letters and digits in the URL
    random_chars = sum(1 for char in user_url if char in string.ascii_letters + string.digits)
    # If the threshold for random characters is 20% for the total URL, the counter is increased again
    check3 = random_chars / len(user_url) > 0.2
    if check3:
        Res += 1
        print("Randomization Detected")
    # Encoding presnece count for more than 2 instances of Encoder Manipulation
    decoded_url = urllib.parse.unquote(user_url)
    check4 = decoded_url.count('%')
    # If the encoding count is more than 2, the counter gets increased further
    if check4 > 2:
        Res += 1
        print("High URL Encoding Detected")
    # If the count is 1 output is no polymorphism
    if Res == 1:
        print("Polymorphic URL Status: No Polymorphic Or Dynamic Properties Detected!")
    # If the count is 2 output is Dynamic URL
    elif Res == 2:
        print("Polymorphic URL Status: Dynamic URL!")
    # If all checks match, the output is Poylmorphic URL
    else:
        print("Polymorphism Detected")
        print(Res)


# User-Terminal Interface 
# Designed under a for loop to allow the 3 options to be chosen and stops (breaks) on option 4
while True:
    print("Choose an option:")
    print("1. Train Model")
    print("2. Check Custom URL or Short URL using the Trained Model")
    print("3. Check Polymorphic URL")
    print("4. Exit")
    
    choice = input("Enter your choice (1/2/3/4): ")
    # Takes to the defined train model function 
    if choice == '1':
        train_model()
    # Takes to the defined Custom URL function
    elif choice == '2':
        custom_url = input("Enter the custom URL to check: ")
        result = classify_custom_url(custom_url)  
        print("Classification Result:", result)
    # Takes to the defined check polymorphic url function 
    elif choice == '3':
        user_url = input("Enter a URL: ")
        print("User Input URL:", user_url)
        check_polymorphic_url(user_url)

        result = classify_custom_url(user_url)  
        print("Classification Result:")
    # Exits the program
    elif choice == '4':
        print("Exiting the program.")
        break
    # Forces the model to only except between choice 1,2,3 and 4
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")


