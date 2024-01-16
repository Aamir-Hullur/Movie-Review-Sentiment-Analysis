import streamlit as st
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load your trained models and other necessary components
model1 = joblib.load(r"saved_models/MNB.pkl")  # MultinomialNB
model2 = joblib.load(r"saved_models/SVC.pkl")  # LinearSVC
tfidf = joblib.load(r"saved_models/tfidf.pkl")

nltk_resources = ['stopwords','wordnet']

for resource in nltk_resources:
    nltk.download(resource)

def cleaning_text(text):
    # Remove non-alphabetic characters and lowercase the text
    text = re.sub("[^a-zA-Z]", " ", text).lower()

    # Tokenize the text and remove stop words
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    cleaned_text = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # cleaned_text = [wordnet.lemmatize(word) for word in words if word not in set(stop_words)]

    # Rejoin the cleaned words
    return " ".join(cleaned_text)

def predict_sentiment(review, model):
    # Preprocess the review
    cleaned_review = cleaning_text(review)
    # Vectorize the review
    vectorized_review = tfidf.transform([cleaned_review])
    # Predict the sentiment
    prediction = model.predict(vectorized_review)
    return prediction[0]

def show_top_influencing_words(model, vectorizer, review, num_words=5):
    # Vectorize the review
    vectorized_review = vectorizer.transform([review])

    # Get feature names and model coefficients
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]

    # Sort the coefficients and get top influencing words
    top_positive_indices = coefficients.argsort()[-num_words:][::-1]
    top_negative_indices = coefficients.argsort()[:num_words]

    top_positive_words = feature_names[top_positive_indices]
    top_negative_words = feature_names[top_negative_indices]

    st.write("Top words influencing positive sentiment: ", ', '.join(top_positive_words))
    st.write("Top words influencing negative sentiment: ", ', '.join(top_negative_words))

def main():
    st.title("Movie Review Sentiment Analysis")

    review = st.text_area("Enter the Review", height=150)

    if st.button("Analyze"):
        if review:
            # Predict sentiment using both models
            prediction1 = predict_sentiment(review, model1)  # MultinomialNB
            prediction2 = predict_sentiment(review, model2)  # LinearSVC

            # Display predictions
            st.markdown("### MultinomialNB Prediction:")
            if prediction1 == 1:
                st.success("Positive")
            else:
                st.error("Negative")

            st.markdown("### LinearSVC Prediction:")
            if prediction2 == 1:
                st.success("Positive")
            else:
                st.error("Negative")

            # Show top influencing words for LinearSVC
            if model2:
                show_top_influencing_words(model2, tfidf, review)

if __name__ == '__main__':
    main()
