import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label_num']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


vectorizer = CountVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)


model = MultinomialNB()
model.fit(X_train_vec, y_train)


def predict_message(message):
    msg_vec = vectorizer.transform([message])
    prediction = model.predict(msg_vec)[0]
    probability = model.predict_proba(msg_vec)[0]
    
    label = "SPAM" if prediction == 1 else "NOT SPAM"
    confidence = probability[prediction] * 100
    
    print(f"Message : \"{message}\"")
    print(f"Result  : {label}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 60)


predict_message("Congratulations! You won a $1000 Walmart gift card. Click here to claim now!")
predict_message("Hey, are we still meeting for lunch tomorrow?")
predict_message("URGENT! Your account has been suspended. Reply YES to reactivate.")
predict_message("Don't forget to bring the documents to the office.")
predict_message("Free entry in 2 a weekly competition to win FA Cup tickets! Text WIN to 87121")