import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

class Topic(BaseModel):
    statement: str

app = FastAPI()

import joblib
model = joblib.load('new/model.pkl')
tf = joblib.load('new/tf.pkl')
le = joblib.load('new/le.pkl')


@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}

@app.post('/prediction')
def classify_topic(data: Topic):
    received = data.dict()
    text = received['statement']
    text = re.sub('[^A-Za-z0-9]', ' ', text.lower())
    print('text - ',text)
    tokenized_text = word_tokenize(text)
    clean_text = [" ".join(lemmatizer.lemmatize(word) for word in tokenized_text
        if word not in stopwords.words('english'))]
    print('--------------',clean_text)
    X = tf.transform(clean_text)
    pred_name = model.predict(X)
    p = model.predict_proba(X)
    p = p[0]
    print('out - ',pred_name,pred_name[0],p[pred_name[0]])
    y_pred = le.inverse_transform(pred_name)
    print('le_out - ',y_pred)
    return {'Output': received['statement'],'predicted_label': y_pred[0],'confidence_score':p[pred_name[0]]*100}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)
