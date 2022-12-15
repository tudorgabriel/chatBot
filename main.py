import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
from sklearn.metrics.pairwise import cosine_similarity

dataframe = pd.read_csv(r'C:\Users\Xivic\Downloads\openfabric-test\openfabric-test\questions.csv', error_bad_lines=False)
dataframe.dropna(inplace=True)

vectorizer=TfidfVectorizer()
vectorizer.fit(numpy.concatenate((dataframe.Question,dataframe.Answer)))

vectorized_questions=vectorizer.transform(dataframe.Question)

print('Hello the ChatBot is online, please type some science questions')

while True:

    user_input=input()

    vectorized_user_input = vectorizer.transform([user_input])

    similarities = cosine_similarity(vectorized_user_input,vectorized_questions)

    closest_question = numpy.argmax(similarities,axis=1)

    answer = dataframe.Answer.iloc[closest_question].values[0]

    print(answer)

    if(user_input=='Stop'):
      break 











