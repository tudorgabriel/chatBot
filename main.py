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











# import os
# import warnings
# # from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

# from openfabric_pysdk.context import OpenfabricExecutionRay
# from openfabric_pysdk.loader import ConfigClass
# from time import time


# ############################################################
# # Callback function called on update config
# ############################################################
# def config(configuration: ConfigClass):
#     # TODO Add code here
#     pass


# ############################################################
# # Callback function called on each execution pass
# ############################################################
# # def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
# #     output = []
# #     for text in request.text:
# #         # TODO Add code here
# #         response = ''
# #         output.append(response)

# #     return SimpleText(dict(text=output))
