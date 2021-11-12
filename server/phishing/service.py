from operator import add
from .schema import Text
import pickle
from model_creation import clean_address, clean_body, clean_text
class Service(object):

  def processText(text):
    print("processing...")
    print(text)
	# Accessing and converting text content. Stored in list so value can be fed to trained model
    cleanedTextContent = [clean_text(text.get("content"))]

    # Loading trained classification model
    textModel = pickle.load(open("text_class_model.sav", 'rb'))

    # Getting prediction with probability
    textPrediction = textModel.predict_proba(cleanedTextContent)
    #textPrediction = textModel.decision_function(cleanedTextContent)
    print(textPrediction)
    return textPrediction[0][1] * 100

  def processEmail(email):
    # Accessing and converting email components. These are stored in lists so that their values can be fed to our trained model to predict on
    cleanedAdd = [clean_address(email.get("email_address"))]
    cleanedSub = [clean_body(email.get("subject_line"))]
    cleanedBod = [clean_body(email.get("content"))]

    # Loading trained class models
    addModel = pickle.load(open("address_class_model.sav", 'rb'))
    subModel = pickle.load(open("subject_class_model.sav", 'rb'))
    bodModel = pickle.load(open("body_class_model.sav", 'rb'))

    # Getting predictions with probability
    addPrediction = addModel.predict_proba(cleanedAdd)
    subPrediction = subModel.predict_proba(cleanedSub)
    bodPrediction = bodModel.predict_proba(cleanedBod)
    #addPrediction = addModel.decision_function(cleanedAdd)
    #subPrediction = subModel.predict_proba(cleanedSub)
    #bodPrediction = bodModel.decision_function(cleanedBod)
    # Creating aggregate prediction
    #overallHamPred = (addPrediction[0] + subPrediction[0] + bodPrediction[0])/3
    #overallSpamPred = (addPrediction[1] + subPrediction[1] + bodPrediction[1])/3
    #overallPred = [overallHamPred, overallSpamPred]
    '''
    predict_proba will return a list of 2 values where the first value is the probability that the input is not spam and 
    the second value is the probability that the input is spam
    '''
    #return overallPred[1]*100
    return ((addPrediction[0][1] + subPrediction[0][1] + bodPrediction[0][1])/3)*100
