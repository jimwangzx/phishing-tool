from .schema import Text
from .schema import Email
import pickle
from model_creation import clean_address, clean_body, clean_text
class Service(object):

  def processText(text):
    # Accessing and converting text content. Stored in list so value can be fed to trained model
    cleanedTextContent = [clean_text(text["text"])]

    # Loading trained classification model
    textModel = pickle.load(open("text_class_model", 'rb'))

    # Getting prediction with probability
    textPrediction = textModel.predict_proba(cleanedTextContent)
    return 99

  def  processEmail(email):
    # Accessing and converting email components. These are stored in lists so that their values can be fed to our trained model to predict on
    cleanedAdd = [clean_address(email["email_address"])]
    cleanedSub = [clean_body(email["subject"])]
    cleanedBod = [clean_body(email["content"])]

    # Loading trained class models
    addModel = pickle.load(open("address_class_model", 'rb'))
    subModel = pickle.load(open("subject_class_model", 'rb'))
    bodModel = pickle.load(open("body_class_model", 'rb'))

    # Getting predictions with probability
    addPrediction = addModel.predict_probs(cleanedAdd)
    subPrediction = subModel.predict_probs(cleanedSub)
    bodPrediction = bodModel.predict_probs(cleanedBod)

    # Creating aggregate prediction
    overallHamPred = (addPrediction[0] + subPrediction[0] + bodPrediction[0])/3
    overallSpamPred = (addPrediction[1] + subPrediction[1] + bodPrediction[1])/3
    overallPred = [overallHamPred, overallSpamPred]
    '''
    predict probs will return a list of 2 values where the first value is the probability that the input is not spam and 
    the second value is the probability that the input is spam
    '''


    return 100

