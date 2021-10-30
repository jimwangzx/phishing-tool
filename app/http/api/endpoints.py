from flask import Flask, json, g, request
from app.phishing.service import Service as service
from app.phishing.schema import Text
from app.phishing.schema import Email 
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route("/phishing/text", methods=["POST"])
def processText():
  print("I'm here")
  print(request.form)
  text  = Text().load(json.loads(request.data))

  if text.errors:
    return json_response({'error': text.errors}, 422)
  print(text)  
  response = service.processText(text.data)
  return json_response(response)

@app.route("/phishing/email", methods=["POST"])
def processEmail():
  email = Email().load(json.loads(request.data))
  if email.errors:
    return json_response({'error': email.errors}, 422)
  response = service.processEmail(email.data)
  return json_response(response)

def json_response(payload, status=200):
  return (json.dumps(payload), status, {'content-type': 'application/json'})
