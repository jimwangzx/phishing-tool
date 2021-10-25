from flask import Flask, json, g, request
from app.phishing.service import Service as service
from app.phishing.schema import Text
from flask_cors import CORS
import json
from types import SimpleNamespace

app = Flask(__name__)
CORS(app)
@app.route("/phishing/text", methods=["POST"])
def processText():
  text  = Text().load(json.loads(request.data))
  print(text)
  if text.errors:
    return json_response({'error': text.errors}, 422)
  response = ""
  if json.loads(text.data)['email_address']:
    response = service.processEmail(data)
  else:
    response = service.processText(data)
  return json_response(response)

def json_response(payload, status=200):
  return (json.dumps(payload), status, {'content-type': 'application/json'})
