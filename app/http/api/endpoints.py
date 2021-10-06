from flask import Flask, json, g, request
from app.phishing.service import Service as service
from app.phishing.schema import TextInQuestion

@app.rout("/phishing, method["POST"])
def process():
  textInQuestion  = TextInQuestion().load.(json.loads(request.data))

  if textInQuestion.errors:
    return json_response({'error': textInQuestion.erros}, 422)
  
  response = service.process(textInQuestion)
  return json_response(response)

def json_response(payload, status=200):
  return (json.dumps(payload), status, {'content-type': 'application/json'})
