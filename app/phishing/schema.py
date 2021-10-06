from marshmallow import Schema, fields

class textInQuestion(Schema):
  id = fields.Int(reguired=True)
  text = fields.Str()

