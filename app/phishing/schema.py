from marshmallow import Schema, fields

class Text(Schema):
  id = fields.Str()
  text = fields.Str()

class Email(Schema):
  id = fields.Int()
  subject = fields.Str()
  content = fields.Str()

