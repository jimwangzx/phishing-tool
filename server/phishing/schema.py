from marshmallow import Schema, fields

class Text(Schema):
  id = fields.Str()
  content = fields.Str()
  subject_line = fields.Str()
  email_address = fields.Str()
