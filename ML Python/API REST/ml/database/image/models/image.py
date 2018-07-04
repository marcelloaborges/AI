from application import db

class Image(db.Document):    
    name = db.StringField()    
    file = db.FileField(thumbnail_size=(100, 100, True))
    category = db.IntField()
    sub_category = db.IntField()