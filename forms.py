from wtforms import Form, StringField, PasswordField, validators, SubmitField, FieldList, MultipleFileField

class AccountForm(Form):
    password = PasswordField('Password', [
        validators.Length(min=4, max=25, message="Password must be between 4 and 25 characters long."),
        validators.InputRequired(),
        validators.Regexp('^[a-zA-Z0-9!@#$%^&*_+-=]+$', message="Password must contain only letters,and common special characters.")
    ])
    submit = SubmitField('Submit')

class AccountFormUpdate(Form):
    current_password = PasswordField('current_password', [
        validators.Length(min=4, max=25, message="current password must be between 4 and 25 characters long."),
        validators.InputRequired(),
        validators.Regexp('^[a-zA-Z0-9!@#$%^&*_+-=]+$', message="Password must contain only letters,and common special characters.")
    ])
    new_password = PasswordField('new_password', [
        validators.Length(min=4, max=25, message="new password must be between 4 and 25 characters long."),
        validators.InputRequired(),
        validators.Regexp('^[a-zA-Z0-9!@#$%^&*_+-=]+$', message="Password must contain only letters,and common special characters.")
    ])
    submit = SubmitField('Submit')

class UserForm(Form):
    username = StringField('Username', [
        validators.Length(min=3, max=32),
        validators.InputRequired(),
        validators.Regexp('^[a-zA-Z]+( [a-zA-Z]+)*$', message="Username must contain only letters and no empty spaces.")
    ])
    # images = FieldList(StringField("Image"))
    submit = SubmitField('Submit')