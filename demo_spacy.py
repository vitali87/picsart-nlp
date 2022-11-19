# Import spaCy
import spacy

# Create a blank English nlp object
nlp = spacy.blank("en")

doc = nlp("Hello world!")

# Index into the Doc to get a single Token
token = doc[1]

# Get the token text via the .text attribute
print(token.text)

# Iterate over tokens in a Doc
for token in doc:
    print(token.text)
