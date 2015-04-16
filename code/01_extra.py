'''
Multi-line comments go between 3 quotation marks.
You can use single or double quotes.
'''

# One-line comments are preceded by the pound symbol


# BASIC DATA TYPES

x = 5               # creates an object
print type(x)       # check the type: int (not declared explicitly)
type(x)             # automatically prints
type(5)             # assigning it to a variable is not required

type(5.0)           # float
type('five')        # str
type(True)          # bool


# LISTS

nums = [5, 5.0, 'five']     # multiple data types
nums                        # print the list
type(nums)                  # check the type: list
len(nums)                   # check the length: 3
nums[0]                     # print first element
nums[0] = 6                 # replace a list element

nums.append(7)              # list 'method' that modifies the list
help(nums.append)           # help on this method
help(nums)                  # help on a list object
nums.remove('five')         # another list method

sorted(nums)                # 'function' that does not modify the list
nums                        # it was not affected
nums = sorted(nums)         # overwrite the original list
sorted(nums, reverse=True)  # optional argument


# FUNCTIONS

def give_me_five():         # function definition ends with colon
    return 5                # indentation required for function body

give_me_five()              # prints the return value (5)
num = give_me_five()        # assigns return value to a variable, doesn't print it


import random               # Import used to being in existing packages
random.choice(nums)         # Use a period to seperate module from function
random.randint(0, 100)      # Another function! Get a random integer between two given ones

for item in nums:           # A loop will go through each element of a list
    print item


# powerful modules for getting data from the internet
import requests
from bs4 import BeautifulSoup

# Use requests to go to a website
r = requests.get("http://en.wikipedia.org/wiki/Data_science")
# BeautifulSoup will read the HTML from that website
b = BeautifulSoup(r.text)
# get the title of the page
title = b.find('title').text
title
# get each paragraph
paragraphs = b.find("body").findAll("p")
text = ""
for paragraph in paragraphs: # I am looping through each paragraph in the HTML!
    text += paragraph.text + " " # adding the text to a variable called text.


# Data Science corpus
text

# very powerful natural language toolkit
import nltk
# tokenize into sentences
sentences = [sent for sent in nltk.sent_tokenize(text)]
sentences[:10]

# tokenize into words
tokens = [word for word in nltk.word_tokenize(text)]
tokens[:100]



# Use an API to do some sentiment analysis
import json

# Sample sentences
sentences = ['I love Sinan!', 'I hate Sinan!', 'I feel nothing about Sinan!']

# API endpoint (i.e.the URL they ask you to send your text to)
url = 'http://www.datasciencetoolkit.org/text2sentiment/'

# Loop through the sentences
for sentence in sentences:
    payload = {'text': sentence} # The sentence we want the sentiment of 
    headers = {'content-type': 'application/json'} # The type of data you are sending
    r = requests.post(url, data=json.dumps(payload), headers=headers) # Send the data
    print sentence, json.loads(r.text)['score'] # Print the results
