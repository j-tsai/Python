import urllib
def read_text():
    quotes = open("/Users/tsai/Dropbox/movie_quotes.txt")
    content = quotes.read()
    print(content)
    quotes.close()
    check_curse(content)

def check_curse(text_check):
    connection = urllib.urlopen("http://www.wdyl.com/profanity?q="+text_check)
    output = connection.read()
    print(output)
    connection.close()
    if "true" in output:
        print("Alert")
    elif "false" in output:
        print("live a little")
    else:
        print("don't know what you are saying")
read_text()
