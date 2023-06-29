# Japanese text scraping processing cluster analysis
# Web scraping

I used Selenium’s ChromeDriver to automatically access a Japanese technology blog by providing the blog's webpage URL to the driver.  
To get each article’s text data, I followed the steps below using the Beautiful soup library:
- collect all the URLs of the buttons that are used to navigate to the next page of the blog
- collect all the URLs of the articles from every page by inspecting the location of the URLs in the HTML tags in advance
- remove tags storing unneeded information for the cluster analysis, such as page metadata, code scripts, etc.
- using regex, remove URLs that lead to external websites, digits, strings with certain patterns common to every article (e.g., dates, blog category), English and Japanese punctuation marks, etc.    

Finally, I stored a total of 54 article URLs and their corresponding preprocessed texts into a pandas dataframe. 
 
# Cluster analysis 

To implement topic modeling, I did the following steps:
- made a function that can tokenize article’s text using the ‘sudachipy’ module while also removing common Japanese language stop words as well as the words written in hiragana
- made a function that returns topics for each article using the ‘gensim’ library
In this case, I chose to limit the number of topics to 1 since adding more topics did not result in more distinct words within the topics.
