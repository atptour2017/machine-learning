from bayes import *
import feedparser

ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')

vocabList,pSF,pNY=localWords(ny,sf)

