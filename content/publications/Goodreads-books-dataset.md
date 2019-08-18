---
layout : post
comments : false
date : "2019-08-18T19:59:47+05:30"
draft : true
title : "Goodreads books dataset"
description : "A comprehensive list of all books listed in goodreads."
subtitle: "Among the hottest datasets in Kaggle"
categories : [dataset]
tags : [dataset, kaggle, EDA, clean data, books dataset, exploratory data analysis, recommendation engine]
---

### Context
The primary reason for creating this dataset is the requirement of a good clean dataset of books. Being a bookie myself (see what I did there?) I had searched for datasets on books in kaggle itself - and I found out that while most of the datasets had a good amount of books listed, there were either 
 + major columns missing, or  
 + grossly unclean data.  
 I mean, you can't determine how good a book is just from a few text reviews, come on! What I needed were numbers, solid integers and floats that say how many people liked the book or hated it, how much did they like it, and stuff like that. Even [the good dataset](https://www.kaggle.com/zygmunt/goodbooks-10k#books.csv) that I found was well-cleaned, it had a number of interlinked files, which increased the hassle. This prompted me to use the Goodreads API to get a well-cleaned dataset, with the promising features only ( minus the redundant ones ), and the result is the dataset you're at now.

### Acknowledgements
This data was entirely scraped via the [Goodreads API](https://goodreads.com/api), so kudos to them for providing such a simple interface to scrape their database.

### Inspiration
The reason behind creating this dataset is pretty straightforward, I'm listing the books for all book-lovers out there, irrespective of the language and publication and all of that. So go ahead and use it to your liking, find out what book you should be reading next ( there are very few free content recommendation systems that suggest books last I checked ), what are the details of every book you have read, create a word cloud from the books you want to read - all possible approaches to exploring this dataset are welcome. I started creating this dataset on May 25, 2019, and intend to update it frequently. P.S. If you like this, please don't forget to give an upvote!

### Details about the features
Here is a short description of the features this dataset includes.

+ bookID: A unique Identification number for each book.
+ title: The name under which the book was published.
+ authors: Names of the authors of the book. Multiple authors are delimited with -.
+ average_rating: The average rating of the book received in total.
+ isbn: Another unique number to identify the book, the International Standard Book Number.
+ isbn13: A 13-digit ISBN to identify the book, instead of the standard 11-digit ISBN.
+ language_code: Helps understand what is the primary language of the book. For instance, eng is standard for English.
+ num_pages: Number of pages the book contains.
+ ratings_count: Total number of ratings the book received.
+ text_reviews_count: Total number of written text reviews the book received.
