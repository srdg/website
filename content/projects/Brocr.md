---
title:  "Brocr"
date:   2018-12-23 15:04:23
categories: [computer vision]
tags: [OCR, tesseract,webapp]
---


A browser based OCR engine working with a python wrapper of [Tesseract](https://github.com/tesseract-ocr/tesseract/wiki) in the backend. The 
front end is rendered using Flask. The latest version of Tesseract is at 
play here and it gives surprisingly good outputs with a few preprocessing 
steps. It's still a work in progress -- I haven't yet figured out why my 
trained files (`.traineddata`) get automatically deleted from the server save the default one
([view the detailed question in stackoverflow](https://stackoverflow.com/q/53096796/8507120).)  

Once I figure that out, I'll add more language supports and choices of preprocessing.  
You can view the current status of the work on [heroku](http://brocr.herokuapp.com).

Warning: This will break if you select the `Bengali` option from the dropdown.