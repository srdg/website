---
title:  "CerGen"
date:   2017-08-02 15:04:23
categories: [automation]
tags: [utility script, automation, hack]
---

This is basically a collection of scripts that automates the process of certificate generation
for various purposes. The only input required would be the template and the raw data (and optionally,
the directory in which the output files are to be stored).  
The idea is simple - you use the template as an image file, open it for every individual, process the file
and save it in some other name. Initially, I had used [Pillow](https://pypi.org/project/Pillow/) to handle the images and [openpyxl](https://pypi.org/project/openpyxl/) to retrieve data from the excel files.  
The most recent version of the scripts use command-line parsing using `argparse` to retrieve the inputs and instead of `.xlsx` file, I use `.csv` now - way easier and way less hassle.
The codes can be viewed on [Github](https://github.com/srdg/CerGen).