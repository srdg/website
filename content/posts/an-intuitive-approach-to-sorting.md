---
layout: post
published: true
title: An intuitive approach to sorting
date: '2020-05-05'
subtitle: Simplifying algorithms
tags:
 - algorithms
 - data structures
categories:
 - algorithms
---

This post is about getting an intuitive approach to a few popular sorting algorithms. Rather than diving headfirst into the algorithms, we'll reach to it from another way. Let's get started.

| 5 | 7 | 8 | 1 | 4 | 6 | 3 | 0 | 2 | 9 |
|---|---|---|---|---|---|---|---|---|---|

Suppose, you want to sort this array. Now, there are a handful of techniques available in the market to solve them. For us, let's review the very popular algorithms called Insertion and Bubble Sort first.

## Insertion Sort
If you ask me, what really is an insertion sort? The basic idea could be summarized as follows:  
Say you are standing at a certain point in the array.


| .. | .. | .. | 1 | 4 | 6 | 3 | 0 | 2 | 9 |
|---|---|---|---|---|---|---|---|---|---|

At this point, what we want to do is take this element, look back at the way we have covered till now, figure out its correct position, shift all elements to right and insert it there. So if you are at index _**i**_ in array _**A**_, after you are done with the iteration, the subarray _**A[1..i]**_ should be in the sorted state. That's basically it. Given the previous state of the array, after we are done with the iteration, it would look like:

| 1 | 5 | 7 | 8 | 4 | 6 | 3 | 0 | 2 | 9 |
|---|---|---|---|---|---|---|---|---|---|

Continue this all the way to the end, and by the time we reach there, if we look back, the entire array would be sorted.

## Bubble Sort
Let's review this next.
The idea is, irrespective of whichever index you are at, you start iteration from the first index. For every iteration over the entire array _**A[start..end]**_, you iterate again over a subarray _**A[start..end-i]**_, if you are at _i'th_ index. 
Iterating over the arrays would somewhat go like:  

|5 |7 |8 |1 |4 |6 |3 |0 |2 |9 |
|---|---|---|---|---|---|---|---|---|---|

|5 |7 |1 |4 |6 |3 |0 |2 |8 |9 |
|---|---|---|---|---|---|---|---|---|---|

|5 |1 |4 |6 |3 |0 |2 |7 |8 |9 |
|---|---|---|---|---|---|---|---|---|---|

|1 |4 |5 |3 |0 |2 |6 |7 |8 |9 |
|---|---|---|---|---|---|---|---|---|---|

|1 |4 |3 |0 |2 |5 |6 |7 |8 |9 |
|---|---|---|---|---|---|---|---|---|---|

and so on. But why are we iterating till _**(end-i)**_ index? If you look closely, you will see that with every iteration, the subarray _**A[end-i..end]**_ is sorted. How is this happening? If you look at the algo, it goes like:

```
for i:=start to end
 for j:=start to end
  if ( A[j] > A[j+1] ) {
   swap ( A[j], A[j+1] )
  }
```
So for every inner iteration, you are comparing the neighbouring elements. Naturally, if the biggest element is somewhere in the beginning, it would percolate all the way through to the last position of the array, because it will always be greater than its neighbouring element. Similarly, the second largest element will percolate through till it reaches the largest element at the last, where it will stop. Continue this for all the elements of the array, and by the time we are finished, we can say that the array will be sorted.
