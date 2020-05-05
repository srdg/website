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


| .. | .. | .. | `1` | 4 | 6 | 3 | 0 | 2 | 9 |
|---|---|---|---|---|---|---|---|---|---|

At this point, what we want to do is take this element, look back at the way we have covered till now, figure out its correct position, shift all elements to right and insert it there. So if you are at index _**i**_ in array _**A**_, after you are done with the iteration, the subarray _**A[1..i]**_ should be in the sorted state. That's basically it. Given the previous state of the array, after we are done with the iteration, it would look like:

| `1` | `5` | `7` | `8` | 4 | 6 | 3 | 0 | 2 | 9 |
|---|---|---|---|---|---|---|---|---|---|

Continue this all the way to the end, and by the time we reach there, if we look back, the entire array would be sorted.

## Bubble Sort
Let's review this next.
The idea is, irrespective of whichever index you are at, you start iteration from the first index. For every iteration over the entire array _**A[start..end]**_, you iterate again over a subarray _**A[start..end-i]**_, if you are at _i<sup>th</sup>_ index. 
Iterating over the arrays would somewhat go like:  

|5 |7 |8 |1 |4 |6 |3 |0 |2 |`9` |
|---|---|---|---|---|---|---|---|---|---|

|5 |7 |1 |4 |6 |3 |0 |2 |`8` |`9`|
|---|---|---|---|---|---|---|---|---|---|

|5 |1 |4 |6 |3 |0 |2 |`7` |`8` |`9` |
|---|---|---|---|---|---|---|---|---|---|

|1 |4 |5 |3 |0 |2 |`6` |`7` |`8` |`9` |
|---|---|---|---|---|---|---|---|---|---|

|1 |4 |3 |0 |2 |`5` |`6` |`7` |`8` |`9` |
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


## Merge Sort
Now that we've got the classic sorts out of the way, let's dig a little deeper. Both the previous approaches required us to stand at a location in the array and look over the rest of the subarray. Both of them would have the time complexity of O(n<sup>2</sup>), and this would be a problem when the input size grows huge. Isn't there an easier way to do this?

Well, by a lucky stroke of genius you'd probably have figured out by now that we are talking about merge and quick sort algorithms. They are quite fast as compared to the classic approaches. (I won't go into the complexity theory now, but just know that these are significantly faster than the sorting algos we just discussed.) The question is, what is that secret sauce that gives them this boost?
The answer lies in a simple strategy called "Divide-and-Conquer", and its not unlike the trademark strategies used during the periods of colonialization. The idea is to divide your huge problem into subproblems, and conquer them. Once you finish accumulating your results, you'd have solved the original problem you began with. (Think of it as like you're given a large bundle of sticks and asked to break it. You couldn't break all of them together, so you took a few sticks at a time and broke them. At the end you collected all the broken sticks and your problem is solved!)

Before we dive further, there is one thing we need to keep in mind.
> An element is trivially sorted. This means that if you have an array with only one element, say 10, it is already sorted - because there is no other element to compare it to.

Here, we get a golden opportunity. If the element is trivially sorted in itself, why not recursively break down our array into subparts? Like below:

| 5 | 7 | 8 | 1 | 4 |                 
|---|---|---|---|---|                 

| 8 | 1 | 4 |
|---|---|---|

| 8 |       
|---|  
and     
| 1 | 4 |
|---|---|

You get the point, right? (Note that I didn't show the complete array here).
And because each element is trivially sorted, we can merge them together as we go up, such that the merged arrays are sorted!
But here's the question, how will we merge? Let's think about it a bit.
**Take any instant of time. You have two sorted subarrays X and Y, of length p and q respectively (p≠q).**
Now if we have to merge them together, we need a new array, right? You can't just add stuff to any of the existing arrays, because _static arrays are immutable w.r.t. size by definition_.
Thus, we need a new array of size **p+q**. Let's name it Z. Now our task is to enter all the elements in X and Y in Z. In order to do that, we need to iterate over the elements X and Y (let's use variables _i_ and _j_ for them).
We look at the first element in both X and Y. Now, there are 3 possibilities : X[0]>Y[0], X[0]<Y[0], or X[0]=Y[0].
The first case, we add X[0] to Z, and increase the index pointer _i_ by 1.
The second case, we add Y[0] to Z, and increase the index pointer _j_ by 1.
The third case, we add both of them to Z, and increase both counters by 1.

Let us continue this for all the elements in X and Y. Sooner or later, one of the arrays will be exhausted of all elements, or both of the arrays will get exhausted at the same time.
If the second case happens, its perfect! Because we don't have to do extra work, and since we are always adding the smaller of the two elements, we can guarantee that the array Z is fully sorted.
Unfortunately, more often than not, the first case is the reality. What can we do here? Well, remember the assumption we started with - that the subarrays are already sorted? Since upto now we've always added the smaller elements in the two subarrays, we can add the rest of the elements that are left out in either X or Y.

Confusing? Here's an example. Say X=(3,27,38,43) and Y=(9,10,82). Initially _i_ points to 3 and _j_ points to 9.  
We compare _X[i] with Y[j]_. Since 3<9, we add Z[0]=3. _i_ now points to 27.  
We compare again. Since 27>9 (remember _i_ was changed but _j_ was not), we add 9 to Z. Z is now (3,9). We increase _j_ now, since we added an element from Y.  
Compare. _i_ points to 27, _j_ to 10. 27>10, so we add 10 to Z and increase _j_. Z=(3,9,10).  
Compare. _i_ still points 27, _j_ does to 82. 27<82, so we add 27 to Z and increase _i_. Z=(3,9,10,27).  
Compare. _i_ points to 38. 38<82. Increase _i_, Z=(3,9,10,27,38).  
Compare. _i_ points to 43<82. Add 43 to Z. Z is now (3,9,10,27,38,43). But we cannot increase _i_ anymore, because X is exhausted!
But did you see what happened? Because we were always adding the smaller of the two elements being indexed, we ensured that whichever part did not get added to Z by the time X finished, is the part we can simply take and append to current Z, because a) that subarray of Y is **already sorted** and b) all the elements in it are larger than the elements in Z.  
Do you see the benefit of having a sorted array now? Because we started with one, we can ensure that the elements left out are simply a sorted subarray (remaning Y) that has elements bigger than the current array(Z) we have! AND we knew that the elements that _i_ and _j_ are pointing to are definitely the lowest elements in the arrays X and Y.  
What would have happened if we started with unsorted subarrays? Well, for one thing, to find out the minimum element, you'd have to go over the entire second array (Y). And if X got exhausted, you'd have to iterate over entire Y everytime to find out the minimum element and append it to Z. (Can you see a faint resemblance to bubble sort here? For each element in primary array X, you have to iterate over the entire subarray Y!) 

Here's a picture for you to see it yourself.  

![Merge-Sort](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Merge_sort_algorithm_diagram.svg/660px-Merge_sort_algorithm_diagram.svg.png)
