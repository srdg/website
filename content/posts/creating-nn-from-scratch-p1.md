---
title: Creating a neural network from scratch in Python 
subtitle: Part I
date: '2018-09-04'
markup: "mmark"
---
### What is this? Why am I here?
If you are a technology oriented person, you must've come across the buzz of neural networks or artificial intelligence by now. There are numerous courses available online that introduce you to neural networks and their workings. But most of them lack either in visualisation, or, if the visualisation was good, they lacked in implementation. It literally took me months to get all the pieces of information linked together. So here's the post, we'll start from the very basics and go deeper into all of it. By the end of this series, you'll not only have created a neural network entirely from scratch, but also understood the workflow and will be able to link them all together flawlessly. Let's get started!

### What are neurons?
A neuron is essentially a specialised cell that is capable of transmitting _nerve impulses_, which are basically electrochemical signals. If you've ever studied the nervous system, you'd know, a neuron somewhat looks like this:


![A single neuron](/images/Neuron.jpg)


[Source](https://no.wikipedia.org/wiki/Fil:Neuron.jpg)  
  
  
Neurons make routes to transmit the impulses between them by making connections with each other via the dendrites and axon terminals, like in the following clip.


![Neurons making connections](/images/Blog+NN+1.gif)[Source](https://www.reddit.com/r/educationalgifs/comments/8c5p1r/fetal_neurons_making_connections/)


  

Now, the term neural networks can be extremely misleading, conveying the idea that it has something to do with mimicking the human brain in a computer model. Truth is, scientists would probably throw a two-week long party if even a rat's brain is mimicked completely in a model - it'd be a tremendous breakthrough. The idea of neural networks is extremely simple, as you'll see just in a few minutes.

So why use the term neural networks at all, if it doesn't actually mimic the huge neural network in our brain? Its because, even though we haven't been able to mimic the brain entirely, the basic working methodology of the neurons in both cases is still the same. You see, in your brain, at any instant of time, a single neuron is either active (1) or inactive (0). See the binary pattern? Well, this is exactly how the artificial "neurons" in a neural network work. At a time, each neuron is either active or inactive.  A simple neural network looks like this-  


![A simple neural network](/images/simple-nn.png)  


You can compare the circles to the cell body containing the nucleus and all other things, and the long connecting lines as the axon. Here, we call the circles "nodes", and the connections "edges" [just like a Graph]. A single neuron takes in multiple inputs and produces a single output. Here, we call the neurons "perceptrons". To get deeper into the neural nets, its better to have an in-depth understanding of perceptrons first.  

### Perceptrons : Your entry point to the field of Neural Network
A single perceptron looks like this.


![A single perceptron](/images/perceptron.png)  


$$x_1, x_2, x_3$$ are the inputs to the perceptron. Associated with each edge (i.e. the lines) is a (random) number, which we call "weight" of that particular edge or connection _For the nerdier folks, this is kind of like a weighted graph - hope you get the similarity._
>  Intuitively, the weight of the connection from a particular input determines how much influence that input has on the output of the node being considered.

Let's get the above line clarified.

For instance, let's say the connection from $$x_1$$to the node has weight 0.67, that from $$x_2$$ to the node has weight 0.93 and that from $$x_3$$ has weight 0.02. Let us denote the weight associated with i-th connection by $$w_i$$. So, you can say that the input $$x_2$$ has the biggest influence on the nature of the output. Next comes $$x_1$$ and $$x_3$$, respectively. The way a perceptron determines the output is by computing the weighted sum $$\sigma (w_i * x_i)$$ and comparing that value against some threshold parameter. Put mathematically, it goes something like :
 
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtable columnalign="right center left" rowspacing="3pt" columnspacing="0 thickmathspace" displaystyle="true">
    <mlabeledtr>
      <mtd id="mjx-eqn-1">
        <mtext>(1)</mtext>
      </mtd>
      <mtd>
        <mstyle displaystyle="false" scriptlevel="0">
          <mtext>output</mtext>
        </mstyle>
      </mtd>
      <mtd>
        <mi></mi>
        <mo>=</mo>
      </mtd>
      <mtd>
        <mrow>
          <mo>{</mo>
          <mtable columnalign="left left" rowspacing="4pt" columnspacing="1em">
            <mtr>
              <mtd>
                <mn>0</mn>
              </mtd>
              <mtd>
                <mstyle displaystyle="false" scriptlevel="0">
                  <mtext>if&#xA0;</mtext>
                </mstyle>
                <munder>
                  <mo>&#x2211;<!-- ∑ --></mo>
                  <mi>j</mi>
                </munder>
                <msub>
                  <mi>w</mi>
                  <mi>j</mi>
                </msub>
                <msub>
                  <mi>x</mi>
                  <mi>j</mi>
                </msub>
                <mo>&#x2264;<!-- ≤ --></mo>
                <mstyle displaystyle="false" scriptlevel="0">
                  <mtext>&#xA0;threshold</mtext>
                </mstyle>
              </mtd>
            </mtr>
            <mtr>
              <mtd>
                <mn>1</mn>
              </mtd>
              <mtd>
                <mstyle displaystyle="false" scriptlevel="0">
                  <mtext>if&#xA0;</mtext>
                </mstyle>
                <munder>
                  <mo>&#x2211;<!-- ∑ --></mo>
                  <mi>j</mi>
                </munder>
                <msub>
                  <mi>w</mi>
                  <mi>j</mi>
                </msub>
                <msub>
                  <mi>x</mi>
                  <mi>j</mi>
                </msub>
                <mo>&gt;</mo>
                <mstyle displaystyle="false" scriptlevel="0">
                  <mtext>&#xA0;threshold</mtext>
                </mstyle>
              </mtd>
            </mtr>
          </mtable>
          <mo fence="true" stretchy="true" symmetric="true"></mo>
        </mrow>
      </mtd>
    </mlabeledtr>
  </mtable>
</math>

That is all there to perceptrons. Simple enough if you paid attention!

### Sigmoid Neurons : the real stuff
So, are the perceptrons the only kind of neurons available in the market? Of course not!! There are many kinds of neurons available - perceptron is the simplest of them all. Neurons are differentiated from one another by what we call "activation functions" - which you'll understand now.

First, why would the perceptrons not work in all cases? You see, the output of a perceptron depends directly on the input values. Something like

$$Output = F(input, values, weights)$$

where F is a weighted product function that computes sum of elementwise products. It turns out, when you use such a function to determine the output of a perceptron, a small change in any one input value may cause the output to change drastically, sometimes even flip from 0 to 1 or vice versa (remember the outputs of a perceptron?). That really isn't a stable way to determine outputs, right?

So, what is the workaround? Well, instead of directly treating the sum of elementwise products as the output, we pass it through another function that limits its output value, and ensures that a small change in the inputs causes only a small change in the output. This another function that we pass it through - this is called the activation function.

There are many activation functions available, but the most common of them all is the sigmoid activation function. Put mathematically, the sigmoid function looks like this :

$$\begin{eqnarray} 
  \sigma(z) \equiv \frac{1}{1+e^{-z}}.
\end{eqnarray}$$

and when plotted, gives a curve like this :


![Logistic curve](/images/320px-Logistic-curve.svg.png)



The range of the output value is from 0 to 1 - this is the way a sigmoid function limits the output value. While in perceptrons we had the outputs 0 or 1, in case of sigmoid neurons it can be anything between 0 to 1. Plugging in the Output from equation (1) in the sigmoid function, we get the output as

$$\begin{eqnarray} 
  output = \frac{1}{1+\exp(-\sum_j w_j x_j-b)}.
\end{eqnarray}$$

What is `b` here? It is the threshold parameter we mentioned above, also stands for bias. And `exp` is just the exponent function. 

>    So, we get the input values, multiply them with their associated weight in the edges, sum all of the products up, add the bias (threshold) and then pass the net sum through an activation function to get the output value from one particular node.

Pretty simple, right?
Don't worry, even if you don't understand it now completely, you'll get it once you've finished the series!

### Bonus: more on activation functions

Sigmoid functions are not the only kind of activation functions used. There are multiple other kinds of activation functions used - such as `ReLU` (Rectified Linear Units), `tanh`, `leaky ReLU`, `softmax`... the list goes on and on. If you want to know more about the various kinds of activation functions - here's a [great blog article](https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html) to give a read.

Now, you've got the gist of all the basic building blocks of a neural network. Next in line is how they all work together, and how in the world do neural networks recognise things like handwritten digits and images (and many other things) by such simple mathematical functions?

We'll explore this missing link in the next post. Stay tuned!