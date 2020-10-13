# public abstract
Deepmind's deep neural system, AlphaGo, successfully learned to play Go at a
high skill level. The system used a number of methods, including a CNN to
examine the game state, a reinforcement learning paradigm, and an MCMC tree
search to explore decisions. I suspected that there could be an effective 
*fully-neural* way to learn to play Go or other task that requires long-term 
decisions. My intuition was that the human brain is better modeled as using
neural networks, not MCMC search frameworks, and is highly effective at making
long-term decisions.

I propose an AlphaOne, a neural model that innovates on the Deepmind AlphaGo
and AlphaZero models. This model uses a very simple concept that I hypothesize
can make an enormous difference, with a core innovation on the experience
replay buffers that have made recent DeepRL systems reach unprecedented success.

This project is an implementation of the AlphaOne model experiment.
Until recently, I have not had a way to explain the rationale for the model,
so conceptual explanations are in the process of being written. However, the 
model itself is straightforward to implement, so I went ahead with that in the
meantime.

<<<<<<< HEAD
( edit: when I say "heroku" i meant "pachi". my brain was weird. )

=======
>>>>>>> 82f3eadf2300257fe749468955036156fa6f5857
# Result
As I've begun to learn, implementing machine learning models is sometimes 
impeded by bugs in source code or in design choices not intended to facilitate
specific iterations. In this project, I implemented the full network based
on Deepmind's source code until I ran into an issue with their Go playing
software, Heroku. Heroku does not innately support the maintenance of multiple
Go game instances at once. That worked fine for Deepmind's objectives. However,
AlphaOne requires maintaining many Go game states at once. This is not a 
demanding task (amounting to merely saving a state such as an epoch's gradient
to backpropagate or an experience to save in an experience replay buffer) in
theory, but due to Heroku's functionality, implementing AlphaOne using Heroku
would require re-running entire sequences of games in order to arrive at a
desired state. That is, the runtime complexity of running any forward
propagations is impaired by a factor linear to the number of steps needed to
reach that game state - meaning, it becomes increasingly difficult to explore
advanced game states, which functionally impairs the ultimate goal of the
network. (Also, there are issues with parallelism and more implementation
overhead due to this limitation of Heroku.) I searched for more Go programs
to integrate into the Deepmind-sourced pipeline, but ultimately, I had
become discouraged at the fact that this project would require substantially
more resources to complete than I had anticipated. So, I set down this 
side project until I was ready to revisit it, ideally by which time I had
worked out the theoretical basis for the project in the first place.

# internal comments for alpha-one, December 2017
Response to AlphaZero: is MCMC tree search necessary? Or can Go be solved without it?
Morgan Bryant, December 2017

12/8 1951: while tied convolution weights are probably a good idea in some aspect,
they are totally unnecessary for the current proof of concept.

Tip: in keras, to make shared weights, simply instantiate the layer object twice 
on the different input streams. 
See: <https://keras.io/getting-started/functional-api-guide/#shared-layers>


# Update: August 28 2018
Reattempting this. Trying to do it more from scratch since there was a fundamental error of some kind that prevented a direct adaptation of the original code.

branch: revamp_2
trying to keep the convention that prefixing a file with "rv2\_" indicates a new file from this era.
files: rv2\_search\_net.py: the reinforce and network part, sans the environment part.

TODO: characterize this error
TODO: identify [new?] source codes to use or how to adapt the existing code. 

note 00     The error had to do with the OpenAI Gym framework ~~

note 01 10/2/2018 0915: The error was due to gym's pachi interface to GO.
If I recall, the issue was that a game state cannot be directly reproduced.
It seems that maintaining a <<list?>> of the actions of each player would 

note 02 0924: Because of piece-taking, a direct algorithm would be a bit 
aggravating to implement, ...., but maybe not as aggravating than interfacing
both pachi and gym. hmm investigating pre-made minimal GO players...
Actually it looks pretty straightforward to implement
