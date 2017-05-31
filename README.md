# Titanic-Competition

This was part of a kaggle.com competition and the results produced from this code can be found here: https://www.kaggle.com/sforsyth6/titanic

This code is my first attempt at machine learning. Within it I use a random forest algorithm to predict who would survive the titanic based on several features such as age, sex, class, and fare. I use a couple methods of feature selections such as adding up to fourth order polynomial terms and pruning features that don't meet a certain threshold. For missing data, I averaged the values of that feature and appeneded that value to it. 

The competition description is as follows:

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
