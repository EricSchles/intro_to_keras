# Understanding Keras

The Keras library is pretty amazing.  It does a pretty good job of wrapping an admittedly hard type of model - a neural network, and makes it pretty easy.  That being said, there documentation, as with most technical documentations, don't explain everything you need to know.

So!  I thought I'd fill the gap with a very neat little example of how to use Keras.

## About the data

The code with an example dataset is in this repo, so you can try it yourself.  Lots of examples work off of datasets that come with libraries, however I find that often in the real world we work with csvs or database tables, so I thought it'd be more realistic to use a csv as my dataset.

The specific data I'll be using is the seattle crime data.  Which I downloaded here:

[https://data.seattle.gov/Public-Safety/Seattle-Police-Department-911-Incident-Response/3k2p-39jp](https://data.seattle.gov/Public-Safety/Seattle-Police-Department-911-Incident-Response/3k2p-39jp)

You'll need to download the dataset yourself if you want to replicate my results.  Unfortunately github won't let me put in anything larger than 100 MB (which I totally understand and don't have a problem with).  But it means you'll have to grab it yourself.  It's 364ish MB unzipped so not that big.  But too big for github.  

This dataset is great!  Although lacking a data dictionary, a common problem with most open data sources.  Although the field names are very good and intuitive for the most part, so it's still fairly easy to work with.

## What we will be trying to do with this basic example

In this basic example we'll be taking a look at how to set up and run a basic neural network.  This set up is not optimizing for anything (in terms of model accuracy).  We are only trying to understand how to get one of the Keras models to run and to explain why it succeeds.

So, let's get into it!

We'll be doing a classification problem, so we'll need:

```
from keras.models import Sequential
from keras.layers import Dense, Dropout
```

Now let's get our data preprocessed:

```
df = df[pd.notnull(df["Event Clearance Group"])]
df = df[pd.notnull(df["Zone/Beat"])]

zone_beat = one_hot_encoding(df["Zone/Beat"])
event_clearance_group = one_hot_encoding(df["Event Clearance Group"])
```

The first thing I do here is get rid of any NaN rows with NaN values.  These won't yield any semantic value so there is no reason to include those rows.

Next I do something called one hot encoding.  Basically what this means is create a bunch of columns that are 1 for the value in the corresponding column and zero every where else.

If this explanation didn't make sense, check out this quora question that explains one hot encoding with a diagram: [https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science](https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science)

Okay, now that we have our data in a suitable form, let's go ahead and try to train a neural network!

```
# create model
basic_model = Sequential()
basic_model.add(Dense(100, input_shape=(90,), activation='relu', name="dense_1"))
basic_model.add(Dense(44, input_shape=(100,), activation="sigmoid", name="dense_4"))
basic_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
basic_model.fit(zone_beat, event_clearance_group, epochs=1, verbose=1)
```

Explanation:

`basic_model.add(Dense(100, input_shape=(90,), activation='relu', name="dense_1"))`

- input_shape here refers to the number of columns in the X data or the input data and these numbers need to link up, the first parameter can be whatever you want.


`basic_model.add(Dense(44, input_shape=(100,), activation="sigmoid", name="dense_4"))`

- Here the first parameter matters, it has to match up with the number of columns of the Y data or the output data.  The input_shape here only has to correspond to the output (first parameter) of the previous Dense so any internal or hidden layers can be whatever you want.


`basic_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])`

- the loss is categorical_cross_entropy to indicate that this is a multi output classification problem.  The optimizer could be adam or sgd or whatever.  metrics refers to what is being optimized for.  In this case accuracy.


`basic_model.fit(zone_beat, event_clearance_group, epochs=1, verbose=1)`

- zone_beat is the X data (or input data), event_clearance_group is the Y data or the output data.


We can double check our results with the more_complete.py which is also in this folder :D  Happy coding!


