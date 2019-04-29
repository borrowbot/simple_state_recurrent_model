This project is an implementation of an ARMA(1, n)-style, linear, recurrent, auto-regressive model that I have recently written about on my [blog](https://frankwang95.github.io/2019/04/simple_linear_recurrent_model). This model is able to extract parts of strings which represent simple semantic entities, such as references to monetary values or dates:

<img src="https://github.com/borrowbot/simple_state_recurrent_model/raw/master/readme_resources/example_inference.png">
<!-- green color: #a1e579, grey color: #474747 -->

Our model is simple, fast, interpretable, easy to train, and data-efficient. The performance is good enough for many basic applications and serves as a good substitute for otherwise unwieldy and complex heuristic parsing rules.

This repository will also contain published, pre-trained models for general-purpose tasks which may be of interested to developers at large. As of now, I am only publishing models which can detect references to monetary amounts though models for other tasks are in development and should be published soon.

This project is published under the permissive Apache Software License.


# Installation

This package is not yet on PyPI. Installation is manual:

1. Clone the repository from github and cd into the project's root directory:
    ```
    git clone https://github.com/borrowbot/simple_state_recurrent_model.git
    cd simple_state_recurrent_model
    ```
2. Install dependency packages:
    ```
    pip install -r requirements.txt
    ```
3. Install the package using pip:
    ```
    pip install .
    ```

# Packaged Models

For those only interested in making use of the packaged models, usage is incredibly simple, as you can see in the following example:

```python
>>> from money_detection_model import run_inference
>>> run_inference("test string $500")

[0.00010524255918530615,
 0.000116869978677247,
 0.0004632890110024569,
 0.00013824601794592935,
 1.209167166778495e-06,
 2.3163583768053688e-05,
 0.00031484312450876786,
 0.00011220331645314339,
 0.0005458306738767712,
 0.0006814267183881327,
 0.00021843669431039945,
 0.00017079662128961727,
 0.002743222863569017,
 0.9999439462468341,
 0.9997941770844324,
 0.9776328761143998]
```

**A note on security:** For those working in sensitive environments, it is worth noting that loading serialized models involves pickling and unpickling objects. This process can be made to execute arbitrary code if a malicious file is injected into the repository's storage for packaged models.


# Training Custom Models

I have written a detailed explanation of the model implemented here on my [blog](https://frankwang95.github.io/2019/04/simple_linear_recurrent_model). Before training a custom model, I suggest going over this documentation in some detail so that it is clear what the different tunable parameters of the model represent.

<img src="https://raw.githubusercontent.com/borrowbot/simple_state_recurrent_model/master/readme_resources/model_diagram.png">

One important detail left out of the theoretical introduction is an explanation for the data format that our implementation expects:

```python
>>> input_data = [
  "input string 1 with semantic label $300",
  "semantic label $8700 and $34 in input string 2",
]
>>> label_data = [
  [[36, 39]],
  [[16, 20], [26, 28]]
]
```

The input strings are fairly self-explanatory. The label `l` for each string `s` should be a list of length-two lists such that `[s[l[i][0]:l[i][1]] for i in range(len(l))]` gives the completed collection of semantic entities for string `s`.

Given data in this form, training the model might look something like this:

```python
>>> from simple_state_recurrent_model.model import SimpleRecurrentModel
>>>
>>> alphabet = "abcdefghijklmnopqrstuvwxyz0123456789$?()"
>>> preprocess_fn = lambda x: x.lower()
>>>
>>> model = SimpleRecurrentModel(alphabet=alphabet, window_size=5, preprocess=preprocess_fn)
>>> model.train(input_data, label_data, 1, 0.1, 10)
```

The `alphabet` parameter expresses the set of characters which the model will recognize. Characters not found in `alphabet`, are lumped together by the model and represented by an encoded 'unknown character'. The `preprocess` parameter allows you to pass in a `str -> str` function which is applied to all inputs in both training and inference. Finally, the `window_size` `window_shift` parameters allow you to control the size and position of the window of characters used in each character-level prediction.

Once the model is trained to your satisfaction, it can be serialized using the `save` method. This model can later be restored with the `load_saved_model` function found in `simple_state_recurrent_model.model`.

Finally, we also provide an implementation of evaluation code which can generate information about model performance using leave-one-out cross-validation. This is still a work in progress - documentation for this feature will be forthcoming.
