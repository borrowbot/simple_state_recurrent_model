This project is a implementation of a ARMA(1, n)-style, linear, recurrent, auto-regressive model that I have recently written about on my blog. This model is able to extract parts of strings which represent simple semantic entities, such as references monetary values or dates:

<img src="https://github.com/borrowbot/simple_state_recurrent_model/raw/master/readme_resources/example_inference.png">

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

**A note on security:** For those working in sensitive environments, it is worth noting that loading serialized models involves picking and unpicking objects. This process can be made to execute arbitrary code if a malicious file is injected into the repository's storage for packaged models.


# Data Format
