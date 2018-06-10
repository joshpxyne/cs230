# CNN in TensorFlow

Based off of the official Tensorflow MNIST model.

[tf.data](https://www.tensorflow.org/api_docs/python/tf/data),
[tf.estimator.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator),
[tf.layers](https://www.tensorflow.org/api_docs/python/tf/layers)


## Setup

To begin, you'll simply need the latest version of TensorFlow installed.
Then to train the model, run the following:

```
python cnn.py
```

The model will begin training and will automatically evaluate itself on the
validation data.

Illustrative unit tests and benchmarks can be run with:

```
python cnn_test.py
python cnn_test.py --benchmarks=.
```

## Exporting the model

You can export the model into Tensorflow [SavedModel](https://www.tensorflow.org/programmers_guide/saved_model) format by using the argument `--export_dir`:

```
python cnn.py --export_dir ./vs
```

The SavedModel will be saved in a timestamped directory under `./vs` (e.g. `./vs/1513630966/`).

**Getting predictions with SavedModel**
Use [`saved_model_cli`](https://www.tensorflow.org/programmers_guide/saved_model#cli_to_inspect_and_execute_savedmodel) to inspect and execute the SavedModel.

```
saved_model_cli run --dir ./vs/<timestamp> --tag_set serve --signature_def classify --inputs image=<image.npy>
```

To convert your own image(s) to this format, run the following command with your images in this directory:

```
python npcon.py --image <"filename(s)"> --output <"filename(s)"> --batch <True/False>
```


The output should look similar to below:
```
DN0a22b616:predict-cnn joshpayne1$ saved_model_cli run --dir ./vs/1518944297 --tag_set serve --signature_def classify --inputs image=images0.npy
2018-02-18 02:19:48.188603: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.2 AVX AVX2 FMA
Result for output key classes:
[22]
Result for output key probabilities:
[[  3.29756364e-15   0.00000000e+00   6.79246181e-15   4.03374295e-16
    2.42251444e-18   1.01116413e-12   3.60628519e-12   0.00000000e+00
    2.15181956e-07   1.01725094e-10   4.70693721e-16   2.39584679e-05
    7.23152370e-32   3.33596926e-19   6.36450947e-10   5.00806389e-23
    2.87304880e-12   2.21608163e-14   0.00000000e+00   3.28333207e-36
    2.62940423e-14   0.00000000e+00   9.99950409e-01 < 1.37660379e-07
    6.71888345e-16   6.79457653e-18   7.18531179e-10   2.12818954e-08
    0.00000000e+00   1.17662371e-19   2.56622176e-18   2.52274021e-05
    1.03750249e-21   3.21115301e-37   1.17415666e-13]]
```

After 12 epochs our augmented 35 class dataset (2000 instances of each class), evaluation accuracy was around ~96%. Predictions were generally good, but some weren't. Likely needs a better dataset. 

(From me in the future: it definitely needed a better dataset! You overfit your training set as well.)
