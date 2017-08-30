# Machine Learning for iOS 

**Last Update: August 30, 2017.**

Curated list of resources for iOS developers in following topics: 

- [Core ML](#coreml)
- [Machine Learning Libraries](#gpmll)
- [Deep Learning Libraries](#dll)
	- [Deep Learning: Model Compression](#dlmc)
- [Computer Vision](#cv)
- [Natural Language Processing](#nlp)
- [Speech Recognition (TTS) and Generation (STT)](#tts)
- [Text Recognition (OCR)](#ocr)
- [Other AI](#ai)
- [Machine Learning Web APIs](#web)
- [Opensource ML Applications](#mlapps)
- [Game AI](#gameai)
- Other related staff
	- [Linear algebra](#la)
	- [Statistics, random numbers](#stat)
	- [Mathematical optimization](#mo)
	- [Feature extraction](#fe)
	- [Data Visualization](#dv)
	- [Bioinformatics (kinda)](#bio)
	- [Big Data (not really)](#bd)
- [iOS ML Blogs](#blogs)
- [Mobile ML books](#books)
- [GPU Computing Blogs](#gpublogs)
- [Learn Machine Learning](#learn)
- [Other Lists](#lists)

Most of the de-facto standard tools in domains listed above are written in iOS-unfriendly languages (Python/Java/R/Matlab) so finding something appropriate for your iOS application may be a challenging task.

This list consists mainly of libraries written in Objective-C, Swift, C, C++, JavaScript and some other languages if they can be easily ported to iOS. Also, links to some relevant web APIs, blog posts, videos and learning materials included.

Resources are sorted alphabetically or randomly. The order doesn't reflect my personal preferences or anything else. Some of the resources are awesome, some are great, some are fun, and some can serve as an inspiration.

Have fun!

**Pull-requests are welcome [here](https://github.com/alexsosn/iOS_ML)**.

# <a name="coreml"/>Core ML

* [coremltools](https://pypi.python.org/pypi/coremltools) is a Python package. It contains converters from some popular machine learning libraries to the Apple format.
* [Core ML](https://developer.apple.com/documentation/coreml) is an Apple framework to run inference on device. It is highly optimized to Apple hardware.

Currently CoreML is compatible (partially) with the following machine learning packages:

- [Caffe](http://caffe.berkeleyvision.org)
- [Keras](https://keras.io/)
- [libSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- [scikit-learn](http://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)

Third-party converters available for:
- [MXNet](https://github.com/apache/incubator-mxnet/tree/master/tools/coreml)

There are many curated lists of pre-trained neural networks in Core ML format: [\[1\]](https://github.com/SwiftBrain/awesome-CoreML-models), [\[2\]](https://github.com/cocoa-ai/ModelZoo), [\[3\]](https://github.com/likedan/Awesome-CoreML-Models).

Core ML currently doesn't support training models, but still, you can replace model by downloading a new one from a server in runtime. [Here is a demo](https://github.com/zedge/DynamicCoreML) of how to do it. It uses generator part of MNIST GAN as Core ML model.

# <a name="gpmll"/>General-Purpose Machine Learning Libraries

<table>
  <tr>
    <th>Library</th>
    <th>Algorithms</th> 
    <th>Language</th> 
    <th>License</th>
    <th>Code</th>
    <th>Dependency manager</th>
  </tr>
    <tr>
    <td><a href="https://github.com/KevinCoble/AIToolbox">AIToolbox</a></td>
    <td>
<ul>
<li>Graphs/Trees</li><ul>
    <li>Depth-first search</li>
    <li>Breadth-first search</li>
    <li>Hill-climb search</li>
    <li>Beam Search</li>
    <li>Optimal Path search</li></ul>
<li>Alpha-Beta (game tree)</li>
<li>Genetic Algorithms</li>
<li>Constraint Propogation</li>
<li>Linear Regression</li>
<li>Non-Linear Regression</li><ul>
    <li>parameter-delta</li>
    <li>Gradient-Descent</li>
    <li>Gauss-Newton</li></ul>
<li>Logistic Regression</li>
<li>Neural Networks</li><ul>
    <li>multiple layers, several non-linearity models</li>
    <li>on-line and batch training</li>
    <li>feed-forward or simple recurrent layers can be mixed in one network</li>
    <li>LSTM network layer implemented - needs more testing</li>
    <li>gradient check routines</li></ul>
<li>Support Vector Machine</li>
<li>K-Means</li>
<li>Principal Component Analysis</li>
<li>Markov Decision Process</li><ul>
    <li>Monte-Carlo (every-visit, and first-visit)</li>
    <li>SARSA</li></ul>
<li>Single and Multivariate Gaussians</li>
<li>Mixture Of Gaussians</li>
<li>Model validation</li>
<li>Deep-Network</li><ul>
    <li>Convolution layers</li>
    <li>Pooling layers</li>
    <li>Fully-connected NN layers</li></ul>
</ul>
</td> 
    <td>Swift</td> 
    <td>Apache 2.0</td>
    <td><a href="https://github.com/KevinCoble/AIToolbox">GitHub</a> </td>
    <td> </td>
  </tr>
  <tr>
    <td><a href="http://dlib.net/"><img width=100 src="http://dlib.net/dlib-logo.png"><br>dlib</a></td>
    <td><ul>
<li>Deep Learning</li><li>Conventional SMO based Support Vector Machines for classification and  regression</li><li>Reduced-rank methods for large-scale classification and regression</li><li>Relevance vector machines for classification and regression</li><li>A Multiclass SVM</li><li>Structural SVM<li>A large-scale SVM-Rank</li><li>An online kernel RLS regression</li><li>An online SVM classification algorithm</li><li>Semidefinite Metric Learning</li><li>An online kernelized centroid estimator/novelty detector and offline support vector one-class classification</li><li>Clustering algorithms: linear or kernel k-means, Chinese Whispers, and Newman clustering</li><li>Radial Basis Function Networks</li><li>Multi layer perceptrons</li></ul>
</td> 
    <td>C++</td> 
    <td>Boost</td>
    <td><a href="https://github.com/davisking/dlib">GitHub</a></td>
    <td><a href="https://cocoapods.org/pods/FANN">Cocoa Pods</a></td>
  </tr>
  <tr>
    <td><a href="http://leenissen.dk/fann/wp/">FANN</a></td>
    <td><ul>
<li>Multilayer Artificial Neural Network</li>
<li>Backpropagation (RPROP, Quickprop, Batch, Incremental)</li>
<li>Evolving topology training</li>
</ul></td> 
    <td>C++</td> 
    <td>GNU LGPL 2.1</td>
    <td><a href="https://github.com/libfann/fann">GitHub</a></td>
    <td><a href="https://cocoapods.org/pods/FANN">Cocoa Pods</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/lemire/lbimproved">lbimproved</a></td>
    <td>k-nearest neighbors and Dynamic Time Warping</td> 
    <td>C++</td> 
    <td>Apache 2.0</td>
    <td><a href="https://github.com/lemire/lbimproved">GitHub</a> </td>
    <td> </td>
  </tr>
    <tr>
    <td><a href="https://github.com/gianlucabertani/MAChineLearning">MAChineLearning</a></td>
    <td>
    <ul><li>Neural Networks</li><ul>
<li>Activation functions: Linear, ReLU, Step, sigmoid, TanH</li>
<li>Cost functions: Squared error, Cross entropy</li>
<li>Backpropagation: Standard, Resilient (a.k.a. RPROP).</li>
<li>Training by sample or by batch.</li>
</ul>
<li>Bag of Words</li>
<li>Word Vectors</li></ul>
</td> 
    <td>Objective-C</td> 
    <td>BSD 3-clause</td>
    <td><a href="https://github.com/gianlucabertani/MAChineLearning">GitHub</a> </td>
    <td> </td>
  </tr>
    
  <tr>
    <td><a href="https://github.com/Somnibyte/MLKit"><img width=100 src="https://github.com/Somnibyte/MLKit/raw/master/MLKitSmallerLogo.png"><br>MLKit</a></td>
    <td> Matrix and Vector Operations (uses Upsurge framework)
 Simple Linear Regression (Allows for 1 feature set)
 Polynomial Regression (Allows for multiple features)
 Ridge Regression
 Multi-Layer Feed Forward Neural Network
 K-Means Clustering
 Genetic Algorithms
 Allows for splitting your data into training, validation, and test sets.
 K-Fold Cross Validation & Ability to test various L2 penalties for Ridge Regression
 Single Layer Perceptron, Multi-Layer Perceptron, & Adaline ANN Architectures
</td> 
    <td>Swift</td>
    <td>MIT</td>
    <td><a href="https://github.com/Somnibyte/MLKit">GitHub</a></td>
    <td><a href="https://cocoapods.org/pods/MachineLearningKit">Cocoa Pods</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/saniul/Mendel"><img width=100 src="https://github.com/saniul/Mendel/raw/master/logo@2x.png"><br>Mendel</a></td>
    <td>Evolutionary/genetic algorithms</td> 
    <td>Swift</td> 
    <td>?</td>
    <td><a href="https://github.com/saniul/Mendel">GitHub</a></td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://github.com/vincentherrmann/multilinear-math">multilinear-math</a></td>
    <td>Swift wrappers of many important functions from the Accelerate framework and LAPACK (vector summation, addition, substraction, matrix and elementwise multiplication, division, matrix inverse, pseudo inverse, eigendecomposition, singular value decomposition...), MultidimensionData protocol for elegant handling of multidimensional data of any kind, Clear, compact and powerful syntax for mathematical operations on tensors, Principal component analysis, Multilinear subspace learning algorithms for dimensionality reduction, Linear and logistic regression, Stochastic gradient descent, Feedforward neural networks, Sigmoid, ReLU, Softplus activation functions, Easy regularizations</td> 
    <td>Swift</td> 
    <td>Apache 2.0</td>
    <td><a href="https://github.com/vincentherrmann/multilinear-math">GitHub</a> </td>
    <td>Swift Package Manager</td>
  </tr>
  <tr>
    <td><a href="http://opencv.org/"><img width=100 src="http://opencv.org/assets/theme/logo.png">OpenCV</a></td>
    <td>Multi-Layer Perceptrons, Boosted tree classifier, decision tree, Expectation Maximization, K-Nearest Neighbors, Logistic Regression, Bayes classifier, random forest, Support Vector Machines,  Stochastic Gradient Descent SVM classifier, grid search, hierarchical k-means, deep neural networks</td> 
    <td>C++</td> 
    <td>3-clause BSD</td>
    <td><a href="https://github.com/opencv">GitHub</a> </td>
    <td> <a href="https://cocoapods.org/pods/OpenCV">Cocoa Pods</a></td>
  </tr>
  <tr>
    <td><a href="http://image.diku.dk/shark/sphinx_pages/build/html/index.html"><img width=100 src="http://image.diku.dk/shark/sphinx_pages/build/html/_static/SharkLogo.png"><br>Shark</a></td>
    <td><b>Supervised:</b> LDA, Fisher–LDA, Linear regression, SVMs, FF NN, RNN, Radial basis function networks, Regularization networks as well as Gaussian processes for regression, Iterative nearest neighbor classification and regression, Decision trees and random forests
<br><b>Unsupervised:</b> [PCA], Restricted Boltzmann machines, Hierarchical clustering, Data structures for efficient distance-based clustering
<br><b>Optimization:</b> Evolutionary algorithms, Single-objective optimization (e.g., CMA–ES), Multi-objective optimization, Basic linear algebra and optimization algorithms</td> 
    <td>C++</td> 
    <td>GNU LGPL</td>
    <td><a href="https://github.com/lemire/lbimproved">GitHub</a> </td>
    <td><a href="https://cocoapods.org/pods/Shark-SDK">Cocoa Pods</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/yconst/YCML"><img width=100 src="https://raw.githubusercontent.com/yconst/YCML/master/Logo.png"><br>YCML</a></td>
    <td>Gradient Descent Backpropagation, Resilient Backpropagation (RProp), Extreme Learning Machines (ELM), Forward Selection using Orthogonal Least Squares (for RBF Net), also with the PRESS statistic, Binary Restricted Boltzmann Machines (CD & PCD) 
    <br><b>Optimization algorithms</b>: Gradient Descent (Single-Objective, Unconstrained), RProp Gradient Descent (Single-Objective, Unconstrained), NSGA-II (Multi-Objective, Constrained)</td> 
    <td>Objective-C</td> 
    <td>GNU GPL 3.0</td>
    <td><a href="https://github.com/yconst/ycml/">GitHub</a> </td>
    <td> </td>
  </tr>
  <tr>
    <td><a href="https://github.com/Kalvar"><img src="https://avatars2.githubusercontent.com/u/1835631?v=4&s=460"><br>Kalvar Lin's libraries</a></td>
    <td>
    <ul>
<li><a href="https://github.com/Kalvar/ios-KRHebbian-Algorithm">ios-KRHebbian-Algorithm</a> - <a href="https://en.wikipedia.org/wiki/Hebbian_theory">Hebbian Theory</a></li>
<li><a href="https://github.com/Kalvar/ios-KRKmeans-Algorithm">ios-KRKmeans-Algorithm</a> - <a href="https://en.wikipedia.org/wiki/K-means_clustering">K-Means</a> clustering method.</li>
<li><a href="https://github.com/Kalvar/ios-KRFuzzyCMeans-Algorithm">ios-KRFuzzyCMeans-Algorithm</a> - <a href="https://en.wikipedia.org/wiki/Fuzzy_clustering">Fuzzy C-Means</a>, the fuzzy clustering algorithm.</li>
<li><a href="https://github.com/Kalvar/ios-KRGreyTheory">ios-KRGreyTheory</a> - <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.678.3477&amp;rep=rep1&amp;type=pdf">Grey Theory</a> / <a href="http://www.mecha.ee.boun.edu.tr/Prof.%20Dr.%20Okyay%20Kaynak%20Publications/c%20Journal%20Papers(appearing%20in%20SCI%20or%20SCIE%20or%20CompuMath)/62.pdf">Grey system theory-based models in time series prediction</a></li>
<li><a href="https://github.com/Kalvar/ios-KRSVM">ios-KRSVM</a> - Support Vector Machine and SMO.</li>
<li><a href="https://github.com/Kalvar/ios-KRKNN">ios-KRKNN</a> - kNN implementation.</li>
<li><a href="https://github.com/Kalvar/ios-KRRBFNN">ios-KRRBFNN</a> - Radial basis function neural network and OLS.</li>
</ul> 
</td> 
    <td>Objective-C</td> 
    <td>MIT</td>
    <td><a href="https://github.com/Kalvar">GitHub</a></td>
    <td></td>
  </tr>
</table>

**Multilayer perceptron implementations:**

- [Brain.js](https://github.com/harthur/brain) - JS
- [SNNeuralNet](https://github.com/devongovett/SNNeuralNet) - Objective-C port of brain.js
- [MLPNeuralNet](https://github.com/nikolaypavlov/MLPNeuralNet) - Objective-C, Accelerate
- [Swift-AI](https://github.com/Swift-AI/Swift-AI) - Swift
- [SwiftSimpleNeuralNetwork](https://github.com/davecom/SwiftSimpleNeuralNetwork) - Swift
- <a href="https://github.com/Kalvar/ios-BPN-NeuralNetwork">ios-BPN-NeuralNetwork</a> - Objective-C
- <a href="https://github.com/Kalvar/ios-Multi-Perceptron-NeuralNetwork">ios-Multi-Perceptron-NeuralNetwork</a>- Objective-C
- <a href="https://github.com/Kalvar/ios-KRDelta">ios-KRDelta</a> - Objective-C
- [ios-KRPerceptron](https://github.com/Kalvar/ios-KRPerceptron) - Objective-C

# <a name="dll"/>Deep Learning Libraries: 

### On-Device training and inference

* [Birdbrain](https://github.com/jordenhill/Birdbrain) - RNNs and FF NNs on top of Metal and Accelerate. Not ready for production.
* [BrainCore](https://github.com/aleph7/BrainCore) - simple but fast neural network framework written in Swift. It uses Metal framework to be as fast as possible. ReLU, LSTM, L2 ...
* [Caffe](http://caffe.berkeleyvision.org) - A deep learning framework developed with cleanliness, readability, and speed in mind. [GitHub](https://github.com/BVLC/caffe). [BSD]
	* [iOS port](https://github.com/aleph7/caffe)
	* [caffe-mobile](https://github.com/solrex/caffe-mobile) - another iOS port.
	* C++ examples: [Classifying ImageNet](http://caffe.berkeleyvision.org/gathered/examples/cpp_classification.html), [Extracting Features](http://caffe.berkeleyvision.org/gathered/examples/feature_extraction.html)
	* [Caffe iOS sample](https://github.com/noradaiko/caffe-ios-sample)
* [Caffe2](https://caffe2.ai/) - a cross-platform framework made with expression, speed, and modularity in mind.
	* [Cocoa Pod](https://github.com/RobertBiehl/caffe2-ios) 
	* [iOS demo app](https://github.com/KleinYuan/Caffe2-iOS)
* [Convnet.js](http://cs.stanford.edu/people/karpathy/convnetjs/) - ConvNetJS is a Javascript library for training Deep Learning models by [Andrej Karpathy](https://twitter.com/karpathy). [GitHub](https://github.com/karpathy/convnetjs)
	* [ConvNetSwift](https://github.com/alexsosn/ConvNetSwift) - Swift port [work in progress].
* [Deep Belief SDK](https://github.com/jetpacapp/DeepBeliefSDK) -  The SDK for Jetpac's iOS Deep Belief image recognition framework
* [TensorFlow](http://www.tensorflow.org/) - an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API.
	* [iOS examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/ios_examples)
	* [another example](https://github.com/hollance/TensorFlow-iOS-Example)
	* [Perfect-TensorFlow](https://github.com/PerfectlySoft/Perfect-TensorFlow) - TensorFlow binding for [Perfect](http://perfect.org/) (server-side Swift framework). Includes only C TF API.
* [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn) - header only, dependency-free deep learning framework in C++11.
	* [iOS example](https://github.com/tiny-dnn/tiny-dnn/tree/d4fff53fa0d01f59eb162de2ec32c652a1f6f467/examples/ios) 
* [Torch](http://torch.ch/) is a scientific computing framework with wide support for machine learning algorithms.
	* [Torch4iOS](https://github.com/jhondge/torch4ios)
	* [Torch-iOS](https://github.com/clementfarabet/torch-ios)

### Deep Learning: Running pre-trained models on device

These libraries doesn't support training, so you need to pre-train models in some ML framework.

* [Bender](https://github.com/xmartlabs/Bender) - Framework for building fast NNs. Supports TensorFlow models. It uses Metal under the hood.
* [Core ML](#coreml)
* [DeepLearningKit](http://deeplearningkit.org/) - Open Source Deep Learning Framework from Memkite for Apple's tvOS, iOS and OS X.
* [Espresso](https://github.com/codinfox/espresso) - A minimal high performance parallel neural network framework running on iOS.
* [Forge](https://github.com/hollance/Forge) - A neural network toolkit for Metal.
* [Keras.js](https://transcranial.github.io/keras-js/#/) - run [Keras](https://keras.io/) models in a web view. 
* [KSJNeuralNetwork](https://github.com/woffle/KSJNeuralNetwork) - A Neural Network Inference Library Built atop BNNS and MPS
	* [Converter for Torch models](https://github.com/woffle/torch2ios)
* [MXNet](https://mxnet.incubator.apache.org/) - MXNet is a deep learning framework designed for both efficiency and flexibility.
	* [Deploying pre-trained mxnet model to a smartphone](https://mxnet.incubator.apache.org/how_to/smart_device.html)
* [Quantized-CNN](https://github.com/jiaxiang-wu/quantized-cnn) - compressed convolutional neural networks for Mobile Devices
* [WebDNN](https://mil-tokyo.github.io/webdnn/) - You can run deep learning model in a web view if you want. Three modes: WebGPU acceleration, WebAssembly acceleration and pure JS (on CPU). No training, inference only.

### Deep Learning: Low-level routines libraries

* [BNNS](https://developer.apple.com/reference/accelerate/1912851-bnns) - Apple Basic neural network subroutines (BNNS) is a collection of functions that you use to implement and run neural networks, using previously obtained training data.
	* [BNNS usage examples](https://github.com/shu223/iOS-10-Sampler) in iOS 10 sampler.
	* [An example](https://github.com/bignerdranch/bnns-cocoa-example) of a neural network trained by tensorflow and executed using BNNS
* [MetalPerformanceShaders](https://developer.apple.com/reference/metalperformanceshaders) - CNNs on GPU from Apple.
	* [MetalCNNWeights](https://github.com/kakugawa/MetalCNNWeights) - a Python script to convert Inception v3 for MPS.
	* [MPSCNNfeeder](https://github.com/kazoo-kmt/MPSCNNfeeder) - Keras to MPS models conversion.
* [NNPACK](https://github.com/Maratyszcza/NNPACK) - Acceleration package for neural networks on multi-core CPUs. Prisma [uses](http://prisma-ai.com/libraries.html) this library in the mobile app.
* [STEM](https://github.com/abeschneider/stem) - Swift Tensor Engine for Machine-learning
	* [Documentation](http://stem.readthedocs.io/en/latest/) 

### <a name="dlmc"/>Deep Learning: Model Compression

* TensorFlow implementation of [knowledge distilling](https://github.com/chengshengchan/model_compression) method
* [MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe) - Caffe Implementation of Google's MobileNets


# <a name="cv"/>Computer Vision


* [ccv](http://libccv.org) - C-based/Cached/Core Computer Vision Library, A Modern Computer Vision Library
	* [iOS demo app](https://github.com/liuliu/klaus)
* [OpenCV](http://opencv.org) – Open Source Computer Vision Library. [BSD]
	* [OpenCV crash course](http://www.pyimagesearch.com/free-opencv-crash-course/) 
	* [OpenCVSwiftStitch](https://github.com/foundry/OpenCVSwiftStitch)
	* [Tutorial: using and building openCV on iOS devices](http://maniacdev.com/2011/07/tutorial-using-and-building-opencv-open-computer-vision-on-ios-devices)
	* [A Collection of OpenCV Samples For iOS](https://github.com/woffle/OpenCV-iOS-Demos)
* [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) – a state-of-the art open source tool intended for facial landmark detection, head pose estimation, facial action unit recognition, and eye-gaze estimation.
	* [iOS port](https://github.com/FaceAR/OpenFaceIOS)
	* [iOS demo](https://github.com/FaceAR/OpenFaceIOS)
* [trackingjs](http://trackingjs.com/) – Object tracking in JS
* [Vision](https://developer.apple.com/documentation/vision) is an Apple framework for computer vision.

# <a name="nlp"/>Natural Language Processing


* [CoreLinguistics](https://github.com/rxwei/CoreLinguistics) - POS tagging (HMM), ngrams, Naive Bayes, IBM alignment models.
* [GloVe](https://github.com/rxwei/GloVe-swift) Swift package. Vector words representations.
* [NSLinguisticTagger](http://nshipster.com/nslinguistictagger/)
* [Parsimmon](https://github.com/ayanonagon/Parsimmon)
* [Twitter text](https://github.com/twitter/twitter-text-objc) - 
An Objective-C implementation of Twitter's text processing library. The library includes methods for extracting user names, mentions headers, hashtags, and more – all the tweet specific language syntax you could ever want.
* [Verbal expressions for Swift](https://github.com/VerbalExpressions/SwiftVerbalExpressions), like regexps for humans.
* [Word2Vec](https://code.google.com/p/word2vec/) - Original C implementation of Word2Vec Deep Learning algorithm. Works on iPhone like a charm.

# <a name="tts"/>Speech Recognition (TTS) and Generation (STT)


* [Kaldi-iOS framework](http://keenresearch.com/) - on-device speech recognition using deep learning.
	* [Proof of concept app](https://github.com/keenresearch/kaldi-ios-poc)
* [MVSpeechSynthesizer](https://github.com/vimalmurugan89/MVSpeechSynthesizer)
* [OpenEars™: free speech recognition and speech synthesis for the iPhone](http://www.politepix.com/openears/) - OpenEars™ makes it simple for you to add offline speech recognition and synthesized speech/TTS to your iPhone app quickly and easily. It lets everyone get the great results of using advanced speech UI concepts like statistical language models and finite state grammars in their app, but with no more effort than creating an NSArray or NSDictionary. 
	* [Tutorial (Russian)](http://habrahabr.ru/post/237589/)
* [TLSphinx](https://github.com/tryolabs/TLSphinx), [Tutorial](http://blog.tryolabs.com/2015/06/15/tlsphinx-automatic-speech-recognition-asr-in-swift/)

# <a name="ocr"/>Text Recognition (OCR)


* [ocrad.js](https://github.com/antimatter15/ocrad.js) - JS OCR
* **Tesseract**
	* [Install and Use Tesseract on iOS](http://lois.di-qual.net/blog/install-and-use-tesseract-on-ios-with-tesseract-ios/)
	* [tesseract-ios-lib](https://github.com/ldiqual/tesseract-ios-lib)
	* [tesseract-ios](https://github.com/ldiqual/tesseract-ios)
	* [Tesseract-OCR-iOS](https://github.com/gali8/Tesseract-OCR-iOS)
	* [OCR-iOS-Example](https://github.com/robmathews/OCR-iOS-Example)

# <a name="ai"/>Other AI


* [Axiomatic](https://github.com/JadenGeller/Axiomatic) - Swift unification framework for logic programming.
* [Build Your Own Lisp In Swift](https://github.com/hollance/BuildYourOwnLispInSwift)
* [Logician](https://github.com/mdiep/Logician) - Logic programming in Swift
* [Swiftlog](https://github.com/JadenGeller/Swiftlog) - A simple Prolog-like language implemented entirely in Swift.

# <a name="web"/>Machine Learning Web APIs


* [**IBM** Watson](http://www.ibm.com/smarterplanet/us/en/ibmwatson/developercloud/) - Enable Cognitive Computing Features In Your App Using IBM Watson's Language, Vision, Speech and Data APIs.
	* [Introducing the (beta) IBM Watson iOS SDK](https://developer.ibm.com/swift/2015/12/18/introducing-the-new-watson-sdk-for-ios-beta/)
* [AlchemyAPI](http://www.alchemyapi.com/) - Semantic Text Analysis APIs Using Natural Language Processing. Now part of IBM Watson.
* [**Microsoft** Project Oxford](https://www.projectoxford.ai/)
* [**Google** Prediction engine](https://cloud.google.com/prediction/docs)
	* [Objective-C API](https://code.google.com/p/google-api-objectivec-client/wiki/Introduction)
* [Google Translate API](https://cloud.google.com/translate/docs)
* [Google Cloud Vision API](https://cloud.google.com/vision/)
* [**Amazon** Machine Learning](http://aws.amazon.com/documentation/machine-learning/) - Amazon ML is a cloud-based service for developers. It provides visualization tools to create machine learning models. Obtain predictions for application using APIs. 
	* [iOS developer guide](https://docs.aws.amazon.com/mobile/sdkforios/developerguide/getting-started-machine-learning.html).
	* [iOS SDK](https://github.com/aws/aws-sdk-ios)
* [**PredictionIO**](https://prediction.io/) - opensource machine learning server for developers and ML engineers. Built on Apache Spark, HBase and Spray.
	* [Swift SDK](https://github.com/minhtule/PredictionIO-Swift-SDK)
	* [Tapster iOS Demo](https://github.com/minhtule/Tapster-iOS-Demo) - This demo demonstrates how to use the PredictionIO Swift SDK to integrate an iOS app with a PredictionIO engine to make your mobile app more interesting.
	* [Tutorial](https://github.com/minhtule/Tapster-iOS-Demo/blob/master/TUTORIAL.md) on using Swift with PredictionIO.
* [**Wit.AI**](https://wit.ai/) - NLP API
* [**Yandex** SpeechKit](https://tech.yandex.com/speechkit/mobilesdk/) Text-to-speech and speech-to-text for Russian language. iOS SDK available.
* [**Abbyy** OCR SDK](http://www.abbyy.com/mobile-ocr/iphone-ocr/)
* [**Clarifai**](http://www.clarifai.com/#) - deep learning web api for image captioning. [iOS starter project](https://github.com/Clarifai/clarifai-ios-starter)
* [**MetaMind**](https://www.metamind.io/) - deep learning web api for image captioning.
* [Api.AI](https://api.ai/) - Build intelligent speech interfaces
for apps, devices, and web
* [**CloudSight.ai**](https://cloudsight.ai/) - deep learning web API for fine grained object detection or whole screen description, including natural language object captions. [Objective-C](https://github.com/cloudsight/cloudsight-objc) API client is available.

# <a name="mlapps"/>Opensource ML Applications


### Deep Learning

* [DeepDreamer](https://github.com/johndpope/deepdreamer) - Deep Dream application
* [DeepDreamApp](https://github.com/johndpope/DeepDreamApp) - Deep Dream Cordova app.
* [Texture Networks](https://github.com/DmitryUlyanov/texture_nets), Lua implementation
* [Feedforward style transfer](https://github.com/jcjohnson/fast-neural-style), Lua implementation
* [TensorFlow implementation of Neural Style](https://github.com/cysmith/neural-style-tf)
* [Corrosion detection app](https://github.com/jmolayem/corrosionapp)
* [ios_camera_object_detection](https://github.com/yjmade/ios_camera_object_detection) - Realtime mobile visualize based Object Detection based on TensorFlow and YOLO model
* [TensorFlow MNIST iOS demo](https://github.com/mattrajca/MNIST) - Getting Started with Deep MNIST and TensorFlow on iOS
* [Drummer App](https://github.com/hollance/RNN-Drummer-Swift) with RNN and Swift
* [What'sThis](https://github.com/pppoe/WhatsThis-iOS)
* [enVision](https://github.com/IDLabs-Gate/enVision) - Deep Learning Models for Vision Tasks on iOS\
* [GoogLeNet on iOS demo](https://github.com/krasin/MetalDetector)
* [Neural style in Android](https://github.com/naman14/Arcade)
* [mnist-bnns](https://github.com/paiv/mnist-bnns) - TensorFlow MNIST demo port to BNNS
* [Benchmark of BNNS vs. MPS](https://github.com/hollance/BNNS-vs-MPSCNN)
* [VGGNet on Metal](https://github.com/hollance/VGGNet-Metal)
* A [Sudoku Solver](https://github.com/waitingcheung/deep-sudoku-solver) that leverages TensorFlow and iOS BNNS for deep learning.
* [HED CoreML Implementation](https://github.com/s1ddok/HED-CoreML) is a demo with tutorial on how to use Holistically-Nested Edge Detection on iOS with CoreML and Swift

### Traditional Computer Vision

* [SwiftOCR](https://github.com/garnele007/SwiftOCR)
* [GrabCutIOS](https://github.com/naver/grabcutios) - Image segmentation using GrabCut algorithm for iOS

### NLP

* [Classical ELIZA chatbot in Swift](https://gist.github.com/hollance/be70d0d7952066cb3160d36f33e5636f)
* [InfiniteMonkeys](https://github.com/craigomac/InfiniteMonkeys) - A Keras-trained RNN to emulate the works of a famous poet, powered by BrainCore

### Other

* [Swift implementation of Joel Grus's "Data Science from Scratch"](https://github.com/graceavery/LearningMachineLearning)
* [Neural Network built in Apple Playground using Swift](https://github.com/Luubra/EmojiIntelligence)

# <a name="gameai"/>Game AI


* [Introduction to AI Programming for Games](http://www.raywenderlich.com/24824/introduction-to-ai-programming-for-games)
* [dlib](http://dlib.net/) is a library which has many useful tools including machine learning.
* [MicroPather](http://www.grinninglizard.com/MicroPather/) is a path finder and A* solver (astar or a-star) written in platform independent C++ that can be easily integrated into existing code.
* Here is a [list](http://www.ogre3d.org/tikiwiki/List+Of+Libraries#Artificial_intelligence) of some AI libraries suggested on OGRE3D website. Seems they are mostly written in C++.
* [GameplayKit Programming Guide](https://developer.apple.com/library/content/documentation/General/Conceptual/GameplayKit_Guide/)

# Other related staff

### <a name="la"/>Linear algebra


* [Accelerate-in-Swift](https://github.com/hyperjeff/Accelerate-in-Swift) - Swift example codes for the Accelerate.framework
* [cuda-swift](https://github.com/rxwei/cuda-swift) - Swift binding to CUDA. Not iOS, but still interesting.
* [Dimensional](https://github.com/JadenGeller/Dimensional) - Swift matrices with friendly semantics and a familiar interface.
* [Eigen](http://eigen.tuxfamily.org/) - A high-level C++ library of template headers for linear algebra, matrix and vector operations, numerical solvers and related algorithms. [MPL2]
* [Matrix](https://github.com/hollance/Matrix) - convenient matrix type with different types of subscripts, custom operators and predefined matrices. A fork of Surge.
* [NDArray](https://github.com/t-ae/ndarray) - Float library for Swift, accelerated with Accelerate Framework.
* [Swift-MathEagle](https://github.com/rugheid/Swift-MathEagle) - A general math framework to make using math easy. Currently supports function solving and optimisation, matrix and vector algebra, complex numbers, big int, big frac, big rational, graphs and general handy extensions and functions.
* [SwiftNum](https://github.com/donald-pinckney/SwiftNum) - linear algebra, fft, gradient descent, conjugate GD, plotting.
* [Swix](https://github.com/scottsievert/swix) - Swift implementation of NumPy and OpenCV wrapper.
* [Surge](https://github.com/mattt/Surge) from Mattt
* [Upsurge](https://github.com/aleph7/Upsurge) - generic tensors, matrices on top of Accelerate. A fork of Surge.
* [YCMatrix](https://github.com/yconst/YCMatrix) - A flexible Matrix library for Objective-C and Swift (OS X / iOS)

### <a name="stat"/>Statistics, random numbers


* [SigmaSwiftStatistics](https://github.com/evgenyneu/SigmaSwiftStatistics) - A collection of functions for statistical calculation written in Swift.
* [SORandom](https://github.com/SebastianOsinski/SORandom) - Collection of functions for generating psuedorandom variables from various distributions
* [RandKit](https://github.com/aidangomez/RandKit) - Swift framework for random numbers & distributions.


### <a name="mo"/>Mathematical optimization


* [fmincg-c](https://github.com/gautambhatrcb/fmincg-c) - Conjugate gradient implementation in C
* [libLBFGS](https://github.com/chokkan/liblbfgs) - a C library of Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
* [SwiftOptimizer](https://github.com/haginile/SwiftOptimizer) - QuantLib Swift port.

### <a name="fe"/>Feature extraction


* [IntuneFeatures](https://github.com/venturemedia/intune-features) framework contains code to generate features from audio files and feature labels from the respective MIDI files.
* [matchbox](https://github.com/hfink/matchbox) - Mel-Frequency-Cepstral-Coefficients and Dynamic-Time-Warping for iOS/OSX. **Warning: the library was updated last time when iOS 4 was still hot.**
* [LibXtract](https://github.com/jamiebullock/LibXtract) is a simple, portable, lightweight library of audio feature extraction functions.

### <a name="dv"/>Data Visualization


* [Charts](https://github.com/danielgindi/Charts) - The Swift port of the MPAndroidChart.
* [iOS-Charts](https://github.com/danielgindi/ios-charts)
* [Core Plot](https://github.com/core-plot/core-plot)
* [Awesome iOS charts](https://github.com/sxyx2008/awesome-ios-chart)
* [JTChartView](https://github.com/kubatru/JTChartView)
* [VTK](http://www.vtk.org/gallery/)
	* [VTK in action](http://www.vtk.org/vtk-in-action/)
* [D3.js iOS binding](https://github.com/lee-leonardo/iOS-D3) 

### <a name="bio"/>Bioinformatics (kinda)


* [BioJS](http://biojs.net/) - a set of tools for bioinformatics in the browser. BioJS builds a infrastructure, guidelines and tools to avoid the reinvention of the wheel in life sciences. Community builds modules than can be reused by anyone.
* [BioCocoa](http://www.bioinformatics.org/biococoa/wiki/pmwiki.php) - BioCocoa is an open source OpenStep (GNUstep/Cocoa) framework for bioinformatics written in Objective-C. [Dead project].
* [iBio](https://github.com/Lizhen0909/iBio) - A Bioinformatics App for iPhone.

### <a name="bd"/>Big Data (not really)


* [HDF5Kit](https://github.com/aleph7/HDF5Kit) - This is a Swift wrapper for the HDF5 file format. HDF5 is used in the scientific comunity for managing large volumes of data. The objective is to make it easy to read and write HDF5 files from Swift, including playgrounds.

### <a name="ip"/>IPython + Swift


* [iSwift](https://github.com/KelvinJin/iSwift) - Swift kernel for IPython notebook.

# <a name="blogs"/>iOS ML Blogs


### Regular mobile ML

* **[Pete Warden's blog](https://petewarden.com/)**
	* [How to Quantize Neural Networks with TensorFlow](https://petewarden.com/2016/05/03/how-to-quantize-neural-networks-with-tensorflow/)
* **[The "Machine, think!" blog](http://machinethink.net/blog/) by Matthijs Hollemans**
	* [The “hello world” of neural networks](http://matthijshollemans.com/2016/08/24/neural-network-hello-world/) - Swift and BNNS
	* [Convolutional neural networks on the iPhone with VGGNet](http://matthijshollemans.com/2016/08/30/vggnet-convolutional-neural-network-iphone/)

### Accidental mobile ML

* **[Invasive Code](https://www.invasivecode.com/weblog/) blog**
	* [Machine Learning for iOS](https://www.invasivecode.com/weblog/machine-learning-swift-ios/)
	* [Convolutional Neural Networks in iOS 10 and macOS](https://www.invasivecode.com/weblog/convolutional-neural-networks-ios-10-macos-sierra/)
* **Big Nerd Ranch** - [Use TensorFlow and BNNS to Add Machine Learning to your Mac or iOS App](https://www.bignerdranch.com/blog/use-tensorflow-and-bnns-to-add-machine-learning-to-your-mac-or-ios-app/)

### Other

* [Intelligence in Mobile Applications](https://medium.com/@sadmansamee/intelligence-in-mobile-applications-ca3be3c0e773#.lgk2gt6ik)
* [An exclusive inside look at how artificial intelligence and machine learning work at Apple](https://backchannel.com/an-exclusive-look-at-how-ai-and-machine-learning-work-at-apple-8dbfb131932b)
* [Presentation on squeezing DNNs for mobile](https://www.slideshare.net/mobile/anirudhkoul/squeezing-deep-learning-into-mobile-phones)
* [Curated list of papers on deep learning models compression and acceleration](https://handong1587.github.io/deep_learning/2015/10/09/acceleration-model-compression.html)

# <a name="gpublogs"/>GPU Computing Blogs


* [OpenCL for iOS](https://github.com/linusyang/opencl-test-ios) - just a test.
* Exploring GPGPU on iOS. 
	* [Article](http://ciechanowski.me/blog/2014/01/05/exploring_gpgpu_on_ios/) 
	* [Code](https://github.com/Ciechan/Exploring-GPGPU-on-iOS)

* GPU-accelerated video processing for Mac and iOS. [Article](http://www.sunsetlakesoftware.com/2010/10/22/gpu-accelerated-video-processing-mac-and-ios0).

* [Concurrency and OpenGL ES](https://developer.apple.com/library/ios/documentation/3ddrawing/conceptual/opengles_programmingguide/ConcurrencyandOpenGLES/ConcurrencyandOpenGLES.html) - Apple programming guide.

* [OpenCV on iOS GPU usage](http://stackoverflow.com/questions/10704916/opencv-on-ios-gpu-usage) - SO discussion.

### Metal

* Simon's Gladman \(aka flexmonkey\) [blog](http://flexmonkey.blogspot.com/)
	* [Talk on iOS GPU programming](https://realm.io/news/altconf-simon-gladman-ios-gpu-programming-with-swift-metal/) with Swift and Metal at Realm Altconf.
	* [The Supercomputer In Your Pocket:
Metal & Swift](https://realm.io/news/swift-summit-simon-gladman-metal/) - a video from the Swift Summit Conference 2015
	* https://github.com/FlexMonkey/MetalReactionDiffusion
	* https://github.com/FlexMonkey/ParticleLab
* [Memkite blog](http://memkite.com/) - startup intended to create deep learning library for iOS.
	* [Swift and Metal example for General Purpose GPU Processing on Apple TVOS 9.0](https://github.com/memkite/MetalForTVOS)
	* [Data Parallel Processing with Swift and Metal on GPU for iOS8](https://github.com/memkite/SwiftMetalGPUParallelProcessing)
	* [Example of Sharing Memory between GPU and CPU with Swift and Metal for iOS8](http://memkite.com/blog/2014/12/30/example-of-sharing-memory-between-gpu-and-cpu-with-swift-and-metal-for-ios8/)
* [Metal by Example blog](http://metalbyexample.com/)
* [objc-io article on Metal](https://www.objc.io/issues/18-games/metal/)

# <a name="books"/>Mobile ML Books

* <b>Building Mobile Applications with TensorFlow</b> by Pete Warden. [Book page](http://www.oreilly.com/data/free/building-mobile-applications-with-tensorflow.csp). <b>[Free download](http://www.oreilly.com/data/free/building-mobile-applications-with-tensorflow.csp?download=true)</b>

# <a name="learn"/>Learn Machine Learning

<i>Please note that in this section, I'm not trying to collect another list of ALL machine learning study resources, but only composing a list of things that I found useful.</i>

* <b>[Academic Torrents](http://academictorrents.com/browse.php?cat=7)</b>. Sometimes awesome courses or datasets got deleted from their sites. But this doesn't mean, that they are lost.
* [Arxiv Sanity Preserver](http://www.arxiv-sanity.com/) - a tool to keep pace with the ML research progress.

## Free Books

* Immersive Linear Algebra [interactive book](http://immersivemath.com/ila/index.html) by J. Ström, K. Åström, and T. Akenine-Möller.
* ["Natural Language Processing with Python"](http://www.nltk.org/book/) - free online book.
* [Probabilistic Programming & Bayesian Methods for Hackers](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/) - An intro to Bayesian methods and probabilistic programming from a computation/understanding-first, mathematics-second point of view. 
* ["Deep learning"](http://www.deeplearningbook.org/) - the book by Ian Goodfellow and Yoshua Bengio and Aaron Courville

## Free Courses

* [Original Machine Learning Coursera course](https://www.coursera.org/learn/machine-learning/home/info) by Andrew Ng.
* [Machine learning playlist on Youtube](https://www.youtube.com/playlist?list=PLD0F06AA0D2E8FFBA).
* Free online interactive book ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/).
* [Heterogeneous Parallel Programming](https://www.coursera.org/course/hetero) course.
* [Deep Learning for Perception](https://computing.ece.vt.edu/~f15ece6504/) by Virginia Tech, Electrical and Computer Engineering, Fall 2015: ECE 6504
* [CAP 5415 - Computer Vision](http://crcv.ucf.edu/courses/CAP5415/Fall2014/index.php) by UCF
* [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/syllabus.html) by Stanford
* [Machine Learning: 2014-2015 Course materials](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/) by Oxford
* [Stanford CS class CS231n: Convolutional Neural Networks for Visual Recognition.](http://cs231n.stanford.edu/)
* [Deep Learning for Natural Language Processing \(without Magic\)](http://nlp.stanford.edu/courses/NAACL2013/)
* [Videos](http://videolectures.net/deeplearning2015_montreal/) from Deep Learning Summer School, Montreal 2015.
* [Deep Learning Summer School, Montreal 2016](http://videolectures.net/deeplearning2016_montreal/)


# <a name="lists"/>Other Lists


* [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning)
* [Machine Learning Courses](https://github.com/prakhar1989/awesome-courses#machine-learning)
* [Awesome Data Science](https://github.com/okulbilisim/awesome-datascience)
* [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision)
* [Speech and language processing](https://github.com/edobashira/speech-language-processing)
* [The Rise of Chat Bots:](https://stanfy.com/blog/the-rise-of-chat-bots-useful-links-articles-libraries-and-platforms/)  Useful Links, Articles, Libraries and Platforms by Pavlo Bashmakov.
* [Awesome Machine Learning for Cyber Security](https://github.com/jivoi/awesome-ml-for-cybersecurity)

[DTW]: https://en.wikipedia.org/wiki/Dynamic_time_warping
[ANN]: https://en.wikipedia.org/wiki/Artificial_neural_network
[FF NN]: https://en.wikipedia.org/wiki/Feedforward_neural_network
[k-NN]: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
[LDA]: https://en.wikipedia.org/wiki/Linear_discriminant_analysis
[SVM]: https://en.wikipedia.org/wiki/Support_vector_machine
[RNN]: https://en.wikipedia.org/wiki/Recurrent_neural_network
[PCA]: https://en.wikipedia.org/wiki/Principal_component_analysis
